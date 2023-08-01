import os
import random
import logging
import pprint
from copy import deepcopy
from collections import defaultdict

import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points, draw_ordermap
from isegm.utils.misc import save_checkpoint
from isegm.utils.serialization import get_config_repr
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from .optimizer import get_optimizer, get_optimizer_with_layerwise_decay
from isegm.model.losses import FocalLoss

import math
from torchvision.utils import save_image

class ISTrainer(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg,
                 trainset, valset,
                 optimizer='adam',
                 optimizer_params=None,
                 layerwise_decay=False,
                 image_dump_interval=200,
                 checkpoint_interval=10,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 net_inputs=('images', 'points'),
                 max_num_next_clicks=0,
                 click_models=None,
                 prev_mask_drop_prob=0.0,
                 use_iterloss=False,
                 iterloss_weights=None,
                 use_random_clicks=True,
                 ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks

        # iterloss
        self.use_iterloss = use_iterloss
        self.iterloss_weights = iterloss_weights
        self.use_random_clicks = use_random_clicks

        self.click_models = click_models
        self.prev_mask_drop_prob = prev_mask_drop_prob

        if cfg.distributed:
            cfg.batch_size //= cfg.ngpus
            cfg.val_batch_size //= cfg.ngpus

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None

        self.trainset = trainset
        self.valset = valset

        logger.info(f'Dataset of {trainset.get_samples_number()} samples was loaded for training.')
        logger.info(f'Dataset of {valset.get_samples_number()} samples was loaded for validation.')

        self.train_data = DataLoader(
            trainset, cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=True, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        self.val_data = DataLoader(
            valset, cfg.val_batch_size,
            sampler=get_sampler(valset, shuffle=False, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        if layerwise_decay:
            self.optim = get_optimizer_with_layerwise_decay(model, optimizer, optimizer_params)
        else:
            self.optim = get_optimizer(model, optimizer, optimizer_params)
        model = self._load_weights(model)

        if cfg.multi_gpu:
            model = get_dp_wrapper(cfg.distributed)(model, device_ids=cfg.gpu_ids,
                                                    output_device=cfg.gpu_ids[0])

        if self.is_master:
            logger.info(model)
            logger.info(get_config_repr(model._config))
            logger.info('Run experiment with config:')
            logger.info(pprint.pformat(cfg, indent=4))

        self.device = cfg.device
        self.net = model.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        if self.click_models is not None:
            for click_model in self.click_models:
                for param in click_model.parameters():
                    param.requires_grad = False
                click_model.to(self.device)
                click_model.eval()

        self.scaler: torch.cuda.amp.GradScaler
        self.focal_loss = torch.hub.load(
                            'adeelh/pytorch-multi-class-focal-loss',
                            model='FocalLoss',
                            alpha=None,
                            gamma=2,
                            reduction='mean',
                            force_reload=False
                        )

    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        logger.info(f'Starting Epoch: {start_epoch}')
        logger.info(f'Total Epochs: {num_epochs}')

        if self.cfg.amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(start_epoch, num_epochs):
            self.training(epoch)
            if validation:
                self.validation(epoch)
            

    def training(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        if self.cfg.distributed:
            self.train_data.sampler.set_epoch(epoch)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100, ascii=True, desc='level_1', position=0)\
            if self.is_master else self.train_data

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        train_loss = 0.0
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i
            if self.cfg.amp:
                with torch.cuda.amp.autocast():
                    loss, losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, False, epoch)
            else:
                loss, losses_logging, splitted_batch_data, outputs = \
                    self.batch_forward(batch_data, False, epoch)

            accumulate_grad = ((i + 1) % self.cfg.accumulate_grad == 0) or \
                (i + 1 == len(self.train_data))

            if self.cfg.amp:
                loss /= self.cfg.accumulate_grad
                self.scaler.scale(loss).backward()
                if accumulate_grad:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.optim.zero_grad()
            else:
                loss.backward()
                if accumulate_grad:
                    self.optim.step()
                    self.optim.zero_grad()

            losses_logging['overall'] = loss
            reduce_loss_dict(losses_logging)

            train_loss += losses_logging['overall'].item()

            if self.is_master:
                for loss_name, loss_value in losses_logging.items():
                    self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                       value=loss_value.item(),
                                       global_step=global_step)

                for k, v in self.loss_cfg.items():
                    if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                        v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

                if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                    self.save_visualization(splitted_batch_data, outputs, global_step, prefix='train', from_logist=True)

                self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate',
                                   value=self.lr if not hasattr(self, 'lr_scheduler') else self.lr_scheduler.get_lr()[-1],
                                   global_step=global_step)

                tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.4f} {metric.get_epoch_value():.4f}')
                
                for metric in self.train_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for metric in self.train_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}',
                                   value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                            epoch=None, multi_gpu=self.cfg.multi_gpu)

            if isinstance(self.checkpoint_interval, (list, tuple)):
                checkpoint_interval = [x for x in self.checkpoint_interval if x[0] <= epoch][-1][1]
            else:
                checkpoint_interval = self.checkpoint_interval

            if epoch % checkpoint_interval == 0:
                save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                                epoch=epoch, multi_gpu=self.cfg.multi_gpu)

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

    def validation(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100) if self.is_master else self.val_data

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_data) + i
            loss, batch_losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, validation=True, epoch=epoch)

            batch_losses_logging['overall'] = loss
            reduce_loss_dict(batch_losses_logging)
            for loss_name, loss_value in batch_losses_logging.items():
                losses_logging[loss_name].append(loss_value.item())

            val_loss += batch_losses_logging['overall'].item()

            if self.is_master:
                tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss/(i + 1):.4f} {metric.get_epoch_value():.4f}')
                for metric in self.val_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                                   global_step=epoch, disable_avg=True)

            for metric in self.val_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

    def batch_forward(self, batch_data, validation=False, epoch=None):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data['points']
            # image = torch.load('/home/guavamin/CFR-ICL-Interactive-Segmentation/image.pt')
            # gt_mask = torch.load('/home/guavamin/CFR-ICL-Interactive-Segmentation/gt_mask.pt')
            # points = torch.load('/home/guavamin/CFR-ICL-Interactive-Segmentation/points.pt')
            
            # self.net.eval()
            
            if random.random() > 0.5:
                prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
                prev_order = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
            else:
                prev_output = torch.ones_like(image, dtype=torch.float32)[:, :1, :, :]
                prev_order = torch.ones_like(image, dtype=torch.float32)[:, :1, :, :]
            loss = 0.0
            # Initialize the previous output list and points
            prev_output_list = [prev_output]
            prev_order_gt_list = [prev_order]

            num_start = random.randint(1, 16)
            if not self.use_random_clicks:
                points[:] = -1

                points = get_next_points(prev_output,
                                         gt_mask,
                                         points)
                # with torch.set_grad_enabled(not validation):
                
                with torch.no_grad():
                    # image = torch.load('/home/guavamin/CFR-ICL-Interactive-Segmentation/image.pt')
                    # gt_mask = torch.load('/home/guavamin/CFR-ICL-Interactive-Segmentation/gt_mask.pt')
                    # points = torch.load('/home/guavamin/CFR-ICL-Interactive-Segmentation/points.pt')
                    # prev_output = torch.load('/home/guavamin/CFR-ICL-Interactive-Segmentation/prev_output.pt')
                    # prev_order = torch.load('/home/guavamin/CFR-ICL-Interactive-Segmentation/prev_order.pt')
                    for init_indx in range(num_start):
                        
                        # torch.save(image, 'image.pt')
                        # torch.save(gt_mask, 'gt_mask.pt')
                        # torch.save(points, 'points.pt')
                        # torch.save(prev_output, 'prev_output.pt')
                        # torch.save(prev_order, 'prev_order.pt')
                        # exit()
                        net_input = torch.cat((image, prev_output, prev_order), dim=1) \
                        if self.net.with_prev_mask else image
                        output = self._forward(self.net, net_input, points)
                        prev_output = torch.sigmoid(output['instances'])#output['instances']#
                        # save_image(prev_output, 'output_train%d.png' % init_indx)
                        # save_image(image, 'test.png')
                        # save_image(gt_mask, 'mask.png')     
                        # print(init_indx)
                        
                        
                        #new order
                        if len(prev_output_list) >= 1:
                            prev_mask_output = prev_output_list[-1].clone()
                            curr_mask_output = torch.sigmoid(output['instances'])# output['instances']#
                            curr_order_output = output['order']# torch.tanh(output['order'])#
                            prev_order_gt = prev_order_gt_list[-1].clone()
                            points_, points_order_ = torch.split(points.clone().view(-1, points.size(2)), [2, 1], dim=1)
                            order_gt = self.order_of_pixel(prev_mask_output, curr_mask_output, prev_order_gt, now_order=torch.max(points_order_))
                            order_embed_gt = order_gt #self.order_encoding_of_pixel(order_gt.clone(), embed_dim=1)
                            prev_order = order_embed_gt
                            prev_order_gt_list.append(order_gt)
                            error_order_gt = self.mark_error_and_modify_order(batch_data['instances'], prev_mask_output, curr_mask_output, prev_order_gt, \
                                                                              order_gt.clone().detach(), torch.max(points_order_))
                            
                            batch_data['order_GT'] = error_order_gt # order_gt

                        prev_output_list.append(torch.sigmoid(output['instances']))
                        # prev_output_list.append(output['instances'])
                        #new order
                        if init_indx <= num_start - 1:
                            points = get_next_points(prev_output,
                                                    gt_mask,
                                                    points)
                    # exit()
                    
            num_iters =  self.max_num_next_clicks#min(epoch+2, self.max_num_next_clicks)#random.randint(1, self.max_num_next_clicks)
            # Define the triplet loss function
            triplet_loss = torch.nn.TripletMarginLoss(margin=0.2)
            mse_loss = torch.nn.MSELoss()
            cosine_similarity = torch.nn.CosineSimilarity(dim=1)
            cross_entropy_loss = torch.nn.CrossEntropyLoss()
            
            
            if self.use_iterloss:
                # iterloss
                for click_indx in range(num_iters):

                    # v1
                    # net_input = torch.cat((image, prev_output), dim=1) \
                    #     if self.net.with_prev_mask else image
                    # v2
                    
                    net_input = torch.cat((image, prev_output, prev_order), dim=1) \
                        if self.net.with_prev_mask else image
                    output = self._forward(self.net, net_input, points)
                    loss = self.add_loss(
                        'instance_loss', loss, losses_logging, validation,
                        lambda: (output['instances'], batch_data['instances']),
                        iterloss_step=click_indx,
                        iterloss_weight=self.iterloss_weights[click_indx])
                    loss = self.add_loss(
                        'instance_aux_loss', loss, losses_logging, validation,
                        lambda: (output['instances'], batch_data['instances']),
                        iterloss_step=click_indx,
                        iterloss_weight=self.iterloss_weights[click_indx])

                    #v3
                    # loss += F.cross_entropy(output['instances'], batch_data['instances'])
                    # loss += F.binary_cross_entropy_with_logits(output['instances'], batch_data['instances'])
                    # loss += F.binary_cross_entropy_with_logits(torch.special.logit(output['instances'], eps=1e-9), batch_data['instances'])
                    # loss += self.focal_loss(F.log_softmax(output['binary'], dim=1).double(), batch_data['instances'])
                    # loss += mse_loss(output['instances'], batch_data['instances'])

                    #new order
                    if len(prev_output_list) >= 1:
                        prev_mask_output = prev_output_list[-1].clone()
                        curr_mask_output = torch.sigmoid(output['instances'])# output['instances'] # #  #new
                        curr_order_output = output['order']#  torch.tanh(output['order']) #
                        prev_order_gt = prev_order_gt_list[-1].clone()
                        points_, points_order_ = torch.split(points.clone().view(-1, points.size(2)), [2, 1], dim=1)

                        # now_order = self.decode_order_similarity_batch(encoded_order=curr_order_output, max_order=20, embed_dim=1)
                        # curr_mask_output = self.order_to_prediction(prev_output=prev_mask_output.detach(), now_order=now_order.clone().detach(), max_order=torch.max(points_order_).clone().detach())
                        # output['instances'] = curr_mask_output

                        order_gt = self.order_of_pixel(prev_mask_output, curr_mask_output, prev_order_gt, now_order=torch.max(points_order_))
                        order_embed_gt = order_gt #self.order_encoding_of_pixel(order_gt.clone(), embed_dim=1) # for input use
                        prev_order = order_embed_gt
                        prev_order_gt_list.append(order_gt)
                        
                        error_order_gt = self.mark_error_and_modify_order(batch_data['instances'], prev_mask_output, curr_mask_output, prev_order_gt, \
                                                                              order_gt.clone().detach(), torch.max(points_order_))
                        
                        batch_data['order_GT'] = error_order_gt # order_gt
                        now_order_embed_gt =  error_order_gt.clone() #self.order_encoding_of_pixel(error_order_gt.clone(), embed_dim=1) # for output use
                        # loss += self.net.calculate_cross_entropy(output['instances'], now_order_embed_gt)
                        # positive_weight = self.compute_weights_3d(now_order_embed_gt).squeeze(1)
                        now_order_embed_gt =  self.net.get_order_embedding(now_order_embed_gt)

                        loss_order = torch.exp(1 - cosine_similarity(curr_order_output, now_order_embed_gt)) - 1

                        n = torch.randint(1, 13, error_order_gt.shape, device=error_order_gt.device)
                        if random.random() > 0.5:
                            error_order_gt_negative = error_order_gt + n
                            error_order_gt_negative = torch.clamp(error_order_gt_negative, min=0, max=20)
                        else:
                            tmp = (error_order_gt == 0)
                            error_order_gt_negative = error_order_gt - n
                            error_order_gt_negative = torch.clamp(error_order_gt_negative, min=0, max=20)
                            error_order_gt_negative[tmp] =  random.randint(1, 13)

                            
                        now_order_embed_gt_negative =  error_order_gt_negative #self.order_encoding_of_pixel(error_order_gt_negative, embed_dim=1) # for output use
                        # negative_weight = self.compute_weights_3d(now_order_embed_gt_negative).squeeze(1)
                        now_order_embed_gt_negative =  self.net.get_order_embedding(now_order_embed_gt_negative)
                        loss_order_negative = torch.clamp(torch.exp(cosine_similarity(curr_order_output, now_order_embed_gt_negative))-1, min=0)
                        
                        # loss_order = torch.sqrt(mse_loss(curr_order_output ,order_embed_gt)) #regression
                        # loss_order = 0.01 * self.focal_loss(curr_order_output, error_order_gt.squeeze(1).long()) #classification
                        # print('order', loss_order, loss_order.dtype)
                        # print('loss', loss, loss.dtype)
                        # loss += 0.1 * (((loss_order*positive_weight).sum((1,2))).mean() + ((loss_order_negative*negative_weight).sum((1,2))).mean())
                        loss += 0.5 * ((loss_order).mean() + loss_order_negative.mean())

                    #new order
                    prev_output = torch.sigmoid(output['instances'])
                    prev_output_list.append(torch.sigmoid(output['instances']))
                    # prev_output = output['instances']
                    # prev_output_list.append(prev_output)

                    if click_indx < num_iters - 1:
                        points = get_next_points(prev_output,
                                                gt_mask,
                                                points)

                    if self.net.with_prev_mask and self.prev_mask_drop_prob > 0:
                        zero_mask = np.random.random(size=prev_output.size(0)) < self.prev_mask_drop_prob
                        prev_output[zero_mask] = torch.zeros_like(prev_output[zero_mask])
                    # Compute the triplet loss between the feature vectors of the previous output list
                    # if len(prev_output_list) >= 3:
                    #     features_a = prev_output_list[-3]
                    #     features_p = prev_output_list[-2]
                    #     features_n = prev_output_list[-1]
                    #     loss_triplet = triplet_loss(features_a, features_p, features_n)
                    #     print(loss.dtype)
                    #     # print(loss_triplet)
                    #     loss +=  10 * loss_triplet
            else:
                # iter mask (RITM)
                points, prev_output = self.find_next_n_points(
                    image,
                    gt_mask,
                    points,
                    prev_output,
                    num_iters,
                    not validation
                )

                net_input = torch.cat((image, prev_output), dim=1) \
                    if self.net.with_prev_mask else image
                output = self._forward(self.net, net_input, points)

                loss = self.add_loss(
                    'instance_loss',
                    loss,
                    losses_logging,
                    validation,
                    lambda: (output['instances'], batch_data['instances']))
                loss = self.add_loss(
                    'instance_aux_loss',
                    loss,
                    losses_logging,
                    validation,
                    lambda: (output['instances'], batch_data['instances']))

            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        m.update(*(output.get(x) for x in m.pred_outputs),
                                 *(batch_data[x] for x in m.gt_outputs))

        batch_data['points'] = points
        return loss, losses_logging, batch_data, output



    def order_to_prediction(self, prev_output, now_order, max_order):
        prev_output = prev_output.bool()  # Convert prev_output to bool
        indx = (now_order == max_order.to(now_order.device))  # Create a boolean mask
        curr_output = prev_output.clone()  # Clone prev_output to create curr_output
        curr_output[indx] = torch.logical_not(prev_output[indx])  # Apply logical_not only on specific index
        return curr_output.float()  # Convert curr_output to float

    def compute_weights_3d(self, now_order_embed_gt):
        batch_size, _, width, height = now_order_embed_gt.shape
        weights = torch.zeros_like(now_order_embed_gt)

        for b in range(batch_size):
            # Compute the histogram
            histogram = torch.histc(now_order_embed_gt[b, 0].float(), 
                                    bins=int(now_order_embed_gt[b, 0].max().item())+1, 
                                    min=0, 
                                    max=int(now_order_embed_gt[b, 0].max().item()))

            # Avoid division by zero
            histogram += 1e-10

            # Compute the weights
            weights_per_integer = 1.0 / histogram

            # Create a weight map that assigns to each position the weight of the integer at that position
            weights[b, 0] = weights_per_integer[now_order_embed_gt[b, 0].long()]

        return weights



    def mark_error_and_modify_order(self, groundtruth, prev_output, curr_output, prev_order_of_pixel_, order_of_pixel_, max_order):
        # Convert groundtruth and output to binary tensors
        groundtruth_binary = groundtruth.float()
        prev_output_binary = (prev_output >= 0.5).float()
        curr_output_binary = (curr_output >= 0.5).float()

        # Get the device of the tensors
        device = groundtruth.device

        # Distinguish false positives and false negatives
        prev_false_positives = (groundtruth_binary < prev_output_binary)
        prev_false_negatives = (groundtruth_binary > prev_output_binary)
        prev_true_positives_curr_false_negative = (groundtruth_binary == prev_output_binary) & (groundtruth_binary > curr_output_binary)

        # Clone the current pixel order to create a new tensor for the new order
        new_order_of_pixel_ = order_of_pixel_.clone()

        # Modify the new order tensor
        new_order_of_pixel_[groundtruth_binary < 1] = 0
        new_order_of_pixel_[prev_false_negatives] = max_order

        # Here is the modification: assign the value from the previous order of pixel 
        # to the locations where the previous output was true positive and the current output is false negative
        new_order_of_pixel_[prev_true_positives_curr_false_negative] = prev_order_of_pixel_[prev_true_positives_curr_false_negative]

        return new_order_of_pixel_.to(device)


    # def order_of_pixel(self, prev_output, curr_output, prev_order_gt, now_order, threshold=0.49):
    #     # Use threshold to generate the order of click of pixel ground truth.
    #     # The pixel order should be in [0, 1, 2, 3, ...].
    #     # There is a change in the pixel value crossing the threshold.
    #     # If the curr_output < 0.49 but prev_output >= 0.49, we define the pixel as belonging to the now_order click.
    #     # If prev_output < 0.49 and curr_output also < 0.49, we don't modify the value.
    #     # If prev_output > 0.49 and curr_output > 0.49, we don't modify the value.
    #     # If prev_output < 0.49 and curr_output > 0.49, we modify the value.
    #     # only compare to the threshold
    #     # Get the device of the outputs
    #     device = prev_output.device

    #     # Create the masks for the conditions
    #     prev_mask = (prev_output >= threshold)
    #     curr_mask = (curr_output < threshold)
    
    #     # Initialize the ground truth for pixel order
    #     order_gt_ = prev_order_gt.clone()
    
    #     # Update the order ground truth map using vectorized operations
    #     updated_pixels = (curr_mask & prev_mask) | (~curr_mask & ~prev_mask)
    #     order_gt_[updated_pixels] = now_order
    
    #     return order_gt_.to(device)
    def order_of_pixel(self, prev_output, curr_output, prev_order_gt, now_order, threshold=0.5):
        # Get the device of the outputs
        device = prev_output.device

        # Create binary masks for the conditions.
        # These masks are True for each pixel where the condition is met and False where it is not.

        # Mask for pixels where previous output is greater or equal to the threshold
        prev_above_threshold = (prev_output >= threshold)
        
        # Mask for pixels where current output is less than the threshold
        curr_below_threshold = (curr_output < threshold)
        
        # Mask for pixels where previous output is less than the threshold
        prev_below_threshold = (prev_output < threshold)
        
        # Mask for pixels where current output is greater or equal to the threshold
        curr_above_threshold = (curr_output >= threshold)

         # Initialize the ground truth for pixel order with a clone of previous order ground truth
        order_gt_ = prev_order_gt.clone()

        # # Condition 1: 
        # # If the current output is less than the threshold but previous output is greater or equal to the threshold, 
        # # we define the pixel as belonging to the 0
        # decrease_across_threshold = prev_above_threshold & curr_below_threshold
        # order_gt_[decrease_across_threshold] = 0
        # Condition 1: 
        # If the current output is less than the threshold but previous output is greater or equal to the threshold, 
        # we define the pixel as belonging to the 0
        decrease_across_threshold = prev_above_threshold & curr_below_threshold
        order_gt_[decrease_across_threshold] = now_order


        # Condition 4:
        # If previous output is less than the threshold and current output is greater or equal to the threshold, 
        # we modify the value to now_order.
        increase_across_threshold = prev_below_threshold & curr_above_threshold
        order_gt_[increase_across_threshold] = now_order

        # Note: For condition 2 and 3 where either both previous and current outputs are less than the threshold
        # or both are greater or equal to the threshold, we do not make any modifications.

        return order_gt_.to(device)


    def order_encoding_of_pixel(self, order_gt, max_order=49, embed_dim=1):
        # Compute the position encoding
        # Get the device of the outputs
        device = order_gt.device
        position = torch.arange(1, max_order+1, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros((int(max_order), embed_dim), device=device)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Check if pe is empty
        if pe.numel() == 0:
            pe = torch.zeros((1, embed_dim), device=device)
         #  Replace order_gt with the position encoding values
        order_gt_flattened = order_gt.view(-1)  # Flatten order_gt
        
        pos_embedding = pe[order_gt_flattened.long()]  # Index into the position encoding array
        pos_embedding = pos_embedding.view(*order_gt.shape, -1).squeeze(-1)  # Reshape pos_embedding to match order_gt with an extra dimension

        return pos_embedding
        
    def decode_order(self, encoded_order, max_order=49, embed_dim=1):
        encoded_order_size = encoded_order.size()
        # Compute the position encoding
        position = torch.arange(1, max_order+1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros((int(max_order), embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Compute the position encoding for the decoded order
        decoded_pe = torch.zeros((max_order, embed_dim))
        decoded_pe[:, 0::2] = torch.sin(position * div_term)
        decoded_pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape the tensors to have the same number of columns
        encoded_order = encoded_order.view(-1, embed_dim).float()
        decoded_pe = decoded_pe.view(-1, embed_dim).float()

        # Compute the distances between the encoded position embedding and the decoded position encoding
        distances = torch.cdist(encoded_order, decoded_pe, p=2)

        # Find the index of the minimum distance for each encoded pixel
        _, min_indices = torch.min(distances, dim=1)

        # Reshape the indices to match the shape of the input tensor
        decoded_order = min_indices.view(encoded_order_size)

        return decoded_order

    def decode_order_similarity_batch(self, encoded_order, max_order=20, embed_dim=1):
        encoded_order_size = encoded_order.size()

        # Compute the position encoding
        position = torch.arange(0, max_order, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros((max_order, embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape the tensors to have the same number of columns
        pe = position.view(max_order,1,1,1).float().cpu() #pe.view(max_order, 1, 1, 1).float()

        # Apply convolution to position encoding
        decoded_pe_embedding = self.net.get_order_embedding(pe.to(self.device))

        decode_order_list = []
        
        for i in range(encoded_order.size(0)):
            # Compute the dot product between the tensors x1 and x2
            dot_product = torch.einsum('nchw,mchw->nmhw', decoded_pe_embedding, encoded_order[i:i+1])

            # Compute the magnitudes of the tensors x1 and x2
            x1_magnitude = torch.sqrt(torch.sum(decoded_pe_embedding ** 2, dim=1, keepdim=True))
            x2_magnitude = torch.sqrt(torch.sum(encoded_order[i:i+1] ** 2, dim=1, keepdim=True))

            # Compute the cosine similarity
            similarities = dot_product / torch.clamp(x1_magnitude * x2_magnitude, min=1e-8)
            
            # Find the index of the maximum similarity for each encoded pixel
            _, max_indices = torch.max(similarities, dim=0)

            # Reshape the indices to match the shape of the input tensor
            decoded_order = max_indices.view(encoded_order_size[2:]).unsqueeze(0).unsqueeze(1)

            decode_order_list.append(decoded_order)
        decoded_order = torch.cat(decode_order_list, dim=0)

        return decoded_order

    def decode_order_similarity(self, encoded_order, max_order=49, embed_dim=1):
        encoded_order_size = encoded_order.size()

        # Compute the position encoding
        position = torch.arange(0, max_order, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros((max_order, embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape the tensors to have the same number of columns
        encoded_order = encoded_order.view(1, 12, 448, 448).float()
        pe = position.view(max_order,1,1,1).float() #pe.view(max_order, 1, 1, 1).float()

        # Apply convolution to position encoding
        decoded_pe_embedding = self.net.get_order_embedding(pe.to(self.device)).cpu()

        # Compute the dot product between the tensors x1 and x2
        dot_product = torch.einsum('nchw,mchw->nmhw', decoded_pe_embedding, encoded_order)

         # Compute the magnitudes of the tensors x1 and x2
        x1_magnitude = torch.sqrt(torch.sum(decoded_pe_embedding ** 2, dim=1, keepdim=True))
        x2_magnitude = torch.sqrt(torch.sum(encoded_order ** 2, dim=1, keepdim=True))

        # Compute the cosine similarity
        similarities = dot_product / torch.clamp(x1_magnitude * x2_magnitude, min=1e-8)

        # Normalize the tensors along the embed_dim dimension for cosine similarity calculation
        # encoded_order_norm = torch.nn.functional.normalize(encoded_order, p=2, dim=1)
        # decoded_pe_embedding_norm = torch.nn.functional.normalize(decoded_pe_embedding, p=2, dim=1)

        # Compute the similarity between the encoded order and the decoded order embeddings
        # similarities = torch.einsum('nchw,mchw->nmhw', decoded_pe_embedding_norm, encoded_order_norm)
        
        # Find the index of the maximum similarity for each encoded pixel
        _, max_indices = torch.max(similarities, dim=0)

        # Reshape the indices to match the shape of the input tensor
        decoded_order = max_indices.view(encoded_order_size[2:])

        return decoded_order.unsqueeze(0)


    def find_next_n_points(self, image, gt_mask, points, prev_output,
                           num_points, eval_mode=False, grad=False):
        with torch.set_grad_enabled(grad):
            for _ in range(num_points):

                if eval_mode:
                    self.net.eval()

                net_input = torch.cat((image, prev_output), dim=1) \
                    if self.net.with_prev_mask else image
                prev_output = torch.sigmoid(
                    self._forward(
                        self.net,
                        net_input,
                        points
                    )['instances']
                )

                points = get_next_points(prev_output, gt_mask, points)

                if eval_mode:
                    self.net.train()

            if self.net.with_prev_mask and self.prev_mask_drop_prob > 0 and num_points > 0:
                zero_mask = np.random.random(
                    size=prev_output.size(0)) < self.prev_mask_drop_prob
                prev_output[zero_mask] = \
                    torch.zeros_like(prev_output[zero_mask])
        return points, prev_output

    def _forward(self, model, net_input, points, *args, **kwargs):
        # handle autocast for automatic mixed precision
        if self.cfg.amp:
            with torch.cuda.amp.autocast():
                output = model(net_input, points, *args, **kwargs)
        else:
            output = model(net_input, points, *args, **kwargs)
        return output

    def add_loss(self, loss_name, total_loss, losses_logging, validation,
                 lambda_loss_inputs, iterloss_step=None, iterloss_weight=1):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)

            if iterloss_step is not None:
                losses_logging[
                    loss_name + f'_{iterloss_step}_{iterloss_weight}'
                ] = loss 
                loss = loss_weight * loss * iterloss_weight
            else:
                # iter mask (RITM)
                losses_logging[loss_name] = loss
                loss = loss_weight * loss

            total_loss = total_loss + loss

        return total_loss

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix, from_logist=False):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = splitted_batch_data['images']
        points = splitted_batch_data['points']
        instance_masks = splitted_batch_data['instances']
        orders_GT = splitted_batch_data['order_GT'].detach().cpu()
        orders = outputs['order'].detach().cpu()


        gt_instance_masks = instance_masks.cpu().numpy()
        if from_logist:
            predicted_instance_masks = torch.sigmoid(outputs['instances']).detach().cpu().numpy()
        else:
            predicted_instance_masks = outputs['instances'].detach().cpu().numpy()
        points = points.detach().cpu().numpy()


        image_blob, points, orders, orders_GT = images[0], points[0], orders[0:1], orders_GT[0:1]
        gt_mask = np.squeeze(gt_instance_masks[0], axis=0)
        predicted_mask = np.squeeze(predicted_instance_masks[0], axis=0)

        image = image_blob.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))

        image_with_points = draw_points(image, points[:self.max_interactive_points], (0, 255, 0))
        image_with_points = draw_points(image_with_points, points[self.max_interactive_points:], (255, 0, 0))

        gt_mask[gt_mask < 0] = 0.25
        gt_mask = draw_probmap(gt_mask)
        predicted_mask = draw_probmap(predicted_mask)
        viz_image = np.hstack((image_with_points, gt_mask, predicted_mask)).astype(np.uint8)

        # Add the order image
        # decoded_order = self.decode_order(orders) # regression
        # decoded_order = torch.argmax(orders, dim=1) #segmentation
        decoded_order = self.decode_order_similarity(orders)
        order_colors = np.zeros((*decoded_order.shape, 3), dtype=np.uint8)
        order_colors_GT = np.zeros((*decoded_order.shape, 3), dtype=np.uint8)

        # for i in range(1, max_order + 1):
        #     mask = (orders == i)[:, None, :, :]
        #     mask_GT = (orders_GT == i)[:, None, :, :]
        #     for c in range(3):
        #         order_colors[..., c] = np.where(mask[..., 0], order_color_map[i][c], order_colors[..., c])
        #         order_colors_GT[..., c] = np.where(mask_GT[..., 0], order_color_map[i][c], order_colors[..., c])

        # print(decoded_order.shape) #1,1,448,448
        order_image = draw_ordermap(decoded_order[0].squeeze())
        order_GT_image = draw_ordermap(orders_GT[0].squeeze())
        #order_image = cv2.resize(order_image, (viz_image.shape[1], viz_image.shape[0]))
        # print(viz_image.shape, order_colors_GT[0][0].shape, order_colors[0][0].shape)
        viz_image = np.hstack((viz_image, order_image, order_GT_image))

        _save_image('instance_segmentation', viz_image[:, :, ::-1])

    def _load_weights(self, net):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                load_weights(net, self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            logger.info(f'Load checkpoint from path: {checkpoint_path}')
            load_weights(net, str(checkpoint_path))
        return net

    @property
    def is_master(self):
        return self.cfg.local_rank == 0


def get_next_points(pred, gt, points, pred_thresh=0.50):
    pred = pred.detach().cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.49

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            order = max(points[bindx, :, 2].max(), 0) + 1
            if is_positive:
                loc = torch.argwhere(points[bindx, :num_points, 2] < 0)
                loc = loc[0, 0] if len(loc) > 0 else num_points - 1
                points[bindx, loc, 0] = float(coords[0])
                points[bindx, loc, 1] = float(coords[1])
                points[bindx, loc, 2] = float(order)
            else:
                loc = torch.argwhere(points[bindx, num_points:, 2] < 0)
                loc = loc[0, 0] + num_points if len(loc) > 0 else 2 * num_points - 1
                points[bindx, loc, 0] = float(coords[0])
                points[bindx, loc, 1] = float(coords[1])
                points[bindx, loc, 2] = float(order)

    return points


def get_iou(pred, gt, pred_thresh=0.50):
    pred_mask = pred > pred_thresh
    gt_mask = gt > 0.49

    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return intersection / union


def load_weights(model, path_to_weights):
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict)
