import torch
import torch.nn as nn
import numpy as np

from isegm.model.ops import DistMaps, BatchImageNormalize, ScaleLayer
import torch.nn.functional as F
import math

class ISModel(nn.Module):
    def __init__(self, with_aux_output=False, norm_radius=5, use_disks=False, cpu_dist_maps=False,
                 use_rgb_conv=False, use_leaky_relu=False, # the two arguments only used for RITM
                 with_prev_mask=False,
                 norm_mean_std=([.485, .456, .406], [.229, .224, .225])):
        super().__init__()

        self.with_aux_output = with_aux_output
        self.with_prev_mask = with_prev_mask
        self.with_points = False
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])
        self.nll = nn.NLLLoss()
        self.coord_feature_ch = 2
        if self.with_prev_mask:
            self.coord_feature_ch += 1

        if use_rgb_conv:
            # Only RITM models need to transform the coordinate features, though they don't use 
            # exact 'rgb_conv'. We keep 'use_rgb_conv' only for compatible issues.
            # The simpleclick models use a patch embedding layer instead 
            mt_layers = [
                nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
            self.maps_transform = nn.Sequential(*mt_layers)
        else:
            self.maps_transform=nn.Identity()

        self.order_embedding = nn.Embedding(50, 12)  # Embedding layer with 49 indices and 12-dimensional embedding
        self.dist_maps = DistMaps(order_embedding=self.order_embedding, norm_radius=norm_radius, spatial_scale=1.0,
                                  cpu_mode=cpu_dist_maps, use_disks=use_disks)
        # self.scale = nn.Parameter(torch.tensor(4.0))
        # mt_layers = [
        #         nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1),
        #         nn.LeakyReLU(negative_slope=0.2),
        #         nn.Conv2d(in_channels=16, out_channels=12, kernel_size=1),
        #         ScaleLayer(init_value=0.05, lr_mult=1)
        #     ]
        # self.order_embedding = nn.Sequential(*mt_layers)
        

    def forward(self, image, points):
        image, prev_mask, order = self.prepare_input(image)
        order = self.get_order_embedding(order)
        
        coord_features = self.get_coord_features(image, prev_mask, points)
        coord_features = self.maps_transform(coord_features)
        if self.with_points:
            outputs = self.backbone_forward(image, coord_features, points)
        else:
            outputs = self.backbone_forward(image, coord_features, order)

        # outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
        #                                                  mode='bilinear', align_corners=True)
        # outputs['order'] = nn.functional.interpolate(outputs['order'], size=image.size()[2:],
        #                                                  mode='bilinear', align_corners=True)
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)
        outputs['instances'] = self.generate_binary_probabilities(outputs['order'], 1)
        outputs['instances'] = torch.special.logit(outputs['instances'], eps=1e-6)
        return outputs

    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:4, :, :]
            order = image[:, 4:5, :, :]
            image = image[:, :3, :, :]
        image = self.normalization(image)
        return image, prev_mask, order

    def backbone_forward(self, image, coord_features=None, points=None):
        raise NotImplementedError

    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features
    
    def get_order_embedding(self, order):
        order = order.squeeze(1).long()  # Convert the tensor type to Long
        return self.order_embedding(order).permute(0, 3, 1, 2)

    def generate_probabilities(self, output, max_order=20):
        # Get the device of the tensors
        device = output.device

        # Store original output shape
        original_shape = output.shape

        # Get position encoding
        position = torch.arange(0, max_order, dtype=torch.float32).unsqueeze(1)  # size: [max_order, 1]
        pe = position.view(max_order, 1, 1, 1).float().cpu()  # size: [max_order, 1, 1, 1]
        decoded_pe_embedding = self.get_order_embedding(pe.to(device))  # size: [max_order, output_shape[1], 1, 1]

        # Reshape the tensors to match
        output = output.view(output.shape[0], output.shape[1], -1)  # size: [batch_size, output_shape[1], output_shape[2]*output_shape[3]]
        decoded_pe_embedding = decoded_pe_embedding.view(decoded_pe_embedding.shape[0], -1, 1)  # size: [max_order, output_shape[1], 1]

        # Calculate the cosine similarity [batch_size, 1, output_shape[1], output_shape[2]*output_shape[3]],  [1, max_order, output_shape[1], 1]
        # cos_sim size: [batch_size, max_order, output_shape[2]*output_shape[3]]
        cos_sim = F.cosine_similarity(output.unsqueeze(1), decoded_pe_embedding.unsqueeze(0), dim=2)

        #sol1 use softmax only
        cos_sim[:, 0] = 3 * cos_sim[:, 0] # sol1
        cos_sim_softmax = F.softmax(cos_sim, dim=1)

        #sol2 use max order and softmax
        # # Compute the maximum cosine similarity for channels 1 onwards
        # max_cos_sim = torch.max(cos_sim[:, 1:], dim=1)[0]

        # # Keep only the first and maximum cosine similarity channels
        # cos_sim_concat = torch.cat((cos_sim[:, :1], max_cos_sim.unsqueeze(1)), dim=1)
        # # Apply softmax to the cosine similarity
        # # cos_sim_softmax size: [batch_size, max_order, output_shape[2]*output_shape[3]]
        # # cos_sim_softmax = F.softmax(cos_sim, dim=1)
        # cos_sim_softmax = F.softmax(cos_sim_concat, dim=1)
        # Reshape back to the original output shape
        # final size: [batch_size, max_order, original_shape[2], original_shape[3]]

        return cos_sim.view(output.shape[0], max_order, original_shape[2], original_shape[3]), cos_sim_softmax.view(output.shape[0], -1, original_shape[2], original_shape[3])

    
    def generate_binary_probabilities(self, output, channel=2, max_order=20):
        # Generate the class probabilities
        cos_sim, probabilities = self.generate_probabilities(output, max_order)

        # Create a binary probabilities tensor
        # Class 0 (negative) stays the same, classes 1 to 20 (positive) are summed up
       
        if channel ==2 :
            binary_probabilities = torch.zeros(probabilities.shape[0], 2, probabilities.shape[2], probabilities.shape[3]).to(probabilities.device)
            # # Assign negative class probabilities
            binary_probabilities[:, 0, :, :] = probabilities[:, 0, :, :]
            
            # Assign positive class probabilities
            binary_probabilities[:, 1, :, :] = probabilities[:, 1:, :, :].sum(dim=1)
        else:
            binary_probabilities = torch.zeros(probabilities.shape[0], 1, probabilities.shape[2], probabilities.shape[3]).to(probabilities.device)
              # Assign positive class probabilities
            # binary_probabilities[:, 0, :, :] = 1 - probabilities[:, 0, :, :]
            binary_probabilities[:, 0, :, :] = probabilities[:, 1:, :, :].sum(dim=1)

        return binary_probabilities

    
    def calculate_cross_entropy(self, order, target):
        # Generate probabilities
        cos_sim, probabilities = self.generate_probabilities(order)

        # # Squeeze target tensor to remove potential singleton dimensions
        target = target.squeeze().long()  # ensure target is long type for nn.NLLLoss

        # # Compute the cross entropy
        # loss = self.nll(torch.log(probabilities), target)
        loss = F.cross_entropy(cos_sim, target)

        return loss





def split_points_by_order(tpoints: torch.Tensor, groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32)
                    for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (group_id == 0 and is_negative):  # disable negative first click
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [torch.tensor(x, dtype=tpoints.dtype, device=tpoints.device)
                    for x in group_points]

    return group_points

