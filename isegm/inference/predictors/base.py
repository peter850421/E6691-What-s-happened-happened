import torch
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide
import math
import numpy as np
import cv2
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

class BasePredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 cascade_step=0,
                 cascade_adaptive=False,
                 cascade_clicks=1,
                 **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None
        self.cascade_step = cascade_step
        self.cascade_adaptive = cascade_adaptive
        self.cascade_clicks = cascade_clicks

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())
       

    def set_input_image(self, image, gt_mask):
        image_nd = self.to_tensor(image)
        self.gt_mask = gt_mask
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
 
        self.prev_prediction = torch.zeros((2,1,448,448)).to(self.device)
        self.prev_order_encode = torch.zeros((2,1,448,448)).to(self.device)
        self.prev_order = torch.zeros((2,1,448,448)).to(self.device) #torch.zeros_like((self.original_image[:, :1, :, :]))
        
        # for visualize ()
        # self.points = torch.zeros(1,48,3).to(self.device)
        # self.points[:] = -1
        # self.prev_output = torch.zeros((1,1,448,448), dtype=torch.float32).to(self.device)
        # self.prev_order = torch.zeros((1,1,448,448), dtype=torch.float32).to(self.device)
        # self.prev_output_list = [self.prev_output]
        # self.prev_order_gt_list = [self.prev_order]

    def get_prediction(self, clicker, prev_mask=None, on_cascade=False):
        clicks_list = clicker.get_clicks()
        # if len(clicks_list) <= self.cascade_clicks and self.cascade_step > 0 and not on_cascade:
        #     for i in range(self.cascade_step):
        #         prediction = self.get_prediction(clicker, None, True)
        #         if self.cascade_adaptive and prev_mask is not None:
        #             diff_num = (
        #                 (prediction > 0.49) != (prev_mask > 0.49)
        #             ).sum()
        #             if diff_num <= 20:
        #                 return prediction
        #         prev_mask = prediction
        #     return prediction

        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]


        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            #input_image = torch.cat((input_image, prev_mask), dim=1)
            pass
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )
        prev_mask = self.prev_prediction = F.interpolate(self.prev_prediction, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])
        self.prev_order_encode = F.interpolate(self.prev_order_encode, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])
        self.prev_order = F.interpolate(self.prev_order, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])

        image_nd = torch.cat((image_nd, prev_mask, self.prev_order_encode), dim=1)
        points = self.get_points_nd(clicks_lists)
        pred_logits = self._get_prediction(image_nd, clicks_lists, is_image_changed)


        #order
        points = self.get_points_nd(clicks_lists)
        points_, points_order_ = torch.split(points.clone().view(-1, points.size(2)), [2, 1], dim=1)
        # Get the maximum value along the third dimension
        #max_val, _ = torch.max(points_order_[:, :, 2], dim=-1)[0].float()
        max_val = torch.max(points_order_).float()
        self.prev_order = self.order_of_pixel(self.prev_prediction, torch.sigmoid(pred_logits), self.prev_order, now_order=max_val)
        self.prev_order_encode = self.prev_order#self.order_encoding_of_pixel(self.prev_order, embed_dim=1)
        #order
        self.prev_prediction = torch.sigmoid(pred_logits)#prediction

        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker)
        

       
        return prediction.cpu().numpy()[0, 0]

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        return self.net(image_nd, points_nd)['instances']

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']

    def order_of_pixel(self, prev_output, curr_output, prev_order_gt, now_order, threshold=0.49):
        # Use threshold to generate the order of click of pixel ground truth.
        # The pixel order should be in [0, 1, 2, 3, ...].
        # There is a change in the pixel value crossing the threshold.
        # If the curr_output < 0.49 but prev_output >= 0.49, we define the pixel as belonging to the now_order click.
        # If prev_output < 0.49 and curr_output also < 0.49, we don't modify the value.
        # If prev_output > 0.49 and curr_output > 0.49, we don't modify the value.
        # If prev_output < 0.49 and curr_output > 0.49, we modify the value.
        # only compare to the threshold
        # Get the device of the outputs
        device = prev_output.device

        # Create the masks for the conditions
        prev_mask = (prev_output >= threshold)
        curr_mask = (curr_output < threshold)
    
        # Initialize the ground truth for pixel order
        order_gt = prev_order_gt.clone()
    
        # Update the order ground truth map using vectorized operations
        updated_pixels = (curr_mask & prev_mask) | (~curr_mask & ~prev_mask)
        order_gt[updated_pixels] = now_order
    
        return order_gt.to(device)

    def order_encoding_of_pixel(self, order_gt, max_order=49, embed_dim=1):
        # Compute the position encoding
        # Get the device of the outputs
        device = order_gt.device
        position = torch.arange(0, max_order, dtype=torch.float32, device=device).unsqueeze(1)
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
    
    def new_get_prediction(self, sample_id):

        image, gt_mask = self.original_image , torch.from_numpy(self.gt_mask).unsqueeze(0).unsqueeze(0).float()

        image = F.interpolate(image, mode='bilinear', align_corners=True,
                                   size=(448,448))
        gt_mask = F.interpolate(gt_mask, mode='nearest',# align_corners=True,
                                   size=(448,448))
        
        

        self.points = get_next_points(self.prev_output, gt_mask, self.points)
        # torch.save(image, 'image.pt')
        # torch.save(gt_mask, 'gt_mask.pt')
        # torch.save(self.points, 'points.pt')
        # torch.save(self.prev_output, 'prev_output.pt')
        # # torch.save(self.prev_order, 'prev_order.pt')
        # image = torch.load('/home/guavamin/CFR-ICL-Interactive-Segmentation/image.pt')
        # gt_mask = torch.load('/home/guavamin/CFR-ICL-Interactive-Segmentation/gt_mask.pt')
        # self.points = torch.load('/home/guavamin/CFR-ICL-Interactive-Segmentation/points.pt')
        # self.prev_output = torch.load('/home/guavamin/CFR-ICL-Interactive-Segmentation/prev_output.pt')
        # self.prev_order = torch.load('/home/guavamin/CFR-ICL-Interactive-Segmentation/prev_order.pt')
        
        net_input =  torch.cat((image, self.prev_output, self.prev_order), dim=1)
        output = self.net(net_input, self.points)
        
        # print(self.points[0,0:3])
        self.prev_output = torch.sigmoid(output['instances'])
        # print(self.prev_output)


        # exit()
        prev_mask_output = self.prev_output_list[-1].clone()
        curr_mask_output = torch.sigmoid(output['instances'])
        prev_order_gt = self.prev_order_gt_list[-1].clone()
        # print(torch.sum(curr_mask_output >= 0.49))

        points, points_order_ = torch.split(self.points.clone().view(-1, self.points.size(2)), [2, 1], dim=1)
        order_gt = self.order_of_pixel(prev_mask_output, curr_mask_output, prev_order_gt, now_order=torch.max(points_order_))
        self.prev_order_gt_list.append(order_gt)
        self.prev_output_list.append(torch.sigmoid(output['instances']))
        self.prev_order = order_gt
        # print(torch.sum(self.prev_order))

        # Visuazlize
        orders = output['order'].detach().cpu()
        decoded_order = self.decode_order_similarity(orders)
        order_gt_error = self.mark_error_and_modify_order(gt_mask.cuda(1), prev_mask_output, order_gt.clone().detach(), torch.max(points_order_))
        order_image = draw_ordermap(decoded_order[0].squeeze())
        order_gt_image = draw_ordermap(order_gt_error.clone().cpu().detach()[0].squeeze())
        points_map, points_order_map = self.net.dist_maps.get_feature(image, self.points)
        positive_points_map, negative_points_map = points_map[:, 0:1], points_map[:, 1:2]
        positive_points_order_map, negative_points_order_map = points_order_map[:, 0:1], points_order_map[:, 1:2]
        positive_points_image = draw_probmap(np.squeeze(positive_points_map.cpu().numpy()[0], axis=0))
        negative_points_image = draw_probmap(np.squeeze(negative_points_map.cpu().numpy()[0], axis=0))
        positive_points_order_image = draw_ordermap(positive_points_order_map.clone().cpu().detach()[0].squeeze())
        negative_points_order_image = draw_ordermap(negative_points_order_map.clone().cpu().detach()[0].squeeze())


        image = image[0].cpu().numpy() * 255
        image = image.transpose((1, 2, 0))
        image_with_points = draw_points(image, points[:24], (0, 255, 0))
        image_with_points = draw_points(image_with_points, points[24:], (255, 0, 0))
        image_with_points = cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB)

        if sample_id == 8: #12
            save_image(gt_mask, './plot/mask.png')
            prob_map = np.squeeze(curr_mask_output.cpu().numpy()[0], axis=0)
            _save_image('./plot/output_test%d.png' % torch.max(points_order_), draw_probmap(prob_map))
            _save_image('./plot/order%d.png' % torch.max(points_order_), order_image)
            _save_image('./plot/order_GT%d.png' % torch.max(points_order_), order_gt_image)
            _save_image('./plot/image_point%d.png' % torch.max(points_order_), image_with_points)
            _save_image('./plot/pos_point.png', positive_points_image)
            _save_image('./plot/neg_point.png', negative_points_image)
            _save_image('./plot/pos_order_point.png', positive_points_order_image)
            _save_image('./plot/neg_order_point.png', negative_points_order_image)
            self.visualize_similarity(21)


            # save_image(curr_mask_output, './plot/output_test%d.png' % torch.max(points_order_))

        curr_mask_output = F.interpolate(output['instances'], mode='bilinear', align_corners=True,
                                   size=self.original_image.size()[2:])
        
        curr_mask_output = torch.sigmoid(curr_mask_output)
 
        
        
        return curr_mask_output.cpu().numpy()[0, 0]
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
    def mark_error_and_modify_order(self, groundtruth, output, order_of_pixel_, max_order):
        # Convert groundtruth and output to binary tensors
        groundtruth_binary = (groundtruth > 0).float()
        output_binary = (output > 0.49).float()

        # Compare groundtruth and output to find errors
        errors = (groundtruth_binary != output_binary)

        # Get the device of the tensors
        device = groundtruth.device

        # Modify the order of pixel for the pixels with errors
        order_of_pixel_[errors] = max_order

        return order_of_pixel_.to(device)

    
    def plot_order_similarity(self, max_order=49, embed_dim=1):
        # Compute the position encoding
        position = torch.arange(0, max_order, dtype=torch.float32).unsqueeze(1)
        pe = position.view(max_order, 1, 1, 1).float()  # pe.view(max_order, 1, 1, 1).float()

        # Apply convolution to position encoding
        decoded_pe_embedding = self.net.get_order_embedding(pe.to(self.device)).cpu().view(max_order, -1)

        # Compute the dot product between the tensors x1 and x2
        dot_product = torch.matmul(decoded_pe_embedding, decoded_pe_embedding.transpose(0, 1))
        # Compute the magnitudes of the tensors x1 and x2
        x1_magnitude = torch.sqrt(torch.sum(decoded_pe_embedding ** 2, dim=1, keepdim=True))
        x2_magnitude = torch.sqrt(torch.sum(decoded_pe_embedding.transpose(0, 1) ** 2, dim=0, keepdim=True))

        # Compute the cosine similarity
        similarities = dot_product / torch.clamp(x1_magnitude * x2_magnitude, min=1e-8)

        # Reshape similarities to max_order x max_order x 1
        similarities = similarities.view(max_order, max_order)

        return similarities.cpu().numpy()
    def visualize_similarity(self, max_order=48):
        # Set global font size
        plt.rcParams.update({'font.size': 12})
        similarities = self.plot_order_similarity(max_order)

        # Normalize the values to range from -1 to 1
        min_val = -1
        max_val = 1
        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 5))
       # Use the 'viridis' colormap from Matplotlib
        cmap = plt.get_cmap("jet")

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(similarities, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.1, cbar_kws={"shrink": .5, "label": "Cosine Similarity"})

        # Set xticks and yticks
        ax.set_xticks(np.arange(0, max_order, 5) + 0.5)
        ax.set_yticks(np.arange(0, max_order, 5) + 0.5)

        # Set xticklabels and yticklabels
        ax.set_xticklabels(np.arange(0, max_order, 5), rotation=-90)
        ax.set_yticklabels(np.arange(0, max_order, 5))

        # Add colorbar and title to the figure
        # plt.colorbar(ax.collections[0], label='Cosine Similarity')
        plt.title("Cosine Similarity Matrix for Order Embedding")
        plt.tight_layout()
        # Save the figure
        plt.savefig("./plot/similarity.png", dpi=300)   


        # Clear the figure
        plt.clf()



def get_next_points(pred, gt, points, pred_thresh=0.49):
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

def draw_probmap(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_OCEAN)

def _save_image(path, image):
    cv2.imwrite(path,
                image, [cv2.IMWRITE_JPEG_QUALITY, 100])

def draw_ordermap(order, color_map=None):
    """
    Draw an RGB order map where each pixel is colored according to its order value.
    
    Args:
        order (torch.Tensor): A tensor of shape (H, W) containing the order values for each pixel.
        color_map (dict, optional): A dictionary mapping order values to unique RGB colors.
        
    Returns:
        A numpy array of shape (H, W, 3) representing the order map.
    """
    # Convert order tensor to numpy array and initialize an empty RGB order map
    order = order.numpy().astype(int)
    h, w = order.shape
    order_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Automatically generate color_map if not provided
    if color_map is None:
        max_order = int(order.max().item()) + 1
        np.random.seed(42)
        color_map = {i: tuple(np.random.randint(15, 255, size=(3,)).astype(int)) for i in range(1, max_order + 1)}
    
    # Create a color map array for vectorized assignment
    max_color_key = max(color_map.keys())
    color_map_array = np.zeros((max_color_key + 1, 3), dtype=np.uint8)
    for key, value in color_map.items():
        color_map_array[key] = value

    # Use advanced indexing to map order values to unique RGB colors
    order_map = color_map_array[order]

    return order_map

def draw_points(image, points, color, radius=10):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        if len(p) == 3:
            marker = {
                0: cv2.MARKER_CROSS,
                1: cv2.MARKER_DIAMOND,
                2: cv2.MARKER_STAR,
                3: cv2.MARKER_TRIANGLE_UP
            }[p[2]] if p[2] <= 3 else cv2.MARKER_SQUARE
            image = cv2.drawMarker(image, (int(p[1]), int(p[0])),
                                   color, marker, 4, 1)
        else:
            pradius = radius
            image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)

    return image
