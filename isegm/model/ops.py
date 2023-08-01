import torch
from torch import nn as nn
import numpy as np
import isegm.model.initializer as initializer
import math

def select_activation_function(activation):
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return nn.ReLU
        elif activation.lower() == 'softplus':
            return nn.Softplus
        else:
            raise ValueError(f"Unknown activation type {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError(f"Unknown activation type {activation}")


class BilinearConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, scale, groups=1):
        kernel_size = 2 * scale - scale % 2
        self.scale = scale

        super().__init__(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=scale,
            padding=1,
            groups=groups,
            bias=False)

        self.apply(initializer.Bilinear(scale=scale, in_channels=in_channels, groups=groups))


class DistMaps(nn.Module):
    def __init__(self, order_embedding, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks
        if self.cpu_mode:
            from isegm.utils.cython import get_dist_maps
            self._get_dist_maps = get_dist_maps
        self.order_embedding =  nn.Embedding(50, 12)  # Embedding layer with 49 indices and 12-dimensional embedding #order_embedding #
    def compute_pos_embedding(self, points_order, embed_dim=1):
        max_order = 49#torch.max(points_order) + 1
        position = torch.arange(1, max_order+1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros((int(max_order), embed_dim)).to(points_order.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if torch.all(points_order == -1):
            return torch.zeros((1, embed_dim)).to(points_order.device)
        pos_embedding = pe[points_order.long()]
        return pos_embedding
    def get_coord_features(self, points, batchsize, rows, cols):
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                coords.append(self._get_dist_maps(points[i].cpu().float().numpy(), rows, cols,
                                                  norm_delimeter))
            coords = torch.from_numpy(np.stack(coords, axis=0)).to(points.device).float()
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, points.size(2))
            points, points_order = torch.split(points, [2, 1], dim=1)
            points_order_embedding = self.compute_pos_embedding(points_order)
            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
            row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
            col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

            coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
            coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)

            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
            coords.add_(-add_xy)
            if not self.use_disks:
                coords.div_(self.norm_radius * self.spatial_scale)
            coords.mul_(coords)

            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]

            coords[invalid_points, :, :, :] = 1e6

            coords = coords.view(-1, num_points, 1, rows, cols)
            coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w
            coords = coords.view(-1, 2, rows, cols)

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_()
        return coords
    
    def get_order_features(self, points, batchsize, rows, cols):
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                coords.append(self._get_dist_maps(points[i].cpu().float().numpy(), rows, cols,
                                                  norm_delimeter))
            coords = torch.from_numpy(np.stack(coords, axis=0)).to(points.device).float()
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, points.size(2))
            points, points_order = torch.split(points, [2, 1], dim=1)
            points_order_embedding = self.compute_pos_embedding(points_order)
            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
            row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
            col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

            coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
            coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)

            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
            coords.add_(-add_xy)
            if not self.use_disks:
                coords.div_(self.norm_radius * self.spatial_scale)
            coords.mul_(coords)

            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]

            coords[invalid_points, :, :, :] = 1e6

            coords = coords.view(-1, num_points, 1, rows, cols)

            coords, indices = coords.min(dim=1)  # -> (bs * num_masks * 2) x 1 x h x w

            # Compute position embedding for each pixel in coords
            #pos_embedding = self.compute_pos_embedding(indices.view(-1), embed_dim=1)
            # Reshape pos_embedding to match the shape of coords
            pos_embedding = indices+1 #pos_embedding.view(coords.size())


            coords = coords.view(-1, 2, rows, cols)
            # Reshape pos_embedding to match the shape of coords
            pos_embedding = pos_embedding.view(-1, 2, rows, cols)
            
        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
            pos_embedding = coords * pos_embedding
            pos_embedding = self.get_order_embedding(pos_embedding.long())
        else:
            coords.sqrt_().mul_(2).tanh_()
        return coords, pos_embedding
    def get_order_embedding(self, order):
        batch_size, channels, height, width = order.size()
        # Reshape the input tensor to (batch_size, channels * height * width)
        reshaped_input = order.view(batch_size, -1)

        # Apply the embedding layer to the reshaped input
        embedded_output = self.order_embedding(reshaped_input)

        # Reshape the embedded output back to (batch_size, channels * embedding_dim, height, width)
        embedded_output = embedded_output.view(batch_size, channels * 12, height, width)

        return embedded_output

    def forward(self, x, coords):
        output_coords, pos_embedding = self.get_order_features(coords, x.shape[0], x.shape[2], x.shape[3])
        # print(output_coords.size(), pos_embedding.size())
        # print(output_coords, pos_embedding)
        
        return torch.cat((output_coords, pos_embedding), dim=1)
    def vis_get_order_features(self, points, batchsize, rows, cols):
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                coords.append(self._get_dist_maps(points[i].cpu().float().numpy(), rows, cols,
                                                  norm_delimeter))
            coords = torch.from_numpy(np.stack(coords, axis=0)).to(points.device).float()
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, points.size(2))
            points, points_order = torch.split(points, [2, 1], dim=1)
            points_order_embedding = self.compute_pos_embedding(points_order)
            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
            row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
            col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

            coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
            coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)

            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
            coords.add_(-add_xy)
            if not self.use_disks:
                coords.div_(self.norm_radius * self.spatial_scale)
            coords.mul_(coords)

            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]

            coords[invalid_points, :, :, :] = 1e8

            coords = coords.view(-1, num_points, 1, rows, cols)

            coords, indices = coords.min(dim=1)  # -> (bs * num_masks * 2) x 1 x h x w

            # Compute position embedding for each pixel in coords
            #pos_embedding = self.compute_pos_embedding(indices.view(-1), embed_dim=1)
            # Reshape pos_embedding to match the shape of coords
            pos_embedding = indices+1 #pos_embedding.view(coords.size())


            coords = coords.view(-1, 2, rows, cols)
            # Reshape pos_embedding to match the shape of coords
            pos_embedding = pos_embedding.view(-1, 2, rows, cols)

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
            pos_embedding = coords*pos_embedding
        else:
            coords.sqrt_().mul_(2).tanh_()

        return coords, pos_embedding

    def get_feature(self, x, coords):
        output_coords, pos_embedding = self.vis_get_order_features(coords, x.shape[0], x.shape[2], x.shape[3])
        return output_coords, pos_embedding


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
        )

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale


class BatchImageNormalize:
    def __init__(self, mean, std, dtype=torch.float):
        self.mean = torch.as_tensor(mean, dtype=dtype)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype)[None, :, None, None]

    def __call__(self, tensor):
        tensor = tensor.clone()

        tensor.sub_(self.mean.to(tensor.device)).div_(self.std.to(tensor.device))
        return tensor
