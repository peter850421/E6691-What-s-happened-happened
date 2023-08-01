import math
import torch.nn as nn
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.models_vit import VisionTransformer, PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim // 2)
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x)
        x_down_16 = self.down_16(x)
        x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]


class PlainVitModel(ISModel):
    @serialize
    def __init__(
        self,
        backbone_params={},
        neck_params={}, 
        head_params={},
        order_params={},
        random_split=False,
        **kwargs
        ):

        super().__init__(**kwargs)
        self.random_split = random_split

        self.patch_embed_coords = PatchEmbed(
            img_size= backbone_params['img_size'],
            patch_size=backbone_params['patch_size'], 
            in_chans=27 if self.with_prev_mask else 2,  #3 [prev_mask(1), positive(1), ngative(1), pos/neg order(b, 2*12,448,448)]
            embed_dim=backbone_params['embed_dim'],
        )
        self.patch_embed_order = PatchEmbed(
            img_size= backbone_params['img_size'],
            patch_size=backbone_params['patch_size'], 
            in_chans=12 if self.with_prev_mask else 2,  #3 [prev_order]
            embed_dim=backbone_params['embed_dim'],
        )
        self.backbone = VisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)
        # self.order = SwinTransfomerSegHead(**order_params)
        # self.guided_map_conv1 = nn.Conv2d(3, 13, 1)
        # self.guided_map_gelu1 = nn.GELU()
        # self.guided_map_conv2 = nn.Conv2d(13, 13, 1)
        # self.guided_filter = GuidedFilter(12, 1e-1)
        # mt_layers = [
        #         nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1),
        #         nn.GroupNorm(1, 12),
        #         nn.GELU(),
        #         nn.Conv2d(12, 12, 1),
        #             ]   
        # self.instances_transform = nn.Sequential(*mt_layers)

    def backbone_forward(self, image, coord_features=None, order=None):
        coord_features = self.patch_embed_coords(coord_features)
        order_features = self.patch_embed_order(order)
        coord_features += order_features

        backbone_features = self.backbone.forward_backbone(image, coord_features, self.random_split)
        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size

        backbone_features = backbone_features.transpose(-1,-2).view(B, C, grid_size[0], grid_size[1])
        multi_scale_features = self.neck(backbone_features)
        output = self.head(multi_scale_features)
        output = nn.functional.interpolate(output, size=image.size()[2:],
                                                mode='bilinear', align_corners=True)

        
        # instances = output[:,0:1]
        # order = output[:,1:13]

        # # order = self.order(multi_scale_features)
        # instances = nn.functional.interpolate(instances, size=image.size()[2:],
        #                                         mode='bilinear', align_corners=True)
        # order = nn.functional.interpolate(order, size=image.size()[2:],
        #                                         mode='bilinear', align_corners=True)
        # guided_image = self.guided_map_gelu1(self.guided_map_conv1(image))
        # guided_image = self.guided_map_conv2(guided_image)
        # guided_image = self.instances_transform(image)

        # output = self.guided_filter(guided_image, output)
        instances = None
        order = output
        # instances = self.instances_transform(order)
        # instances = output[:,0:1]
        # order = output[:,1:13]

        
        return {'instances': instances, 'order': order, 'instances_aux': None}
    
def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)
class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b