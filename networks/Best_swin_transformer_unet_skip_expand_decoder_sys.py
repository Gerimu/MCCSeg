import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#stable
from reweighting import weight_learner
from lib.Res2Net_v1b import res2net50_v1b_26w_4s


class BasicConv2d(nn.Module):   #BGM gaos
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Scale(nn.Module):     # Scale branch output gao
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class Wide_Focus(nn.Module):
    """
    Wide-Focus module.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 H):
        super().__init__()
        self.resolution = H
        self.dim = in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=3)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")

    def forward(self, x):
        H, W = self.resolution
        dim = self.dim
        x_reshape = x.view(-1, H, W, dim).permute(0, 3, 1, 2).contiguous()
        x1 = self.conv1(x_reshape)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, 0.1)
        x2 = self.conv2(x_reshape)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, 0.1)
        x3 = self.conv3(x_reshape)
        x3 = F.gelu(x3)
        x3 = F.dropout(x3, 0.1)
        added = torch.add(x1, x2)
        added = torch.add(added, x3)
        x_out = self.conv4(added)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, 0.1)
        x_out = rearrange(x_out, 'b c h w -> b (h w) c')
        return x_out

class Multi_scale(nn.Module):       #muti scale gao

    def __init__(self, channels):
        super().__init__()
        # Multi-scale input
        self.dim = channels
        in_channels = self.dim
        out_channels = self.dim * 2
        self.scale_img = nn.AvgPool2d(2, 2)
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.scale_conv1 = nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding="same")
        self.scale_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same")
        self.scale_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same")
        self.linear = nn.Linear(out_channels, in_channels, bias=True)

    def forward(self, x, x_scale):
        dim =self.dim
        B, C, H, W = x_scale.shape
        x1 = x.view(-1, H, W, dim)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = torch.cat((F.relu(self.scale_conv1(x_scale)), x1), axis=1)
        x1 = F.relu(self.scale_conv2(x1))
        x1 = F.relu(self.scale_conv3(x1))
        x1 = F.dropout(x1, 0.3)
        # x1 = F.max_pool2d(x1, (2, 2))
        x1 = rearrange(x1, 'b c h w -> b (h w ) c')
        x1 = self.linear(x1)

        return  x1

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale_init_value=None, res_scale_init_value=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        # Scale gao
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        # Wide_Foucus
        self.wide_focus = Wide_Focus(dim, dim, self.input_resolution)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        # x = shortcut + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = self.res_scale1(shortcut) + self.layer_scale1(self.drop_path(x))
        x = self.res_scale2(x) + self.layer_scale2(self.drop_path(self.mlp(self.norm2(x))))     # Scale gao
        # x = self.res_scale2(x) + self.layer_scale2(self.drop_path(self.wide_focus(self.norm2(x))))  # wide_focus gao

        # # 可视化
        # img = x
        # img = img.view(B, H, W, C)
        # # img = img.view(B, 4 * H, 4 * W, -1)
        # # img = rearrange(img, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=4, p2=4, c=C // 16)
        # img = img.cpu()
        # img = np.squeeze(img.detach().numpy())
        # plt.figure()
        # for i in range(8):
        #     ax = plt.subplot(2, 4, i + 1)
        #     # [H, W, C]
        #     plt.imshow(img[:, :, i], cmap='gray')
        # plt.show()
        # # 可视化

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 layer_scale_init_values=[1.0, 1.0, 1.0, 1.0],
                 res_scale_init_values=[1.0, 1.0, 1.0, 1.0],
                 ):         # Scale gao

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        num_stage = 4         # Scale gao
        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 layer_scale_init_value=layer_scale_init_values[i],
                                 res_scale_init_value=res_scale_init_values[i],
                                 )
            for i in range(depth)])         # Scale gao

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # num_stage = 4         # Scale gao
        # if not isinstance(layer_scale_init_values, (list, tuple)):
        #     layer_scale_init_values = [layer_scale_init_values] * num_stage
        # if not isinstance(res_scale_init_values, (list, tuple)):
        #     res_scale_init_values = [res_scale_init_values] * num_stage

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 )
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths,
        depths_decoder,drop_path_rate,num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        
        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up= norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)

        self.apply(self._init_weights)

        #stable
        self.linear = nn.Linear(37632, 512, bias=True)  # 降维   change to 1024 or 768? 37632=49*768
        #stable

        ##BGM conv
        self.nf = 32 #channal
        self.nc = 9
        act_fn = nn.ReLU(inplace=True)
        self.edge_conv0 = nn.Sequential(nn.Conv2d(64, self.nf, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.nf), act_fn)
        self.edge_conv1 = nn.Sequential(nn.Conv2d(256, self.nf, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.nf), act_fn)
        self.edge_conv2 = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.nf), act_fn)
        self.edge_conv3 = BasicConv2d(self.nf, 1, kernel_size=3, padding=1)
        self.up_4_BGM = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        #BGM guidance
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        self.fu_4 = nn.Sequential(nn.Conv2d(768 + self.nf, self.nf, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(self.nf), act_fn)
        self.fu_3 = nn.Sequential(nn.Conv2d(384 + self.nf, self.nf, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(self.nf), act_fn)
        self.fu_2 = nn.Sequential(nn.Conv2d(self.nf + 192, self.nf, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(self.nf), act_fn)
        self.fu_1 = nn.Sequential(nn.Conv2d(self.nf + 96, self.nf, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(self.nf), act_fn)
        ## layer_out
        self.layer_out4 = nn.Sequential(nn.Conv2d(768, self.nc, kernel_size=3, stride=1, padding=1))        # de_BGM gao
        self.layer_out3 = nn.Sequential(nn.Conv2d(384, self.nc, kernel_size=3, stride=1, padding=1))
        self.layer_out2 = nn.Sequential(nn.Conv2d(192, self.nc, kernel_size=3, stride=1, padding=1))
        self.layer_out1 = nn.Sequential(nn.Conv2d(96, self.nc, kernel_size=3, stride=1, padding=1))

        # self.layer_out4 = nn.Sequential(nn.Conv2d(self.nf, self.nc, kernel_size=3, stride=1, padding=1))
        # self.layer_out3 = nn.Sequential(nn.Conv2d(self.nf, self.nc, kernel_size=3, stride=1, padding=1))
        # self.layer_out2 = nn.Sequential(nn.Conv2d(self.nf, self.nc, kernel_size=3, stride=1, padding=1))
        # self.layer_out1 = nn.Sequential(nn.Conv2d(self.nf, self.nc, kernel_size=3, stride=1, padding=1))

        ## up
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up_32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        ## Multi_scale
        self.scale_img = nn.AvgPool2d(2, 2)
        self.multi_scale2 = Multi_scale(channels=96)
        self.multi_scale3 = Multi_scale(channels=192)
        self.multi_scale4 = Multi_scale(channels=384)

        ##Conv_add
        self.Conv_add2 = nn.Linear(256, 96)
        self.Conv_add3 = nn.Linear(512, 192)
        self.Conv_add4 = nn.Linear(1024, 384)
        self.norm_2 = nn.LayerNorm(96)
        self.norm_3 = nn.LayerNorm(192)
        self.norm_4 = nn.LayerNorm(384)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def stable_features(self, x, args, p_fs, p_w1, epoch, i):         # def stable
        x = torch.flatten(x, 1)  # 将Encoder部分的features[24, 49, 768]拉平成[24, 49*768]
        cfeatures = self.linear(x)
        # flatten_features = x
        # cfeatures = x

        # pre_features = p_fs.cuda()
        # pre_weight1 = p_w1.cuda()

        weight1, pre_features, pre_weight1 = weight_learner(cfeatures, p_fs, p_w1, args, epoch, i)

        return weight1, pre_features, pre_weight1

    def data_normal(self, x):       # def normal for edge
        min = x.min()
        if min < 0:
            x = x + torch.abs(min)
            min = x.min()
        max = x.max()
        dst = max - min
        normal_x = (x - min).true_divide(dst)
        return normal_x
        # return 1.0 / (1 + np.exp(-x))

    def BGM_features(self, x1, x2):             # def BGM
        # x1 = x1.view(-1, 28, 28, 192).permute(0, 3, 1, 2).contiguous()  # B C W H
        # x2 = x2.view(-1, 28, 28, 96).permute(0,3,1,2)
        # x2 = x2.view(-1, 14, 14, 384).permute(0, 3, 1, 2).contiguous()  # 1.0  interpolate
        # x2 = F.interpolate(x2, size=[28, 28], mode='bilinear')
        # x1 = self.data_normal(x1)  #normal 0-1
        # x2 = self.data_normal(x2)

        x21 = self.edge_conv1(x2)
        edge_guidance = self.edge_conv2(self.edge_conv0(x1) + x21)
        edge_out = self.up_4_BGM(self.edge_conv3(edge_guidance))

        # edge_out_show = edge_out.view(-1, 224, 224)
        # edge_guidance_show = edge_guidance.view(-1, 224, 224)
        #
        # edge_out_show = edge_out_show.cpu().detach().numpy()
        # edge_guidance_show = edge_guidance_show.cpu().detach().numpy()
        #
        # edge_guidance_show[edge_guidance_show == 1] = 255
        # edge_out_show[edge_out_show == 1] = 255
        #
        # plt.imshow(edge_out_show)
        # plt.imshow(edge_guidance_show)

        edge_out = edge_out.view(-1, 224, 224)

        edge_out = self.data_normal(edge_out)  # normal gao

        return edge_out, edge_guidance

    def guidance(self, output, edge_guidance):      #guidance gao
        output[0] = output[0].view(-1, 7, 7, 768).permute(0, 3, 1, 2).contiguous()    #B C H W
        output[1] = output[1].view(-1, 14, 14, 384).permute(0, 3, 1, 2).contiguous()
        output[2] = output[2].view(-1, 28, 28, 192).permute(0, 3, 1, 2).contiguous()
        output[3] = output[3].view(-1, 56, 56, 96).permute(0, 3, 1, 2).contiguous()

        # out4 = self.fu_4(torch.cat((output[0], F.interpolate(edge_guidance, scale_factor=1/8, mode='bilinear')), dim=1))
        out4 = output[0]
        o4 = self.up_32(self.layer_out4(out4))

        # out3 = self.fu_3(torch.cat((output[1], F.interpolate(edge_guidance, scale_factor=1/4, mode='bilinear')), dim=1))
        out3 = output[1]
        o3 = self.up_16(self.layer_out3(out3))

        # out2 = self.fu_2(torch.cat((output[2], F.interpolate(edge_guidance, scale_factor=1/2, mode='bilinear')), dim=1))
        out2 = output[2]
        o2 = self.up_8(self.layer_out2(out2))

        # out1 = self.fu_1(torch.cat((output[3], edge_guidance), dim=1))
        out1 = output[3]
        o1 = self.up_4(self.layer_out1(out1))

        return o4, o3, o2, o1

    #Encoder and Bottleneck
    def forward_features(self, x):
        # add2 = x2
        # add3 = x3
        # add4 = x4

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        # multi gao
        i = 0
        # multi gao
        for layer in self.layers:
            x_downsample.append(x)

            # if i == 0:                                  #multi scale gao
            #     add2 = self.Conv_add2(add2.view(-1, 256, 3136).permute(0, 2, 1).contiguous())
            #     x = self.norm_2(x + add2)
            # if i == 1:
            #     add3 = self.Conv_add3(add3.view(-1, 512, 784).permute(0, 2, 1).contiguous())
            #     x = self.norm_3(x + add3)
            # if i == 2:
            #     add4 = self.Conv_add4(add4.view(-1, 1024, 196).permute(0, 2, 1).contiguous())
            #     x = self.norm_4(x + add4)

            x = layer(x)
            i = i + 1

        x = self.norm(x)  # B L C

        return x, x_downsample

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        x_deepsupervision = []          # 1.0 gao
        x_deepsupervision.append(x)
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
                x_deepsupervision.append(x)
            else:
                x = torch.cat([x,x_downsample[3-inx]],-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
                x_deepsupervision.append(x)

        x = self.norm_up(x)   # B L C

        return x, x_deepsupervision

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":
            x = self.up(x)
            x = x.view(B,4*H,4*W,-1)
            x = x.permute(0,3,1,2) #B,C,H,W
            x = self.output(x)
            
        return x

    # def forward(self, x, args, p_fs, p_w1, epoch_num, i_batch):
    #     # Resnet
    #     xx = x
    #     xx = self.resnet.conv1(xx)
    #     xx = self.resnet.bn1(xx)
    #     xx = self.resnet.relu(xx)
    #
    #     x1 = self.resnet.maxpool(xx)  # (BS, 64, 56, 56)
    #     x2 = self.resnet.layer1(x1)  # (BS, 256, 56, 56)
    #     x3 = self.resnet.layer2(x2)  # (BS, 512, 44, 44)        # multi scale input gao
    #     x4 = self.resnet.layer3(x3)  # (BS, 1024, 22, 22)
    #     # Resnet
    #
    #     # # Multi-scale input
    #     # scale_img_1 = self.scale_img(x)  # x shape[batch_size,channel(1),224,224]
    #     # scale_img_2 = self.scale_img(scale_img_1)  # 56 56
    #     # scale_img_3 = self.scale_img(scale_img_2)  # shape[batch,1,28,28]
    #     # scale_img_4 = self.scale_img(scale_img_3)  # shape[batch,1,14,14]
    #
    #     x, x_downsample = self.forward_features(x, x2, x3, x4)               #multi scale gao
    #
    #     #stable
    #     weight1, pre_features, pre_weight1 = self.stable_features(x, args, p_fs, p_w1, epoch_num, i_batch)
    #     #stable
    #
    #     #BGM
    #     # x1 = x1_x2[0]
    #     # x2 = x1_x2[1]
    #     edge_out, edge_guidance = self.BGM_features(x1, x2)        # edge gao
    #     #BGM
    #
    #     x, x_deeploss = self.forward_up_features(x, x_downsample)   # deeploss is deepsupervision
    #
    #     # DeepSupervision
    #     out4, out3, out2, out1 = self.guidance(x_deeploss, edge_guidance)  #edge_guidance
    #     # DeepSupervision
    #
    #     x = self.up_x4(x)
    #
    #     return x, weight1, pre_features, pre_weight1, out4, out3, out2, out1, edge_out

    def forward(self, x):
        # Multi-scale input
        # scale_img_1 = self.scale_img(x)  # x shape[batch_size,channel(1),224,224]
        # scale_img_2 = self.scale_img(scale_img_1)  # 56 56
        # scale_img_3 = self.scale_img(scale_img_2)  # shape[batch,1,28,28]
        # scale_img_4 = self.scale_img(scale_img_3)  # shape[batch,1,14,14]

        # Resnet
        # xx = x
        # xx = self.resnet.conv1(xx)
        # xx = self.resnet.bn1(xx)
        # xx = self.resnet.relu(xx)
        #
        # x1 = self.resnet.maxpool(xx)  # (BS, 64, 56, 56)
        # x2 = self.resnet.layer1(x1)  # (BS, 256, 56, 56)
        # x3 = self.resnet.layer2(x2)  # (BS, 512, 44, 44)        # multi scale input gao
        # x4 = self.resnet.layer3(x3)  # (BS, 1024, 22, 22)
        # Resnet

        # x, x_downsample = self.forward_features(x, scale_img_2, scale_img_3, scale_img_4)  # multi scale gao
        # x, x_downsample = self.forward_features(x, x2, x3, x4)               #multi scale input gao
        x, x_downsample = self.forward_features(x)

        x, x_deeploss = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
