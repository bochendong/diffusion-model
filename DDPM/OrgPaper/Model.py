import math
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

# ===== Neural network building defaults =====
DEFAULT_DTYPE = torch.float32


def default_init(scale):
    return nn.init.kaiming_uniform_ if scale == 0 else nn.init.kaiming_uniform_


# ===== Utilities =====

def debug_print(x, name):
    print(name, x.mean().item(), x.std().item(), x.min().item(), x.max().item())
    return x


def flatten(x):
    return x.view(x.size(0), -1)


def sumflat(x):
    return x.sum(dim=list(range(1, len(x.size()))))


def meanflat(x):
    return x.mean(dim=list(range(1, len(x.size()))))


# ===== Neural network layers =====

def _einsum(a, b, c, x, y):
    einsum_str = f"{''.join(a)}, {''.join(b)} -> {''.join(c)}"
    return einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[:len(x.size())])
    y_chars = list(string.ascii_uppercase[:len(y.size())])
    assert len(x_chars) == len(x.size()) and len(y_chars) == len(y.size())
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class Nin(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=1.):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_dim, num_units, dtype=DEFAULT_DTYPE))
        self.b = nn.Parameter(torch.zeros(num_units, dtype=DEFAULT_DTYPE))
        default_init(init_scale)(self.W)

    def forward(self, x):
        y = contract_inner(x, self.W) + self.b
        assert y.size() == x.size()[:-1] + (self.W.size(-1),)
        return y


class Dense(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=1., bias=True):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_dim, num_units, dtype=DEFAULT_DTYPE))
        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.zeros(num_units, dtype=DEFAULT_DTYPE))
        else:
            self.b = None
        default_init(init_scale)(self.W)

    def forward(self, x):
        z = torch.matmul(x, self.W)
        if self.bias:
            return z + self.b
        return z


class Conv2d(nn.Module):
    def __init__(self, in_dim, num_units, filter_size=(3, 3), stride=1, dilation=None, pad='SAME', init_scale=1., bias=True):
        super().__init__()
        if isinstance(filter_size, int):
            filter_size = (filter_size, filter_size)

        self.padding = 'same' if pad == 'SAME' else pad
        self.stride = stride
        self.dilation = dilation if dilation is not None else 1
        self.W = nn.Parameter(torch.empty(*filter_size, in_dim, num_units, dtype=DEFAULT_DTYPE))
        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.zeros(num_units, dtype=DEFAULT_DTYPE))
        else:
            self.b = None
        default_init(init_scale)(self.W)

    def forward(self, x):
        z = F.conv2d(x, self.W, stride=self.stride, padding=self.padding, dilation=self.dilation)
        if self.bias:
            return z + self.b
        return z


def get_timestep_embedding(timesteps, embedding_dim: int):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=DEFAULT_DTYPE) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def nonlinearity(x):
    return F.silu(x)


class Normalize(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, num_channels)

    def forward(self, x):
        return self.norm(x)


class Upsample(nn.Module):
    def __init__(self, num_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = Conv2d(num_channels, num_channels, filter_size=3, stride=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, num_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = Conv2d(num_channels, num_channels, filter_size=3, stride=2)
        else:
            self.avg_pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        if self.with_conv:
            return self.conv(x)
        else:
            return self.avg_pool(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, temb_channels, out_channels=None, conv_shortcut=False, dropout=0.):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, filter_size=3, stride=1)
        self.temb_proj = Dense(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = Conv2d(out_channels, out_channels, filter_size=3, stride=1)

        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, filter_size=1, stride=1) if conv_shortcut else Nin(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb):
        h = nonlinearity(self.norm1(x))
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, None, None, :]
        h = nonlinearity(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        x = self.shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = Normalize(num_channels)
        self.q = Nin(num_channels, num_channels)
        self.k = Nin(num_channels, num_channels)
        self.v = Nin(num_channels, num_channels)
        self.proj_out = Nin(num_channels, num_channels)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        w = einsum('bhwc,bHWc->bhwHW', q, k) * (C ** -0.5)
        w = w.view(B, H, W, H * W)
        w = F.softmax(w, dim=-1)
        w = w.view(B, H, W, H, W)

        h = einsum('bhwHW,bHWc->bhwc', w, v)
        h = self.proj_out(h)

        return x + h


class Model(nn.Module):
    def __init__(self, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks=2, attn_resolutions=(16,), dropout=0., resamp_with_conv=True):
        super().__init__()
        num_resolutions = len(ch_mult)
        self.ch = ch

        # Timestep embedding layers
        self.temb_dense0 = Dense(ch, ch * 4)
        self.temb_dense1 = Dense(ch * 4, ch * 4)

        # Downsampling layers
        self.conv_in = Conv2d(1, ch, filter_size=3, stride=1)
        self.down_layers = nn.ModuleList()
        in_ch = ch
        for i_level, mult in enumerate(ch_mult):
            out_ch = ch * mult
            block = nn.ModuleList([ResnetBlock(in_ch, ch * 4, out_ch, dropout=dropout) for _ in range(num_res_blocks)])
            attn_block = nn.ModuleList([AttnBlock(out_ch) for _ in range(num_res_blocks) if mult in attn_resolutions])
            downsample = Downsample(out_ch, resamp_with_conv) if i_level != num_resolutions - 1 else nn.Identity()
            self.down_layers.append((block, attn_block, downsample))
            in_ch = out_ch

        # Middle layers
        self.mid_block1 = ResnetBlock(in_ch, ch * 4, dropout=dropout)
        self.mid_attn = AttnBlock(in_ch)
        self.mid_block2 = ResnetBlock(in_ch, ch * 4, dropout=dropout)

        # Upsampling layers
        self.up_layers = nn.ModuleList()
        for i_level, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            block = nn.ModuleList([ResnetBlock(in_ch + out_ch, ch * 4, out_ch, dropout=dropout) for _ in range(num_res_blocks + 1)])
            attn_block = nn.ModuleList([AttnBlock(out_ch) for _ in range(num_res_blocks + 1) if mult in attn_resolutions])
            upsample = Upsample(out_ch, resamp_with_conv) if i_level != 0 else nn.Identity()
            self.up_layers.append((block, attn_block, upsample))
            in_ch = out_ch

        # Final layers
        self.norm_out = Normalize(in_ch)
        self.conv_out = Conv2d(in_ch, out_ch, filter_size=3, stride=1)

    def forward(self, x, t):
        B, S, _, _ = x.shape
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb_dense0(temb)
        temb = self.temb_dense1(nonlinearity(temb))

        h = [self.conv_in(x)]
        for block, attn_block, downsample in self.down_layers:
            for b in block:
                h.append(b(h[-1], temb))
            for a in attn_block:
                h[-1] = a(h[-1])
            h.append(downsample(h[-1]))

        h[-1] = self.mid_block1(h[-1], temb)
        h[-1] = self.mid_attn(h[-1])
        h[-1] = self.mid_block2(h[-1], temb)

        for block, attn_block, upsample in self.up_layers:
            for b in block:
                h[-1] = b(torch.cat([h.pop(), h[-1]], dim=-1), temb)
            for a in attn_block:
                h[-1] = a(h[-1])
            h[-1] = upsample(h[-1])

        h[-1] = nonlinearity(self.norm_out(h[-1]))
        return self.conv_out(h[-1])
