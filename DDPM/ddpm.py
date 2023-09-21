from contextlib import contextmanager
from copy import deepcopy
import math
import os
import glob
from IPython import display
from matplotlib import pyplot as plt
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms, utils, models
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm, trange
from torchvision.utils import save_image
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True) if not is_last else nn.Identity(),
            nn.ReLU(inplace=True) if not is_last else nn.Identity(),
        ], skip)


class SelfAttention2d(nn.Module):
    def __init__(self, c_in, n_head=1, dropout_rate=0.1):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Dropout2d(dropout_rate, inplace=True)

    def forward(self, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3]**-0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.dropout(self.out_proj(y))


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        self.timestep_embed = FourierFeatures(1, 16)
        self.class_embed = nn.Embedding(11, 4)
        self.skip = nn.Identity()
        self.block_1 = nn.Sequential(                   # 32x32
            ResConvBlock(3 + 16 + 4, 64, 64),
            ResConvBlock(64, 64, 64)
        )

        self.block_2 = nn.Sequential( 
            nn.AvgPool2d(2),                            # 32x32 -> 16x16
            ResConvBlock(64, 128, 128),
            ResConvBlock(128, 128, 128)
        )


        self.block_3 = nn.Sequential( 
            nn.AvgPool2d(2),                            # 16x16 -> 8x8
            ResConvBlock(128, 256, 256),
            SelfAttention2d(256, 256 // 64),
            ResConvBlock(256, 256, 256),
            SelfAttention2d(256, 256 // 64),
        )

        self.block_4 =  nn.Sequential( 
            nn.AvgPool2d(2),                            # 8x8 -> 4x4
            ResConvBlock(256, 512, 512),
            SelfAttention2d(512, 512 // 64),
            ResConvBlock(512, 512, 512),
            SelfAttention2d(512, 512// 64),
            ResConvBlock(512, 512, 512),
            SelfAttention2d(512, 512 // 64),
            ResConvBlock(512, 512, 256),
            SelfAttention2d(256, 256// 64),
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
        )

        self.block_5 = nn.Sequential(                   # 4x4 -> 8x8
            ResConvBlock(512, 256, 256),
            SelfAttention2d(256, 256 // 64),
            ResConvBlock(256, 256, 128),
            SelfAttention2d(128, 128// 64),
        )

        self.block_6 = nn.Sequential(                   # 8x8 -> 16x16
            ResConvBlock(256, 128, 128),
            ResConvBlock(128, 128, 64),
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False),
        )

        self.block_7 = nn.Sequential(
            ResConvBlock(128, 64, 64),
            ResConvBlock(64, 64, 3, is_last=True)
        )
    def forward(self, input, t, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        class_embed = expand_to_planes(self.class_embed(cond + 1), input.shape)

        x = torch.cat([input, class_embed, timestep_embed], dim=1)

        x_1 = self.block_1(x)

        x_2 = self.block_2(x_1)

        x_3 = self.block_3(x_2)

        x_4 = self.block_4(x_4)
        x_4 =  torch.cat([x_4, self.skip(x_3)], dim=1)

        x_5 = self.block_5(x_4)
        x_5 = torch.cat([x_5, self.skip(x_2)], dim=1)

        x_6 = self.block_6(x_5)
        x_6 = torch.cat([x_6, self.skip(x_1)], dim=1)

        x_7 = self.block_7(x_6)

        return x_7

c = 64
self.net = nn.Sequential(   # 32x32
            ResConvBlock(3 + 16 + 4, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.AvgPool2d(2),  # 32x32 -> 16x16
                ResConvBlock(c, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.AvgPool2d(2),  # 16x16 -> 8x8
                    ResConvBlock(c * 2, c * 4, c * 4),
                    SelfAttention2d(c * 4, c * 4 // 64),
                    ResConvBlock(c * 4, c * 4, c * 4),
                    SelfAttention2d(c * 4, c * 4 // 64),
                    SkipBlock([
                        nn.AvgPool2d(2),  # 8x8 -> 4x4
                        ResConvBlock(c * 4, c * 8, c * 8),
                        SelfAttention2d(c * 8, c * 8 // 64),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        SelfAttention2d(c * 8, c * 8 // 64),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        SelfAttention2d(c * 8, c * 8 // 64),
                        ResConvBlock(c * 8, c * 8, c * 4),
                        SelfAttention2d(c * 4, c * 4 // 64),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ]),  # 4x4 -> 8x8
                    ResConvBlock(c * 8, c * 4, c * 4),
                    SelfAttention2d(c * 4, c * 4 // 64),
                    ResConvBlock(c * 4, c * 4, c * 2),
                    SelfAttention2d(c * 2, c * 2 // 64),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ]),  # 8x8 -> 16x16
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ]),  # 16x16 -> 32x32
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, is_last=True),
        )