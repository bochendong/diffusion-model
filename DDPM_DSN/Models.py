import torch
import math
from torch import nn
from torch.autograd import Function
from torchvision.models import resnet50

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

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

class FeatureEmbedding(nn.Module):
    def __init__(self):
        super(FeatureEmbedding, self).__init__()

        self.fe = nn.Sequential(
            nn.Linear(1000, 3072),
            nn.Tanh(),
        )

        self.pretrained_model = resnet50(pretrained=True)
        self.pretrained_model = self.pretrained_model

    def forward(self, x):
        with torch.no_grad():
            x = self.pretrained_model(x)
        x = self.fe(x)
        x = x.view(-1, 3, 32, 32)
        return x

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 2),
            nn.LogSoftmax(dim = 1)
        )

    def forward(self, input):
        input = input.view(-1, 256 * 8 * 8)
        return self.net(input)

def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64
        self.avg_1 = nn.AvgPool2d(2)
        self.avg_2 = nn.AvgPool2d(2)

        self.down_0 = nn.Sequential(
            ResConvBlock(3 + 16 + 4, c, c),
            ResConvBlock(c, c, c),
        )

        self.down_1 = nn.Sequential(
            ResConvBlock(c, c * 2, c * 2),
            ResConvBlock(c * 2, c * 2, c * 2),
        )

        self.down_2 = nn.Sequential(
            ResConvBlock(c * 2, c * 4, c * 4),
            SelfAttention2d(c * 4, c * 4 // 64),
            ResConvBlock(c * 4, c * 4, c * 4),
            SelfAttention2d(c * 4, c * 4 // 64),
        )

    def forward(self, input):
        down_sample = self.down_0(input)
        identity_0 = nn.Identity()(down_sample)

        down_sample = self.avg_1(self.down_1(down_sample))
        identity_1 = nn.Identity()(down_sample)

        down_sample = self.avg_2(self.down_2(down_sample))
        identity_2 = nn.Identity()(down_sample)

        return down_sample, identity_0, identity_1, identity_2

class MidSample(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64
        self.avg_3 = nn.AvgPool2d(2)
        self.conv = nn.Conv2d(512, 256, kernel_size = 3, padding = 1);

        self.mid_1 = nn.Sequential(
            ResConvBlock(c * 4, c * 8, c * 8),
            SelfAttention2d(c * 8, c * 8 // 64),
            ResConvBlock(c * 8, c * 8, c * 8),
            SelfAttention2d(c * 8, c * 8 // 64),
            ResConvBlock(c * 8, c * 8, c * 8),
            SelfAttention2d(c * 8, c * 8 // 64),
            ResConvBlock(c * 8, c * 8, c * 4),
            SelfAttention2d(c * 4, c * 4 // 64),
        )

        self.mid_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, down_sample):

        middle_sample = self.mid_1(self.conv(down_sample))
        middle_sample = self.avg_3(self.mid_2(middle_sample))

        return middle_sample

class UpSample(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64
        self.conv_0 = nn.Conv2d(128, 64, kernel_size = 3, padding = 1)
        self.conv_1 = nn.Conv2d(256, 128, kernel_size = 3, padding = 1)
        self.conv_2 = nn.Conv2d(512, 256, kernel_size = 3, padding = 1)

        self.up_1 = nn.Sequential(
            ResConvBlock(c * 8, c * 4, c * 4),
            SelfAttention2d(c * 4, c * 4 // 64),
            ResConvBlock(c * 4, c * 4, c * 2),
            SelfAttention2d(c * 2, c * 2 // 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.up_2 = nn.Sequential(
            ResConvBlock(c * 4, c * 2, c * 2),
            ResConvBlock(c * 2, c * 2, c),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.up_3 = nn.Sequential(
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, is_last=True),
        )

    def forward(self, middle_sample, identity_0, identity_1, identity_2):
        

        up_sample = torch.cat([middle_sample, self.conv_2(identity_2)], dim=1)
        up_sample = self.up_1(up_sample)

        up_sample = torch.cat([up_sample, self.conv_1(identity_1)], dim=1)
        up_sample = self.up_2(up_sample)

        up_sample = torch.cat([up_sample, self.conv_0(identity_0)], dim=1)

        return self.up_3(up_sample)

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64
        self.feature_embed = FeatureEmbedding()
        self.domain_classifier = DomainClassifier()
        self.timestep_embed = FourierFeatures(1, 16)
        self.class_embed = nn.Embedding(11, 4)

        self.common_feature_extractor = DownSample()
        self.source_feature_extractor = DownSample()
        self.target_feature_extractor = DownSample()

        self.mid_sample_net = MidSample()
        self.up_sample_net = UpSample()

    def forward(self, input, t, cond, type = "Source"):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        class_embed = expand_to_planes(self.class_embed(cond + 1), input.shape)

        if (type == "Source"):
            x = torch.cat([input, timestep_embed, class_embed], dim=1)
            private_feature = self.source_feature_extractor(x)
        else:
            feature_embed = self.feature_embed(input)
            x = torch.cat([feature_embed, timestep_embed, class_embed], dim=1)
            private_feature = self.target_feature_extractor(x)

        common_feature = self.common_feature_extractor(x)

        down_sample = torch.cat([common_feature[0], private_feature[0]], dim = 1)           # torch.Size([batch_size, 512, 8, 8])
        
        middle_sample = self.mid_sample_net(down_sample)                                    # torch.Size([batch_size, 256, 8, 8])

        reverse_feature = ReverseLayerF.apply(middle_sample, 0.005)
        domain_output = self.domain_classifier(reverse_feature)

        identity_0  = torch.cat([common_feature[1], private_feature[1]], dim = 1)           # torch.Size([batch_size, 128, 32, 32])
        identity_1  = torch.cat([common_feature[2], private_feature[2]], dim = 1)           # torch.Size([batch_size, 256, 16, 16])
        identity_2  = torch.cat([common_feature[3], private_feature[3]], dim = 1)           # torch.Size([batch_size, 512, 8, 8])

        up_sample = self.up_sample_net(middle_sample, identity_0, identity_1, identity_2)   # torch.Size([batch_size, 3, 32, 32])

        return up_sample, common_feature, private_feature, domain_output
    

