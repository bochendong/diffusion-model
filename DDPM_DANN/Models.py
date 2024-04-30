import torch
from torch import optim, nn
import math
from torch.autograd import Function
from torchvision.models import resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_alphas_sigmas(t):
    """
    Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep.
    """
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

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
        self.fc1 = nn.Linear(1000, 2048)
        self.fc2 = nn.Linear(2048, 3072)

        self.pretrained_model = resnet50(pretrained=True)
        self.pretrained_model = self.pretrained_model.to(device)

    def forward(self, x):
        with torch.no_grad():
            x = self.pretrained_model(x)
        x = self.fc1(x)
        x = self.fc2(x)
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
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 2),
            nn.LogSoftmax(dim = 1)
        )

    def forward(self, input):
        input = input.view(-1, 256 * 8 * 8)
        return self.net(input)
    
def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64
        self.feature_embed = FeatureEmbedding()
        self.domain_classifier = DomainClassifier()
        self.timestep_embed = FourierFeatures(1, 16)
        self.class_embed = nn.Embedding(11, 4)

        self.avg_1 = nn.AvgPool2d(2)
        self.avg_2 = nn.AvgPool2d(2)
        self.avg_3 = nn.AvgPool2d(2)

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

        self.down_3 = nn.Sequential(
            ResConvBlock(c * 4, c * 8, c * 8),
            SelfAttention2d(c * 8, c * 8 // 64),
            ResConvBlock(c * 8, c * 8, c * 8),
            SelfAttention2d(c * 8, c * 8 // 64),
            ResConvBlock(c * 8, c * 8, c * 8),
            SelfAttention2d(c * 8, c * 8 // 64),
            ResConvBlock(c * 8, c * 8, c * 4),
            SelfAttention2d(c * 4, c * 4 // 64),
        )

        self.up_0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

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

    def forward(self, input, t, cond, type = "Target"):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        class_embed = expand_to_planes(self.class_embed(cond + 1), input.shape)

        if (type == "Target"):
            feature_embed = self.feature_embed(input)
            x = torch.cat([feature_embed, timestep_embed, class_embed], dim=1)
        else:
            x = torch.cat([input, timestep_embed, class_embed], dim=1)

        # Down Sample
        down_sample = self.down_0(x)
        identity_0 = nn.Identity()(down_sample)

        down_sample = self.avg_1(self.down_1(down_sample))
        identity_1 = nn.Identity()(down_sample)

        down_sample = self.avg_2(self.down_2(down_sample))
        identity_2 = nn.Identity()(down_sample)

        # Middle Sample
        middle_sample = self.down_3(down_sample)
        middle_sample = self.avg_3(self.up_0(middle_sample))

        reverse_feature = ReverseLayerF.apply(middle_sample, 0.01)
        domain_output = self.domain_classifier(reverse_feature)

        # Up Sample
        up_sample = torch.cat([middle_sample, identity_2], dim=1)
        up_sample = self.up_1(up_sample)

        up_sample = torch.cat([up_sample, identity_1], dim=1)
        up_sample = self.up_2(up_sample)

        up_sample = torch.cat([up_sample, identity_0], dim=1)

        return self.up_3(up_sample), domain_output
    
@torch.no_grad()
def sample(model, x, steps, eta, classes, guidance_scale=1.):
    """
    Draws samples from a model given starting noise.
    """
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in range(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            x_in = torch.cat([x, x])
            ts_in = torch.cat([ts, ts])
            classes_in = torch.cat([-torch.ones_like(classes), classes])
            v_uncond, v_cond = model(x_in, ts_in * t[i], classes_in)[0].float().chunk(2)
        v = v_uncond + guidance_scale * (v_cond - v_uncond)

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred
