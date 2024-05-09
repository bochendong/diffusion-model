import torch
from torch import optim, nn
from torch.nn import functional as F
from torchvision.utils import save_image

from Models import get_alphas_sigmas, sample

rng = torch.quasirandom.SobolEngine(1, scramble=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

steps = 500
eta = 1.
ema_decay = 0.999
guidance_scale = 2.

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss
    
def generate_diffussion_target(images, labels):
    t = rng.draw(labels.shape[0])[:, 0].to(device)

    alphas, sigmas = get_alphas_sigmas(t)

    alphas = alphas[:, None, None, None]
    sigmas = sigmas[:, None, None, None]

    noise = torch.randn_like(images)
    noised_reals = images * alphas + noise * sigmas

    targets = noise * alphas - images * sigmas

    return t, noised_reals, targets

def train_diffusion(epoch, model, source_dl, target_dl, 
                     alpha, gamma,
                     optimizer, criterion, diff_loss, 
                     src_domain_label, tgt_domain_label):
    
    source_iter = iter(source_dl)
    model.train()

    for tgt_images, tgt_labels in target_dl:
        tgt_images, tgt_labels = tgt_images.to(device), tgt_labels.to(device)
        src_images, src_labels = next(source_iter)
        src_images, src_labels = src_images.to(device), src_labels.to(device)

        t, noised_src, src_recon_targets = generate_diffussion_target(src_images, src_labels)

        optimizer.zero_grad()

        to_drop = torch.rand(src_labels.shape, device=src_labels.device).le(0.2)
        classes_drop = torch.where(to_drop, -torch.ones_like(src_labels), src_labels)

        output, common_feature, private_feature, domain_output = model(noised_src, t, classes_drop, type = "Source")
        diffused_loss = F.mse_loss(output, src_recon_targets)
        loss_s_domain = criterion(domain_output, src_domain_label)
        diff_loss_src = diff_loss(common_feature[0], private_feature[0])

        output, common_feature, private_feature, domain_output = model(tgt_images, t, tgt_labels, type = "Target")
        loss_t_domain = criterion(domain_output, tgt_domain_label)
        diff_loss_tgt = diff_loss(common_feature[0], private_feature[0]) 

        loss = alpha * (loss_s_domain + loss_t_domain) + gamma * (diff_loss_src + diff_loss_tgt) + diffused_loss

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}:")
    print('Diffusion_loss', loss.item())
    print("Domain Similarity Loss", 'src: ', loss_s_domain.item(), 'tgt: ', loss_t_domain.item())
    print('Domain Diff Loss', 'src: ', diff_loss_src.item(), 'tgt: ', diff_loss_tgt.item())
    noise = torch.randn([10, 3, 32, 32], device=device)
    fakes_classes = torch.arange(10, device=device)
    fakes = sample(model, noise, steps, eta, fakes_classes, guidance_scale)
    fakes = (fakes + 1) / 2
    fakes = torch.clamp(fakes, min=0, max = 1)
    save_image(fakes.data, './output/%03d_train.png' % epoch)