import os
import glob
import torch
from torch.nn import functional as F
from torchvision.utils import save_image

from Models import get_alphas_sigmas, sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng = torch.quasirandom.SobolEngine(1, scramble=True)

steps = 500
eta = 1.
ema_decay = 0.999
guidance_scale = 2.

def generate_diffussion_target(images, labels):
    t = rng.draw(labels.shape[0])[:, 0].to(device)

    alphas, sigmas = get_alphas_sigmas(t)

    alphas = alphas[:, None, None, None]
    sigmas = sigmas[:, None, None, None]

    noise = torch.randn_like(images)
    noised_reals = images * alphas + noise * sigmas

    targets = noise * alphas - images * sigmas

    return t, noised_reals, targets

def train_model(model, optimizer, source_dl, target_dl, criterion, epoches, device, batch_size):
    src_domain_label = torch.zeros(batch_size).long().to(device)
    tgt_domain_label = torch.ones(batch_size).long().to(device)

    for epoch in range(epoches):
        total_src_loss = 0
        total_tgt_loss = 0
        total_diff_loss = 0
        target_iter = iter(target_dl)
        model.train()
        for src_images, src_labels in source_dl:
            # Source
            src_images, src_labels = src_images.to(device), src_labels.to(device)

            t, noised_src, src_recon_targets = generate_diffussion_target(src_images, src_labels)

            optimizer.zero_grad()

            to_drop = torch.rand(src_labels.shape, device=src_labels.device).le(0.2)
            classes_drop = torch.where(to_drop, -torch.ones_like(src_labels), src_labels)

            output, domain_output = model(noised_src, t, classes_drop, type = "Source")
            diffused_loss = F.mse_loss(output, src_recon_targets)

            loss_s_domain = criterion(domain_output, src_domain_label)

            # Target
            tgt_images, tgt_labels = next(target_iter)
            tgt_images, tgt_labels = tgt_images.to(device), tgt_labels.to(device)

            output, domain_output = model(tgt_images, t, tgt_labels, type = "Target")

            loss_t_domain = criterion(domain_output, tgt_domain_label)

            # TODO: It should be a class recon classifier here

            loss = loss_s_domain + loss_t_domain + diffused_loss
            loss.backward()
            
            optimizer.step()

            total_src_loss += loss_s_domain
            total_tgt_loss += loss_t_domain
            total_diff_loss = diffused_loss

        print(f"Epoch {epoch+1}:")
        print("src", total_src_loss.item(), 'tgt', total_tgt_loss.item(), 'diff', total_diff_loss.item())
        noise = torch.randn([10, 3, 32, 32], device=device)
        fakes_classes = torch.arange(10, device=device)
        fakes = sample(model, noise, steps, eta, fakes_classes, guidance_scale)
        fakes = (fakes + 1) / 2
        fakes = torch.clamp(fakes, min=0, max = 1)
        save_image(fakes.data, './output/%03d_train.png' % epoch)
