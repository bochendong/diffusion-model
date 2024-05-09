import torch
import math
from torch import optim, nn
from torch.nn import functional as F
from torchvision.utils import save_image


rng = torch.quasirandom.SobolEngine(1, scramble=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_alphas_sigmas(t):
    """
    Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep.
    """
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

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
                     optimizer, criterion, diff_loss, 
                     src_domain_label, tgt_domain_label,
                     alpha, gamma,
                     steps, eta, ema_decay, guidance_scale, scheduler):
    
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

    scheduler(optimizer, epoch)

    print(f"Epoch {epoch+1}:")
    print('Diffusion_loss', diffused_loss.item())
    print("Domain Similarity Loss", 'src: ', loss_s_domain.item(), 'tgt: ', loss_t_domain.item())
    print('Domain Diff Loss', 'src: ', diff_loss_src.item(), 'tgt: ', diff_loss_tgt.item())
    noise = torch.randn([10, 3, 32, 32], device=device)
    fakes_classes = torch.arange(10, device=device)
    fakes = sample(model, noise, steps, eta, fakes_classes, guidance_scale)
    fakes = (fakes + 1) / 2
    fakes = torch.clamp(fakes, min=0, max = 1)
    save_image(fakes.data, './output/%03d_train.png' % epoch)

    return loss.item(), loss_s_domain.item(), loss_t_domain.item(), diff_loss_src.item(), diff_loss_tgt.item()