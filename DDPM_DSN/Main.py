import os
import glob
import json
import torch
from torch import optim, nn
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from LoadDataset import load_data_set
from Train import train_diffusion, train_model, sample
from Models import Diffusion
from Loss import DiffLoss

# Constants
OUTPUT_DIR = "./output"
LOG_FILE = os.path.join(OUTPUT_DIR, "loss_history.json")
BATCH_SIZE = 150
EPOCHS = 150
STEPS = 500
ETA = 1.0
ALPHA = 0.1
GAMMA = 1.0
EMA_DECAY = 0.999
GUIDANCE_SCALE = 2.0
LEARNING_RATE = 8e-4
TRANSFER_START_EPOCH = 50
CIFAR_10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def adjust_learning_rate(optimizer, epoch, initial_lr, decay_interval=25, decay_factor=0.5):
    """Adjusts the learning rate according to the given decay schedule."""
    lr = initial_lr * (decay_factor ** (epoch // decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def setup_output_directory(output_dir):
    """Ensure output directory exists and is clean."""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    files = glob.glob(os.path.join(output_dir, "*.png"))
    for f in files:
        os.remove(f)

def plot_loss_history(loss_history, output_dir):
    loss_names = [
        "Diffusion Loss History",
        "Domain Similarity Loss History (Src)",
        "Domain Similarity Loss History (Tgt)",
        "Domain Diff Loss History (Src)",
        "Domain Diff Loss History (Tgt)"
    ]
    for i, loss in enumerate(loss_history):
        plt.figure()
        plt.plot(range(len(loss[i])), loss)
        plt.title(loss_names[i])
        plt.savefig(os.path.join(output_dir, f"{loss_names[i]}.png"))
        plt.close()

def save_loss_history(loss_history, log_file):
    """Saves the loss history to a JSON file."""
    loss_data = {
        "Diffusion Loss History": loss_history[0],
        "Domain Similarity Loss History (Src)": loss_history[1],
        "Domain Similarity Loss History (Tgt)": loss_history[2],
        "Domain Diff Loss History (Src)": loss_history[3],
        "Domain Diff Loss History (Tgt)": loss_history[4]
    }
    with open(log_file, 'w') as file:
        json.dump(loss_data, file, indent=4)

def generate_class(model, epoch, label, steps, eta, guidance_scale, device):
    noise = torch.randn([10, 3, 32, 32], device=device)
    fakes_classes = torch.ones(10, device=device, dtype=torch.long) * label
    fakes = sample(model, noise, steps, eta, fakes_classes, guidance_scale)
    fakes = (fakes + 1) / 2
    fakes = torch.clamp(fakes, min=0, max = 1)

    save_image(fakes.data, f'./output/{CIFAR_10_CLASSES[label]}_{epoch}_output.png')

def main():
    setup_output_directory(OUTPUT_DIR)
    torch.cuda.empty_cache()

    source_dl, target_dl, test_dl = load_data_set(batch_size=BATCH_SIZE)
    
    criterion = nn.NLLLoss()
    diff_loss = DiffLoss()
    model = Diffusion().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    adjust_lr = lambda opt, ep: adjust_learning_rate(opt, ep, LEARNING_RATE)
    
    src_domain_label = torch.zeros(BATCH_SIZE).long().to(DEVICE)
    tgt_domain_label = torch.ones(BATCH_SIZE).long().to(DEVICE)

    loss_history = [[], [], [], [], []]

    for epoch in range(EPOCHS):
        if (epoch < TRANSFER_START_EPOCH):
            epoch_loss = train_diffusion(epoch, model, source_dl, optimizer, 
                     steps=STEPS, eta=ETA, ema_decay=EMA_DECAY, 
                    guidance_scale=GUIDANCE_SCALE, scheduler=adjust_lr
            )
            loss_history[0].append(epoch_loss)
        else:
            epoch_loss = train_diffusion(
                epoch, model, source_dl, target_dl, 
                optimizer, criterion, diff_loss, 
                src_domain_label, tgt_domain_label, 
                alpha = ALPHA, gamma = GAMMA,
                steps=STEPS, eta=ETA, ema_decay=EMA_DECAY, 
                guidance_scale=GUIDANCE_SCALE, scheduler=adjust_lr
            )
            
            for i in range(len(loss_history)):
                loss_history[i].append(epoch_loss[i])

        if ((epoch + 1) % 25 == 0):
            for label in range(0, 10):
                generate_class(model, epoch,label, steps = STEPS, eta = ETA, 
                    guidance_scale = GUIDANCE_SCALE, 
                    evice = DEVICE)
    
    plot_loss_history(loss_history, OUTPUT_DIR)
    save_loss_history(loss_history, LOG_FILE)

if __name__ == "__main__":
    main()
