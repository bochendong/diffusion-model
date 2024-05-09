import os
import glob
import torch
from torch import optim, nn
import matplotlib.pyplot as plt

from LoadDataset import load_data_set
from Train import train_diffusion
from Models import Diffusion
from Loss import DiffLoss

# Constants
OUTPUT_DIR = "./output"
BATCH_SIZE = 100
EPOCHS = 10
STEPS = 500
ETA = 1.0
ALPHA = 0.1
GAMMA = 1.0
EMA_DECAY = 0.999
GUIDANCE_SCALE = 2.0
LEARNING_RATE = 4e-4

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_output_directory(output_dir):
    """Ensure output directory exists and is clean."""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    files = glob.glob(os.path.join(output_dir, "*.png"))
    for f in files:
        os.remove(f)

def adjust_learning_rate(optimizer, epoch, initial_lr, decay_interval=30, decay_factor=0.5):
    """Adjusts the learning rate according to the given decay schedule."""
    lr = initial_lr * (decay_factor ** (epoch // decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def plot_loss_history(loss_history, epochs, output_dir):
    loss_names = [
        "Diffusion Loss History",
        "Domain Similarity Loss History (Src)",
        "Domain Similarity Loss History (Tgt)",
        "Domain Diff Loss History (Src)",
        "Domain Diff Loss History (Tgt)"
    ]
    for i, loss in enumerate(loss_history):
        plt.figure()
        plt.plot(range(epochs), loss)
        plt.title(loss_names[i])
        plt.savefig(os.path.join(output_dir, f"{loss_names[i]}.png"))
        plt.close()

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
    
    plot_loss_history(loss_history, EPOCHS, OUTPUT_DIR)

if __name__ == "__main__":
    main()
