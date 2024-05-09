import os
import glob
import torch
from torch import optim, nn

from LoadDataset import load_data_set
from Train import train_diffusion, DiffLoss
from Models import Diffusion


if (os.path.exists("./output")) == False:
    os.mkdir("output")

files = glob.glob("./output/*.png")

for f in files:
    os.remove(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
epoches = 100

torch.cuda.empty_cache()

source_dl, target_dl, test_dl = load_data_set(batch_size = batch_size)
criterion = nn.NLLLoss()
diff_loss = DiffLoss()
model = Diffusion().to(device)
optimizer = optim.Adam(model.parameters(), lr = 2e-4)

epoch = 0
src_domain_label = torch.zeros(batch_size).long().to(device)
tgt_domain_label = torch.ones(batch_size).long().to(device)

while epoch < epoches:
    train_diffusion(epoch, model, source_dl, target_dl, 
                     0.1, 0.1,
                     optimizer, criterion, diff_loss, 
                     src_domain_label, tgt_domain_label)
    epoch += 1
