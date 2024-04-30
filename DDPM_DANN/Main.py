import os
import glob
import torch
from torch import optim, nn

from LoadDataset import load_data_set
from Train import train_model
from Models import Diffusion


if (os.path.exists("./output")) == False:
    os.mkdir("output")

files = glob.glob("./output/*.png")

for f in files:
    os.remove(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
epoches = 30

source_dl, target_dl, test_dl = load_data_set(batch_size = batch_size)
criterion = nn.NLLLoss()
model = Diffusion().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

train_model(model, optimizer, source_dl, target_dl, criterion = criterion, 
            epoches=epoches, device=device, batch_size = batch_size)


