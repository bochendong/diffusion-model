from torchvision.transforms import functional as TF
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Dataset, TensorDataset

def load_data_set(batch_size = 64):
    tf = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    source_set = datasets.CIFAR10(root='data', train=True, download=True, transform=tf, target_transform=lambda x: x if x < 8 else -1)
    source_dl = DataLoader(source_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last = True)

    target_set = datasets.CIFAR10(root='data', train=True, download=True, transform=tf, target_transform=lambda x: x if (x == 8) else -1)
    target_dl = DataLoader(target_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last = True)

    test_set = datasets.CIFAR10(root='data', train=True, download=True, transform=tf, target_transform=lambda x: x if (x == 9) else -1)
    test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last = True)

    return source_dl, target_dl, test_dl
