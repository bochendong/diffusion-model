from torchvision.transforms import functional as TF
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset


def load_data_set(batch_size=25):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Load the entire CIFAR10 dataset
    full_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=tf)

    # Create filtered Subsets for each category
    source_indices = [i for i, (_, label) in enumerate(full_dataset) if label < 7]
    target_indices = [i for i, (_, label) in enumerate(full_dataset) if (label == 8 or label == 7)]
    test_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 9]

    source_set = Subset(full_dataset, source_indices)
    target_set = Subset(full_dataset, target_indices)
    test_set = Subset(full_dataset, test_indices)

    # Create DataLoaders for each subset
    source_dl = DataLoader(source_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    target_dl = DataLoader(target_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    return source_dl, target_dl, test_dl
