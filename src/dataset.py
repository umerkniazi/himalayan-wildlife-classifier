from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path


def get_dataloaders(data_dir: str, batch_size: int = 16):
    data_path = Path(data_dir)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
    }

    image_datasets = {
        split: datasets.ImageFolder(
            data_path / split,
            data_transforms[split]
        )
        for split in ['train', 'val']
    }

    dataloaders = {
        'train': DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            shuffle=True
        ),
        'val': DataLoader(
            image_datasets['val'],
            batch_size=batch_size,
            shuffle=False
        )
    }

    dataset_sizes = {
        split: len(image_datasets[split])
        for split in ['train', 'val']
    }

    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names