import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def calculate_mean_std(dataset):
    preprocess = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ],
    )

    total_pixels = 0
    sum_pixels = torch.zeros(3)

    mean_loader = tqdm(dataset, desc="Pass 1/2: Computing Mean")

    for img, _ in mean_loader:
        img_tensor = preprocess(img)
        pixels = img_tensor.view(3, -1)
        sum_pixels += pixels.sum(dim=1)
        total_pixels += pixels.size(1)

    mean = sum_pixels / total_pixels

    sum_squared_diff = torch.zeros(3)

    std_loader = tqdm(dataset, desc="Pass 2/2: Computing Std")

    for img, _ in std_loader:
        img_tensor = preprocess(img)
        pixels = img_tensor.view(3, -1)
        diff = pixels - mean.unsqueeze(1)
        sum_squared_diff += (diff**2).sum(dim=1)

    std = torch.sqrt(sum_squared_diff / total_pixels)

    return mean, std


def get_transforms(dataset: Dataset | None, mean: float | None, std: float | None):
    if not mean and not std:
        mean, std = calculate_mean_std(dataset)

    augmentations_transforms = [
        transforms.RandomRotation(degrees=(90, 90)),
        transforms.RandomVerticalFlip(p=0.5),
    ]

    main_transforms = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    augmentation_transform = transforms.Compose(
        main_transforms + augmentations_transforms,
    )
    transform = transforms.Compose(main_transforms)

    return augmentation_transform, transform
