from pathlib import Path

from torch.utils.data import DataLoader

from src.dataset import Bdd100kDataset
from src.utils import collate_fn, get_transforms


class DataManager:
    def __init__(self, root_dir: Path, batch_size: int):
        self.root_dir = root_dir
        self.batch_size = batch_size

    def build_dataloaders(self):
        train_dataset, val_dataset, test_dataset = self._get_data()

        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, self.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False, collate_fn=collate_fn)

        return train_loader, val_loader, test_loader

    def _get_data(self):
        augmentation_transform, transform = get_transforms(
            dataset=None, mean=0.5, std=0.5
        )

        train_dataset = Bdd100kDataset(
            self.root_dir,
            split="train",
            transform=augmentation_transform,
        )
        val_dataset = Bdd100kDataset(self.root_dir, split="val", transform=transform)
        test_dataset = Bdd100kDataset(self.root_dir, split="val", transform=transform)

        return train_dataset, val_dataset, test_dataset
