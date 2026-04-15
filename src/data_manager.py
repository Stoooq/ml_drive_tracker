from torch.utils.data import DataLoader

from dataset import Bdd100kDataset


class DataManager:
    def __init__(self, root_dir: str, batch_size: int):
        self.root_dir = root_dir
        self.batch_size = batch_size

    def build_dataloaders(self):
        train_dataset, val_dataset, test_dataset = self._get_data()

        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def _get_data(self):
        train_dataset = Bdd100kDataset(self.root_dir, split="train")
        val_dataset = Bdd100kDataset(self.root_dir, split="val")
        test_dataset = Bdd100kDataset(self.root_dir, split="val")

        return train_dataset, val_dataset, test_dataset
