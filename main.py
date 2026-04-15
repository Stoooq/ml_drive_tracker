import json
from pathlib import Path

from src.dataset import Bdd100kDataset
from.src.data_manager import DataManager


def main():
    root_dir = Path("datasets/bdd100k")

    data_manager = DataManager(root_dir=root_dir, batch_size=32)

    train_loader, val_loader, test_loader = data_manager.build_dataloaders()

    print("Hello from ml-drive-tracker!")


if __name__ == "__main__":
    main()
