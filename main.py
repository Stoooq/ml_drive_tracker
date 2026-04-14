import json
from pathlib import Path
from src.dataset import Bdd100kDataset


def main():
    root_dir = Path("datasets/bdd100k")

    dataset = Bdd100kDataset(root_dir, split="train")

    image, target = dataset[1]

    print(target)

    print("Hello from ml-drive-tracker!")


if __name__ == "__main__":
    main()
