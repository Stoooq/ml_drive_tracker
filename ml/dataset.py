import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class Bdd100kDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.images_dir = Path(f"{root_dir}/bdd100k/bdd100k/images/10k/{split}")
        self.labels_path = Path(
            f"{root_dir}/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_{split}.json",
        )

        with Path.open(self.labels_path) as json_data:
            all_labels = json.load(json_data)

        valid_images = {f.name for f in self.images_dir.iterdir() if f.is_file()}

        self.labels = [lbl for lbl in all_labels if lbl["name"] in valid_images]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_name = label["name"]

        image_path = Path(f"{self.images_dir}/{image_name}")
        image = self.retrieve_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        boxes, labels = self.retrieve_boxes(label)

        target = {"boxes": boxes, "labels": labels}

        return image, target

    def retrieve_image(self, path):
        with Image.open(path) as img:
            img.verify()

        image = Image.open(path)
        image.load()

        return image

    def retrieve_boxes(self, label):
        boxes = []
        labels = []

        annotations = label.get("labels", [])

        for annotation in annotations:
            if "box2d" in annotation:
                box = annotation["box2d"]
                boxes.append(
                    [box["x1"], box["y1"], box["x2"], box["y2"]],
                )
                labels.append(annotation["id"])

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        return boxes, labels

    def load_and_correct_labels(self):
        for file in self.labels_path.iterdir():
            print(file)
