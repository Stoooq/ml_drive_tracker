from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from ultralytics import YOLO


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, epochs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.epochs = epochs

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _train_epoch(self) -> float:
        self.model.train()
        epoch_loss = 0.0

        for X, y in self.train_loader:
            X_batch = X.to(self.device)
            y_batch = y.to(self.device)

            losses = self.model(X_batch, y_batch)

            loss = sum(losses.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss

    def _validate_epoch(self) -> float:
        self.model.eval()

        with torch.no_grad():
            metric = MeanAveragePrecision(iou_type="bbox")

            for X, y in self.val_loader:
                X_batch = X.to(self.device)
                y_batch = y.to(self.device)

                output = self.model(X_batch)

                metric.update(output, y_batch)

            metrics = metric.compute()

        return metrics["map"]

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            train_loss = self._train_epoch()

            val_map = self._validate_epoch()

            print(
                f"Epoch {epoch + 1}: train loss {train_loss}, validation map {val_map}",
            )


class YOLOTrainer:
    def __init__(
        self,
        model_path: Path,
        output_path: Path,
        data_path: Path,
        epochs: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.data_path = data_path
        self.epochs = epochs
        self.device = device

        self.model = YOLO(self.model_path)

    def train(self):
        results = self.model.train(
            project=str(self.output_path.parent),
            name=self.output_path.name,
            data=str(self.data_path),
            epochs=self.epochs,
            device=self.device,
            imgsz=640,
        )

        return results


if __name__ == "__main__":
    trainer = YOLOTrainer(
        model_path=Path("storage/models/yolov8n.pt"),
        output_path=Path("storage/"),
        data_path=Path("data/bdd100k.yaml"),
        epochs=50,
        device="mps",
    )

    trainer.train()
