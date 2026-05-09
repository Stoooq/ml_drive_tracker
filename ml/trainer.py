import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm


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
