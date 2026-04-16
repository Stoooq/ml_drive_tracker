import torch
from tqdm import tqdm


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, epochs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
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

            output = self.model(X_batch)

            loss = self.loss_fn(output, y_batch.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() / len(y)

        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss

    def _validate_epoch(self) -> float:
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for X, y in self.val_loader:
                X_batch = X.to(self.device)
                y_batch = y.to(self.device)

                output = self.model(X_batch)

                loss = self.loss_fn(output, y_batch.unsqueeze(1))

                epoch_loss = loss.item() / len(y)

        avg_loss = epoch_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            train_loss = self._train_epoch()

            val_loss = self._validate_epoch()

            print(f"Epoch {epoch + 1}: train loss {train_loss}, validation loss {val_loss}")
