from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from config import ModelSettings, model_settings, settings
from logger import set_logger

# Initialize logger using the custom function
logger = set_logger()


class GCN(nn.Module):
    def __init__(self, settings: ModelSettings):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(settings.num_node_features, settings.hidden_dim)
        self.conv2 = GCNConv(settings.hidden_dim, settings.num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, _ = (
            data.x,
            data.edge_index,
            data.batch,
        )  # Use _ for unused batch variable

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    def train_step(
        self, data: Data, optimizer: optim.Optimizer, criterion: nn.Module
    ) -> float:
        self.train()
        optimizer.zero_grad()
        out = self(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def validate(self, data: Data, criterion: nn.Module) -> float:
        self.eval()
        with torch.no_grad():
            out = self(data)
            loss = criterion(out, data.y)
        return loss.item()

    def predict(self, data: Data) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            out = self(data)
            pred = out.argmax(dim=1)
        return pred

    def get_embeddings(self, data: Data) -> torch.Tensor:
        """Extracts node embeddings from the second-to-last layer."""
        self.eval()
        with torch.no_grad():
            x, edge_index, _ = data.x, data.edge_index, data.batch
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            # Embeddings are the output of the first GCN layer after ReLU
            # Optionally, could add dropout here if desired during inference, but typically not.
            # x = F.dropout(x, training=self.training) # Usually False during eval
        return x

    def fit(
        self,
        data_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epochs: int = 100,
        patience: int = 10,
        # Use settings for default paths, but allow override
        checkpoint_path: str = None,
        mlflow_log_path: str = None,
    ):
        """
        Trains the GCN model with early stopping and checkpoint saving using a DataLoader.

        Args:
            data_loader (torch_geometric.loader.DataLoader): The DataLoader for training data.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            criterion (torch.nn.Module): The loss function to use.
            epochs (int): The number of epochs to train for.
            patience (int): The number of epochs to wait before early stopping.
            checkpoint_path (str, optional): The path to save the best model checkpoint.
                                            Defaults to path constructed from settings.
            mlflow_log_path (str, optional): The path to save the MLflow-compatible CSV log.
                                             Defaults to path constructed from settings.
        """
        # Construct default paths if not provided
        if checkpoint_path is None:
            checkpoint_path = str(settings.checkpoint_dir / "checkpoint.pth")
        if mlflow_log_path is None:
            mlflow_log_path = str(settings.log_dir / "mlflow_log.csv")

        # Ensure directories exist (config.py already does this, but good practice here too)
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        Path(mlflow_log_path).parent.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_model_state = None

        # Prepare MLflow-compatible CSV log
        with open(mlflow_log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,test_acc\n")

        for epoch in range(epochs):
            # Train step
            train_loss = self._train_epoch(data_loader, optimizer, criterion)

            # Validation
            val_loss = self._validate_epoch(data_loader, criterion)

            # Test
            test_acc = self._test_epoch(data_loader)

            # Log metrics to CSV
            with open(mlflow_log_path, "a") as f:
                f.write(f"{epoch},{train_loss},{val_loss},{test_acc}\n")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = self.state_dict()
                torch.save(best_model_state, checkpoint_path)
                logger.info(
                    f"Epoch {epoch}: Validation loss improved to {best_val_loss:.4f}. Saving checkpoint."
                )
            else:
                epochs_without_improvement += 1
                logger.info(
                    f"Epoch {epoch}: Validation loss did not improve. Patience: {epochs_without_improvement}/{patience}"
                )
                if epochs_without_improvement > patience:
                    logger.info("Early stopping triggered")
                    break

        # Load best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            logger.info(f"Loaded best model from {checkpoint_path}")
        else:
            logger.info("No improvement during training, using initial model")

    def _train_epoch(
        self, data_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module
    ) -> float:
        self.train()
        total_loss = 0
        num_batches = 0
        for data in data_loader:
            loss = self.train_step(data, optimizer, criterion)
            total_loss += loss
            num_batches += 1
        # Avoid division by zero if data_loader is empty
        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate_epoch(self, data_loader: DataLoader, criterion: nn.Module) -> float:
        self.eval()
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for data in data_loader:
                loss = self.validate(data, criterion)
                total_loss += loss
                num_batches += 1
        # Avoid division by zero if data_loader is empty
        return total_loss / num_batches if num_batches > 0 else float("inf")

    def _test_epoch(self, data_loader: DataLoader) -> float:
        self.eval()
        total_acc = 0
        num_batches = 0
        with torch.no_grad():
            for data in data_loader:
                pred = self.predict(data).cpu()
                correct = (pred == data.y).sum().item()
                # Ensure num_nodes is not zero before division
                acc = correct / data.num_nodes if data.num_nodes > 0 else 0.0
                total_acc += acc
                num_batches += 1
        # Avoid division by zero if data_loader is empty
        return total_acc / num_batches if num_batches > 0 else 0.0


if __name__ == "__main__":
    # Note: Logging is configured globally when logger is initialized

    # Generate dummy data
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn(3, 16)  # 3 nodes, 16 features
    y = torch.tensor([[0], [1], [0]], dtype=torch.long)  # Example labels
    data = Data(x=x, edge_index=edge_index, y=y)

    # Initialize the model parameters
    # num_node_features = data.num_node_features
    # num_classes = 2  # Binary classification

    # Initialize the model, passing in settings
    model = GCN(model_settings)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    from torch_geometric.loader import DataLoader

    data_loader = DataLoader([data], batch_size=2, shuffle=True)

    # Construct paths using settings
    checkpoint_file = "main_checkpoint.pth"
    log_file = "main_mlflow_log.csv"
    checkpoint_path = str(settings.checkpoint_dir / checkpoint_file)
    mlflow_log_path = str(settings.log_dir / log_file)

    # Fit the model
    model.fit(
        data_loader,
        optimizer,
        criterion,
        epochs=5,  # Reduced epochs for quick demo
        patience=2,
        checkpoint_path=checkpoint_path,
        mlflow_log_path=mlflow_log_path,
    )

    # Predict
    predicted_classes = model.predict(data)
    print(f"Predicted classes: {predicted_classes}")
