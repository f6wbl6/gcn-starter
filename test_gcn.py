import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pathlib import Path
import torch.optim as optim # Import optim
import torch.nn as nn # Import nn

from main import GCN  # Import the GCN class from main.py
from config import settings # Import settings for potential default overrides if needed

@pytest.fixture
def dummy_data():
    # Generate dummy data
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn(3, 16)  # 3 nodes, 16 features
    y = torch.tensor([0, 1, 0], dtype=torch.long)  # Example labels
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


# --- Helper Functions for Assertions ---

def _assert_checkpoint_exists(tmp_path: Path, filename: str):
    """Asserts that the checkpoint file exists."""
    # Check within the configured directory relative to tmp_path
    assert (tmp_path / settings.checkpoint_dir / filename).exists(), f"Checkpoint file '{filename}' should exist in {settings.checkpoint_dir}"

def _assert_log_file_valid(tmp_path: Path, filename: str):
    """Asserts that the log file exists and has the correct format."""
    # Check within the configured directory relative to tmp_path
    log_path = tmp_path / settings.log_dir / filename
    assert log_path.exists(), f"MLflow log file '{filename}' should exist in {settings.log_dir}"
    with open(log_path, "r") as f:
        lines = f.readlines()
        assert len(lines) > 1, (
            f"MLflow log file '{filename}' should contain more than just the header"
        )
        header = lines[0].strip()
        assert header == "epoch,train_loss,val_loss,test_acc", (
            f"MLflow log file '{filename}' header is incorrect: {header}"
        )
        data_line = lines[1].strip().split(",")
        assert len(data_line) == 4, (
            f"MLflow log file '{filename}' data line has incorrect number of columns: {len(data_line)}"
        )
        try:
            # Attempt to parse, ensuring data types are correct
            int(data_line[0])
            float(data_line[1])
            float(data_line[2])
            float(data_line[3])
        except ValueError as e:
            raise AssertionError(
                f"MLflow log file '{filename}' data line contains invalid data: {e}"
            )

# --- Test Functions ---

def test_gcn_output_shape(dummy_data):
    # Model parameters
    num_node_features = dummy_data.num_node_features
    num_classes = 2  # Binary classification

    # Initialize the model
    model = GCN(num_node_features, num_classes)

    # Forward pass
    out = model(dummy_data)

    # Unit Test (basic shape check)
    assert out.shape == torch.Size([3, 2]), "Output shape is incorrect"


def test_gcn_output_values(dummy_data):
    # Model parameters
    num_node_features = dummy_data.num_node_features
    num_classes = 2  # Binary classification

    # Initialize the model
    model = GCN(num_node_features, num_classes)

    # Forward pass
    out = model(dummy_data)

    # Assert that the output values are log probabilities (negative values)
    assert torch.all(out <= 0), "Output values are not log probabilities"


def test_gcn_train_step(dummy_data):
    # Model parameters
    num_node_features = dummy_data.num_node_features
    num_classes = 2  # Binary classification

    # Initialize the model
    model = GCN(num_node_features, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    # Train step
    loss = model.train_step(dummy_data, optimizer, criterion)

    # Assert that the loss is a float and is non-negative
    assert isinstance(loss, float), "Loss should be a float"
    assert loss >= 0, "Loss should be non-negative"


def test_gcn_validate(dummy_data):
    # Model parameters
    num_node_features = dummy_data.num_node_features
    num_classes = 2  # Binary classification

    # Initialize the model
    model = GCN(num_node_features, num_classes)
    criterion = nn.NLLLoss()

    # Validation
    loss = model.validate(dummy_data, criterion)

    # Assert that the loss is a float and is non-negative
    assert isinstance(loss, float), "Loss should be a float"
    assert loss >= 0, "Loss should be non-negative"


def test_gcn_fit(dummy_data, tmp_path):
    # Model parameters
    num_node_features = dummy_data.num_node_features
    num_classes = 2  # Binary classification

    # Initialize the model
    model = GCN(num_node_features, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    # Set checkpoint and log paths using tmp_path and configured subdirs
    checkpoint_filename = "fit_checkpoint.pth"
    mlflow_log_filename = "fit_mlflow_log.csv"
    # Construct paths within the temporary directory, respecting configured subdirs
    checkpoint_path = str(tmp_path / settings.checkpoint_dir / checkpoint_filename)
    mlflow_log_path = str(tmp_path / settings.log_dir / mlflow_log_filename)


    # Create a DataLoader for batch learning
    data_loader = DataLoader([dummy_data], batch_size=2, shuffle=True)

    # Fit the model
    model.fit(
        data_loader,
        optimizer,
        criterion,
        epochs=5,
        patience=2,
        checkpoint_path=checkpoint_path, # Pass the full temp path
        mlflow_log_path=mlflow_log_path, # Pass the full temp path
    )

    # Assertions using helper functions (passing tmp_path and only the filename)
    _assert_checkpoint_exists(tmp_path, checkpoint_filename)
    _assert_log_file_valid(tmp_path, mlflow_log_filename)


def test_gcn_batch_learning(dummy_data, tmp_path):
    # Model parameters
    num_node_features = dummy_data.num_node_features
    num_classes = 2  # Binary classification

    # Initialize the model
    model = GCN(num_node_features, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    # Set checkpoint and log paths using tmp_path and configured subdirs
    checkpoint_filename = "batch_checkpoint.pth"
    mlflow_log_filename = "batch_mlflow_log.csv"
    # Construct paths within the temporary directory, respecting configured subdirs
    checkpoint_path = str(tmp_path / settings.checkpoint_dir / checkpoint_filename)
    mlflow_log_path = str(tmp_path / settings.log_dir / mlflow_log_filename)

    # Create a DataLoader for batch learning
    data_loader = DataLoader([dummy_data], batch_size=2, shuffle=True)

    # Fit the model
    model.fit(
        data_loader,
        optimizer,
        criterion,
        epochs=5,
        patience=2,
        checkpoint_path=checkpoint_path, # Pass the full temp path
        mlflow_log_path=mlflow_log_path, # Pass the full temp path
    )

    # Assertions using helper functions (passing tmp_path and only the filename)
    _assert_checkpoint_exists(tmp_path, checkpoint_filename)
    _assert_log_file_valid(tmp_path, mlflow_log_filename)


def test_gcn_predict(dummy_data):
    # Model parameters
    num_node_features = dummy_data.num_node_features
    num_classes = 2  # Binary classification

    # Initialize the model
    model = GCN(num_node_features, num_classes)

    # Forward pass
    predictions = model.predict(dummy_data)

    # Assert that the output shape is correct
    assert predictions.shape == torch.Size([3]), "Output shape is incorrect"

    # Assert that the output values are integers (class labels)
    assert torch.all(predictions >= 0)
    assert torch.all(predictions < num_classes)
