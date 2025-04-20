# Graph Convolutional Network (GCN) Implementation

## Data Generation

To generate data for the GCN model, you can use the following code snippet:

```python
import torch
from torch_geometric.data import Data

# Generate dummy data
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 16)  # 3 nodes, 16 features
y = torch.tensor([0, 1, 0], dtype=torch.long)  # Example labels

data = Data(x=x, edge_index=edge_index, y=y)
```

You can modify the `edge_index`, `x`, and `y` tensors to create different graph structures and node features.

## Debugging with Sample Data

To debug the GCN model, you can use the sample data provided in the `main.py` file. This data consists of a small graph with 3 nodes and 2 edges. You can run the `main.py` file to see the output of the model on this sample data.

To run the `main.py` file, you can use the following command:

```bash
python main.py
```

This will print the output of the model and the predicted classes for each node. You can then compare the predicted classes with the actual labels to see if the model is performing correctly.

## Running Unit Tests

To run the unit tests, you need to install `pytest`:

```bash
pip install pytest
```

Then, you can run the tests using the following command:

```bash
pytest
```

This will run all the tests in the `test_gcn.py` file and report the results.
