# Product Context: GCN Starter

## Problem Solved

Provides a foundational, well-tested, and documented starting point for projects requiring a Graph Convolutional Network. This avoids the need to implement a basic GCN from scratch for every new graph-based machine learning task.

## How It Works

-   The core GCN model is defined in `main.py`.
-   It takes graph data (nodes features `x`, edge structure `edge_index`, and optionally labels `y`) as input, typically via a `DataLoader`.
-   The `fit` method trains the model using the provided data, optimizer, and loss function, incorporating early stopping and checkpointing.
-   The `predict` method allows using the trained model to generate predictions for new graph data.
-   Unit tests in `test_gcn.py` verify the functionality of the model components.
-   `README.md` guides users on setup and basic usage.

## User Experience Goals

-   **Easy Setup:** Users should be able to quickly set up the environment using `uv`.
-   **Clear Usage:** The `README.md` and code comments should make it straightforward to understand how to use the GCN model.
-   **Reliable:** The model should be robust, verified by comprehensive unit tests.
-   **Extensible:** The codebase should be clean and well-structured, allowing for easy extension and modification for specific use cases.
