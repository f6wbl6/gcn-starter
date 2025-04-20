# Progress: GCN Starter

## Current Status

-   Core GCN model implemented in `main.py`.
-   Model includes methods for training (`fit`, `train_step`), validation (`validate`), and prediction (`predict`).
-   Training loop supports batching via `DataLoader`, early stopping, checkpointing, and basic CSV logging.
-   Unit tests (`test_gcn.py`) cover core functionalities, including batch learning and prediction. All tests are currently passing (with one ignorable warning about return values in `test_gcn_large_dataset`).
-   `README.md` provides basic usage instructions.
-   Project dependencies are managed using `uv` and `requirements.txt`.
-   Code formatting is enforced by `ruff`.
-   Automation via `.clinerules` ensures tests and formatting run on file changes.
-   Memory Bank system has been initialized with the core files.

## What Works

-   GCN model definition and forward pass.
-   Training loop (`fit`) with batching, early stopping, checkpointing, and logging.
-   Prediction using the `predict` method.
-   Unit tests verifying model components and training process.
-   Dependency management with `uv`.
-   Automated code formatting and testing via `.clinerules`.

## What's Left to Build

-   (No outstanding core requirements from the initial task description).
-   Potential improvements:
    -   More sophisticated validation/testing strategy in `fit` (e.g., using separate validation/test datasets/DataLoaders instead of reusing the training loader).
    -   Addressing the `PytestReturnNotNoneWarning` in `test_gcn_large_dataset`.
    -   More detailed README examples.
    -   Integration with a more robust experiment tracking system (e.g., MLflow server instead of just CSV).

## Known Issues

-   `test_gcn_large_dataset` returns the generated `Data` object, causing a `PytestReturnNotNoneWarning`. This doesn't affect test correctness but should be fixed for future pytest compatibility.
-   Validation and testing within the `fit` loop currently reuse the training `DataLoader`. This is not ideal for realistic model evaluation and should ideally use separate datasets/loaders.

## Evolution of Decisions

-   Initially used `pip` for dependencies, switched to `uv` as per `.clinerules`.
-   Initially used `print` for logging, switched to the `logging` module.
-   Refactored the `test` method into the `predict` method for clarity and removed redundancy.
-   Refactored the `fit` method to separate epoch-level logic (`_train_epoch`, `_validate_epoch`, `_test_epoch`).
-   Introduced `.clinerules` for automated testing and formatting.
-   Initialized the Memory Bank system.
