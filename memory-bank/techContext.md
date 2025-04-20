# Tech Context: GCN Starter

## Core Technologies

-   **Python**: Primary programming language (version specified in `.python-version`, likely 3.11 based on previous logs).
-   **PyTorch**: Core deep learning framework.
-   **PyTorch Geometric (PyG)**: Extension library for graph neural networks in PyTorch. Used for `GCNConv` layers and `Data`/`DataLoader` objects.
-   **pydantic-settings**: Used for managing configuration, including model parameters and file paths.

## Development Environment & Tooling

-   **Package Management**: `uv` is used for installing and managing Python dependencies, as specified in `.clinerules/overview.md`. Dependencies are listed in `requirements.txt`.
-   **Testing**: `pytest` is the framework used for running unit tests located in `test_gcn.py`.
-   **Code Formatting/Linting**: `ruff` is used for code formatting and linting, enforced via `.clinerules`.
-   **Configuration**: Model parameters (`num_node_features`, `num_classes`, `hidden_dim`) are managed using `pydantic-settings` via the `ModelSettings` class in `config.py`. General settings (checkpoint and log directories) are managed via the `Settings` class in `config.py`.
-   **Automation**: `.clinerules` file is configured to automatically run `pytest` and `ruff format` whenever Python files (`main.py`, `test_gcn.py`, `config.py`) are changed.
-   **Version Control**: `.git` directory indicates Git is used for version control. `.gitignore` specifies files/directories to exclude.
-   **Python Version Management**: `.python-version` file suggests `pyenv` might be used to manage the Python version.
-   **Virtual Environment**: `.venv` directory indicates a virtual environment is likely used (managed by `uv`).

## Key Dependencies (from `requirements.txt` likely includes):

-   `torch`
-   `torch-geometric`
-   `pytest`
-   `ruff`
-   `uv`
-   *(Other transitive dependencies installed by the above)*

## Technical Constraints

-   Must adhere to the practices defined in `.clinerules/overview.md` (using `uv`, running `ruff` and `pytest`).
-   Code should be compatible with the specified Python version.

## Tool Usage Patterns

-   Install dependencies: `uv pip install -r requirements.txt`
-   Run tests: `pytest`
-   Format code: `ruff format main.py test_gcn.py`
-   Run the main script (e.g., for a demo): `python main.py`
