# Neural Movie Recommendation System

Python project scaffold for exploring MovieLens 1M and later building a
recommendation system with PyTorch. No data processing, training, evaluation,
or ML models have been implemented yet.

## Project structure

```text
.
|-- data/
|   |-- raw/
|   |   `-- .gitkeep
|   `-- processed/
|       `-- .gitkeep
|-- notebooks/
|   `-- .gitkeep
|-- src/
|   `-- neural_recommendation/
|       `-- __init__.py
|-- results/
|   |-- figures/
|   |   `-- .gitkeep
|   |-- metrics/
|   |   `-- .gitkeep
|   `-- checkpoints/
|       `-- .gitkeep
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Folders and files

- `data/raw/`: original, unmodified MovieLens 1M files. Place the extracted
  dataset here, for example `data/raw/ml-1m/ratings.dat`, `movies.dat`, and
  `users.dat`, alongside the dataset's README.
- `data/processed/`: future cleaned data, encoded IDs, and train/validation/test
  splits. Keep derived data separate from the original files.
- `notebooks/`: future notebooks for data exploration and experiment analysis.
  Use descriptive names such as `01_data_exploration.ipynb`.
- `src/neural_recommendation/`: reusable Python code will live here as the
  project grows. Add preprocessing, model, training, and evaluation modules
  when needed.
- `src/neural_recommendation/__init__.py`: marks the directory as a Python
  package; currently contains only a package description.
- `results/figures/`: generated plots and visualizations.
- `results/metrics/`: exported evaluation scores and experiment summaries.
- `results/checkpoints/`: saved model weights and training checkpoints, once
  training is implemented.
- `.gitkeep`: empty placeholder files so Git can track otherwise empty folders.
  They have no effect on Python.
- `requirements.txt`: dependencies for data handling (`numpy`, `pandas`),
  neural networks (`torch`), future splitting and evaluation (`scikit-learn`),
  notebooks (`jupyterlab`, `ipykernel`), plotting (`matplotlib`, `seaborn`), and
  progress bars (`tqdm`). Versions are unpinned at this scaffold stage; capture
  exact versions after validating your development environment.
- `.gitignore`: excludes local environments, secrets, caches, notebook
  checkpoints, datasets, and generated results while preserving placeholders.
- `README.md`: project overview, directory guide, and setup instructions.

## Setup (Windows PowerShell)

Run these commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Using the virtual environment's Python directly avoids needing to activate it.
For a GPU-specific PyTorch installation, use the installation command appropriate
to your hardware from the official PyTorch installation guide.

Download MovieLens 1M separately and extract it under `data/raw/`. The dataset
is not bundled with this scaffold. Run scripts and notebooks with paths rooted
at this project directory; reusable source modules will be added later.
