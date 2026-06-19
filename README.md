# Machine Learning Project

This repository contains machine learning and reinforcement learning portfolio work, including Q-learning simulations, Responsible AI analysis, MLflow tracking examples, and supporting project documents.

## Features

- Dynamic grid-world Q-learning simulation with temporary obstacles.
- Traffic signal control Q-learning example using traffic volume data.
- Responsible AI dashboard workflow for model explainability and error analysis.
- Integrated model training workflow using scikit-learn and MLflow.
- Technical paper and experimentation documents.

## Tech Stack

- Python
- Jupyter Notebook
- pandas and NumPy
- scikit-learn
- MLflow
- Responsible AI Toolbox and RAI Widgets
- tabulate

## Installation

Create a Python environment and install the main dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn mlflow responsibleai raiwidgets tabulate jupyter
```

Some scripts require local CSV datasets. Those datasets are intentionally ignored and are not included in the repository.

## How To Run

Run the dynamic grid-world Q-learning example:

```powershell
python "Q-learning(2) (1).py"
```

Run the traffic signal Q-learning example after placing the required traffic CSV dataset in the expected location:

```powershell
python "RL Traffic Signal Control Model - Copy (1).py"
```

Run the Responsible AI scripts after adding the expected training and test CSV files:

```powershell
python "RAI_error_analysis (1).py"
python "combinedModel (1).py"
```

Open the notebook:

```powershell
jupyter notebook "MCRAI&MLFLOW.ipynb"
```

## Folder Structure

```text
.
├── combinedModel (1).py
├── Q-learning(2) (1).py
├── RAI_error_analysis (1).py
├── RL Traffic Signal Control Model - Copy (1).py
├── MCRAI&MLFLOW.ipynb
├── *.docx
├── .gitignore
└── README.md
```

## Notes And Limitations

- Datasets, archives, model binaries, generated caches, and local environment files are excluded from Git.
- Some scripts contain machine-specific file paths and may need path updates before running on another computer.
- Large archives and duplicate placeholder files are kept locally but are not committed.
