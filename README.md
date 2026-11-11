# Lab Exercise 6 – MLP Implementation

## Included files
- `Lab6_MLP_Implementation.ipynb` – complete notebook implementation with explanations, mathematical parameter calculations, evaluation, plots, activation comparison, and dropout experiment.
- `app.py` – interactive Streamlit dashboard with configurable architecture, activation, learning rate, epochs, batch size, dropout, early stopping, model summary, metrics, plots, and activation comparison.
- `requirements.txt` – Python dependencies.

## Dataset
The project uses the Breast Cancer Wisconsin Diagnostic dataset bundled with scikit-learn. It has 569 observations and 30 numerical input features. The target is binary:
- `0`: malignant
- `1`: benign

The dataset is loaded by the code; no separate CSV download is needed.

## Setup
Use Python 3.10 or 3.11 in a virtual environment.

### Windows (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Jupyter Notebook
```powershell
pip install -r requirements.txt
jupyter notebook
```
Open `Lab6_MLP_Implementation.ipynb` and choose **Run All**.

## Architecture and parameter calculation
Baseline architecture:
`30 -> Dense(16, ReLU) -> Dense(8, ReLU) -> Dense(1, Sigmoid)`

- Dense 1: `(30 × 16) + 16 = 496`
- Dense 2: `(16 × 8) + 8 = 136`
- Output: `(8 × 1) + 1 = 9`
- Total: `496 + 136 + 9 = 641` trainable parameters

Dropout and the input layer do not add trainable parameters.

## Self-learning work
1. Compare ReLU, tanh and sigmoid hidden-layer activations.
2. Compare a baseline model with a dropout-regularized model.
3. Use early stopping and inspect training/validation curves to discuss overfitting and underfitting.

## Submission checklist
- Run all cells in the notebook and save it with outputs.
- Run `streamlit run app.py` and test the dashboard.
- Take screenshots of the running app and key notebook outputs if your instructor expects demonstration evidence.
- Replace `XXX` in the project folder/file name with the last three digits of your register number. The lab sheet's example is `Aaron_101_Lab1`; follow the same naming pattern with your own name, register suffix, and lab number.
- Do not present the model as a medical diagnostic tool. It is strictly an academic ML demonstration.
