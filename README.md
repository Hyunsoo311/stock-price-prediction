# 📈 Stock Price Prediction with Machine Learning

This repository contains an exploratory and experimental project that uses historical stock market data (downloaded from Yahoo Finance) to build and compare machine learning models for predicting stock prices. The project includes data collection, EDA (exploratory data analysis), feature engineering (technical indicators such as moving averages and RSI), training and evaluation of several regression models, and experiments with sequential models (LSTM).

This README gives a clear overview of the repository, how to reproduce the results, what each file/folder contains, and guidance for customizing or extending the work.

---

## Table of contents
- Project overview
- Repository structure
- Key files and outputs
- Requirements
- Quick start
- How the notebook works (high-level)
- Models & evaluation
- Reproducing results
- Customization & extension
- Notes, limitations & next steps
- Contact

---

## Project overview
Goal: predict future closing prices of individual stocks using historical OHLCV (open-high-low-close-volume) data together with derived technical indicators.

Approach:
- Download historical data using `yfinance`.
- Perform EDA and visualize distributions, correlations, and technical indicators.
- Create technical indicator features (e.g., moving averages, RSI).
- Train and compare multiple regression models (including linear models and tree-based models).
- Experiment with sequential models (LSTM) to capture temporal dependencies.
- Evaluate models using standard regression metrics and visualize predictions vs actuals.

Repository contains EDA outputs, model comparison graphs, and the main notebook `prediction.ipynb` that documents the workflow and results.

---

## Repository structure (top-level)
- prediction.ipynb — Main Jupyter notebook with data collection, preprocessing, EDA, model training, evaluation and visualization.
- EDA_table_output.txt — Summary table output from EDA (descriptive statistics / feature summary).
- EDA_Graph_Output.png — Combined EDA visual output (plots summarizing features).
- APPL.png, APPL_RSI.png, APPL_FEATURE CORRELATION.png, APPL_CLOSE DISTRIBUTION.png — Example EDA and indicator visualizations for AAPL (Apple Inc.).
- APPL-Best Model-Linear Regerssion.png — Example visualization showing the best model found for AAPL (linear regression in this experiment).
- RSME comparison across regression models.png — RMSE comparison chart across models (filename contains typo; metric is RMSE).
- baseline_output_graph.png — Baseline prediction visualization for comparison.
- LSTM results/ — Folder for LSTM model outputs (may contain weights, logs, or plots).
- .gitignore — Git ignore rules.
- (Note: a virtual environment `.venv/` is present in the repository tree—committed by mistake. It is recommended to remove `.venv/` from the repository and add it to `.gitignore` to keep repo size small.)

---

## Key files & outputs
- `prediction.ipynb` — The single main notebook that walks through: data download (yfinance), feature engineering (MA, RSI, etc), EDA, model training, evaluation, and plotting results.
- EDA images and model-comparison plots — visual evidence of the analyses and model results. These are saved PNGs that make it easy to review outcomes without rerunning the whole notebook.
- `EDA_table_output.txt` — tabular EDA results (summary statistics).

---

## Requirements

Minimum packages (tested):
- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- yfinance
- ta (technical indicators)
- jupyter / jupyterlab

Optional (if you plan to run LSTM experiments):
- tensorflow (or tensorflow-cpu)
- keras

Install with pip (example):
```bash
pip install numpy pandas matplotlib seaborn scikit-learn yfinance ta jupyter
# Optional for LSTM:
pip install tensorflow
```

If you want to create a fresh virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt  # if a requirements file is added
```

---

## Quick start — run the notebook
1. Clone the repo:
   ```bash
   git clone https://github.com/Hyunsoo311/stock-price-prediction.git
   cd stock-price-prediction
   ```
2. Install dependencies (see Requirements).
3. Start Jupyter:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```
4. Open `prediction.ipynb` and run cells from top to bottom. The notebook is interactive and includes sections for changing the ticker symbol, date range, and model hyperparameters.

Tip: If you run into long download times for data, reduce the date range or cache the downloaded CSV locally.

---

## How the notebook works (high-level)
- Data collection: `yfinance` downloads historical OHLCV for a specified ticker and date range.
- Preprocessing:
  - Handle missing values (drop or forward-fill).
  - Create lagged features and rolling statistics (e.g., MA windows).
  - Compute technical indicators (RSI, moving averages, etc.) using the `ta` package or custom functions.
  - Split data into training and test sets (time-series split; training set earlier in time).
  - Scale features as required by models (e.g., StandardScaler for regression; min-max for NN).
- Modeling:
  - Baseline model (for reference).
  - Regression models (Linear Regression, and additional models compared in RMSE chart).
  - Sequential model(s) (LSTM) for temporal modeling (placed in `LSTM results/`).
- Evaluation:
  - Metrics: RMSE, MAE (and optionally R²).
  - Visual comparison: predicted vs actual closing prices, residuals, and error histograms.
- Output:
  - Figures are saved to repository as PNGs and can be reviewed in the repository root.

---

## Models & evaluation
- The repository contains comparison plots showing RMSE across models (see `RSME comparison across regression models.png`). For the AAPL experiments included, the notebook/visuals show Linear Regression as performing best among the set of regression models used (see `APPL-Best Model-Linear Regerssion.png`).
- Evaluation metrics used: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE). Plots visualize performance and assist with error analysis.

---

## Reproducing results
To reproduce the experiments:
1. Open `prediction.ipynb`.
2. Set the ticker symbol (e.g., `AAPL`) and date range at the top of the notebook.
3. Run all cells. Ensure you have the required packages installed.
4. The notebook will produce and save EDA plots and model comparison graphs in the repository root (or in a results folder if configured).

If you plan to reproduce LSTM experiments:
- Ensure `tensorflow` is installed and you have sufficient memory/compute for training.
- You can reduce sequence length or number of epochs to speed up tests.

---

## Customization & extension
- Change the ticker: replace the ticker variable in the notebook to any Yahoo Finance ticker (e.g., `MSFT`, `TSLA`, `GOOGL`).
- Add more technical indicators: use the `ta` library or create custom features (MACD, Bollinger Bands, ATR, etc.).
- Try more models: add Random Forest, XGBoost/LightGBM, or more complex deep-learning architectures.
- Cross-validation: implement time-series aware cross-validation (e.g., expanding window CV).
- Save & load models: integrate joblib or model.save for re-using trained models without retraining.
- Clean up repo: remove committed `.venv/` folder and add `requirements.txt` for easier reproducibility.

---

## Notes, limitations & suggestions
- The repository currently includes a committed `.venv/` directory. This is not recommended. Remove the virtual environment from the repository and add it to `.gitignore` to reduce repo size.
- `prediction.ipynb` is the single source of truth for the workflow. Consider breaking the notebook into modular scripts (data, features, training, evaluation) for production use.
- Results are experimental. Financial predictions are inherently noisy and past performance is not indicative of future returns. Use these experiments for educational and research purposes only, not for live trading without rigorous validation and risk controls.

---

## Next steps (suggested)
- Add a `requirements.txt` or `environment.yml` for reproducibility.
- Remove `.venv/` from the repository and re-commit.
- Split the notebook into scripts or package the pipeline for cleaner experimentation and CI.
- Add automated tests for data processing and feature generation.
- Add a short summary of the best-performing models and their hyperparameters in the repo.

---

## Contact
If you want me to:
- Commit this updated README.md directly to the repository, I can prepare a PR (please confirm).
- Extract and examine `prediction.ipynb` cell-by-cell and produce a precise step-by-step reproduction guide or convert the notebook to scripts, I can do that next.
- Clean the repo (remove `.venv/`) and add a `requirements.txt`, I can prepare changes.

Feel free to tell me which next action you'd like me to take.
