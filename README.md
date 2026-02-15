# Machine Learning Pipeline for Pairs Trading

This repository contains the code for the paper *"Machine Learning Pipeline for Pairs Trading"*. It uses a pipeline developed to identify mean reverting ETF pairs, utilizes machine learning models to reliably backtest these pairs and evaluate performance, and analyzes whether mean reverting relationships in ETFs reliably propagate to their individual stock holdings. 

This repository is intended for transparency, and is not feasibly executable with a single command under a time constraint.

---
## Overview
- **Goal:** Identify whether a machine learning pipeline increases the proportion of tradable stock pairs identified through stock-level ETF decomposition.
- **Steps:**
  - Filtering U.S. Equity ETFs by time, volume and liquidity
  - Identifying 15 mean reverting ETF pairs using rolling cointegration, correlation filters and K-means clustering with PCA Analysis
  - Using **XGBoost**, **Random Forest**, **LSTM**, **KNN**, and **Logistic Regression** models to simulate pair performance and select the most suitable model for stock pair evaluation
  - Decomposing ETFs into their top 3 stock holdings (that have sufficient data in the yfinance library) and running them against each other to identify the proportion of tradable stock pairs

---
## Data
Only assets with appropriate historical data were retained for this project.
- **Source:** Yahoo Finance (yfinance library)
- **Assets**: U.S. Equity ETFs and corresponding stock holdings
- **Train/Test Split:**  
  - Train: January 1, 2015 to January 1, 2020
  - Test: January 2, 2020 to December 31, 2024

---
## Reproducibility:
- This code does not have one-click reproducibility.
- Full execution of this code requires significant computational cost, and rerunning the pipeline to reproduce results is not expected
- All figures and tables reported are in the code as is, and clicking on the individual notebooks to view results is sufficient
- This repository prioritizes code transparency rather than code runnability
- Please do not attempt to rerun pair_searching/etf_searching.ipynb/

---
## 🛠️ Requirements
- pandas, numpy, scikit-learn, yfinance, matplotlib, seaborn, statsmodels, pytorch, xgboost, pip, ipykernel
- Install with:
  
  ```bash
  pip install -r environment.yml
  ```

---
## Citations
Please cite the associated paper to reference this work.
