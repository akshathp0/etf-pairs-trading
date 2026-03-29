# Machine Learning Pipeline for ETF Pairs Trading and Stock-Level Decomposition

**Authors:** Akshath Pasam, Arnav Nair, Arush Shah

**Stack:** Python (NumPy, Pandas, Scikit-learn, XGBoost, PyTorch, Statsmodels, yfinance)

---

## Overview

This project investigates where machine learning adds value in pairs trading (pair selection or trade execution) by building a full pipeline from ETF screening through stock-level decomposition. A multi-stage pair-selection pipeline (rolling cointegration, correlation filtering, PCA + K-means clustering) identifies 15 cointegrated ETF pairs from 62 filtered U.S. equity ETFs. Five ML models are trained and compared during an ETF-level model development phase, after which the best-performing model (XGBoost) is frozen and applied to stock pairs derived from decomposing those ETFs into their top 3 constituent holdings.

The core finding: **the ML pair-selection pipeline does not significantly improve the proportion of tradable stock pairs over random ETF pairings.** Of 135 pipeline-selected stock pairs, 15.6% met performance thresholds (>10% return, >0.5 Sharpe), compared to 20.0% from 135 randomly paired stock pairs, which is a difference that is not statistically significant (z = −0.95, p = 0.34). The XGBoost trading model and the process of stock-level ETF decomposition, not the pair-selection pipeline, drove the non-trivial proportion of profitable outcomes across both groups.

---

## Methodology

### 1. ETF Universe Construction

Starting from 916 U.S. equity ETFs, filters were applied to produce a universe of 62 ETFs:

- **Time filter:** Must exist since January 1, 2015 (10-year window)
- **Volume filter:** Average daily volume > 500,000 shares
- **Price filter:** Average closing price > $10

### 2. Pair Selection Pipeline

Using training data only (Jan 2015 – Dec 2019):

1. **Rolling cointegration:** Augmented Engle-Granger test with 252-day windows and 21-day steps. Pairs retained if >40% of windows show significant cointegration (p < 0.10).
2. **Correlation filter:** Pairs with both 60-day and 252-day rolling correlations below 0.98 are retained, removing near-identical ETFs.
3. **PCA + K-means clustering:** PCA with 2 components fit to scaled returns. K-means with silhouette-optimized cluster count (k = 2). Only pairs within the same cluster are kept.

**Result:** 15 ETF pairs selected for model development.

### 3. Model Development (ETF-Level)

Five models were trained on spread-based features (spread lags, z-score lags, rolling statistics, volatility) with binary labels indicating fixed-horizon (10-day) trade profitability when a z-score threshold (|Z| ≥ 1.0) is crossed. Hyperparameters were tuned via grid search during this phase, then frozen.

| Model               | Total Return (%) | Sharpe Ratio | Max Drawdown (%) |
|----------------------|------------------|--------------|-------------------|
| **XGBoost**          | **3.08**         | **1.35**     | **−0.82**         |
| Random Forest        | 2.11             | 1.01         | −0.78             |
| KNN                  | 2.53             | 0.33         | −3.95             |
| LSTM                 | 0.96             | 0.32         | −2.26             |
| Logistic Regression  | −0.14            | −0.02        | −2.50             |

Model rankings remained stable under alternative position-sizing schemes, including hedge-ratio-based sizing and different long-short dollar splits.

### 4. Stock-Level ETF Decomposition

The frozen XGBoost model was applied to stock pairs formed by pairing the top 3 holdings of each ETF in the 15 pipeline-selected pairs (135 stock pairs) and 15 randomly selected ETF pairs (135 stock pairs). Performance was compared using a tradability threshold of >10% total return and >0.5 Sharpe ratio.

---

## Results

### Paired vs. Random Stock Pair Performance

| Dataset        | Tradable Pairs (%) | Median Return (%) | Median Sharpe | Median Trades |
|----------------|--------------------|--------------------|---------------|---------------|
| Paired SPs     | 15.6               | 22.06              | 0.85          | 31            |
| Random SPs     | 20.0               | 28.99              | 0.72          | 36            |
| Aggregate SPs  | 17.8               | 25.89              | 0.76          | 32.5          |

- The difference in proportions is **not statistically significant** (two-proportion z-test, p = 0.34, 95% CI: [−0.136, 0.047]).
- Median aggregate Sharpe (0.76) is comparable to SPY's Sharpe (0.75) over the same period.
- Transaction costs of 1–5 bps per round-trip reduce returns by at most ~2.5%, not materially affecting conclusions.

### Key Takeaways

- **Trade execution, not pair selection, is where ML adds value.** The XGBoost model drove profitability across both pipeline-selected and random stock pairs.
- **ETF-level cointegration does not reliably propagate to constituent stocks.** ETF cointegration likely reflects shared factor exposure and diversification dynamics rather than stock-level mean reversion.
- **ETF decomposition yields a non-trivial proportion of tradable stock pairs** (~18% aggregate), but the ML pipeline does not significantly improve that proportion over random selection.
- **Evaluating ML in pairs trading requires separating pair discovery from trade execution.** Conflating the two can produce misleading conclusions about ML effectiveness.

---

## Data

- **Source:** Yahoo Finance (yfinance library)
- **Assets:** U.S. equity ETFs and corresponding top stock holdings
- **Train/Test Split:**
  - Train: January 1, 2015 – January 1, 2020
  - Test: January 2, 2020 – December 31, 2024

Only assets with sufficient historical data were retained.

---

## Reproducibility

- **This code does not have one-click reproducibility.** Full execution requires significant computational cost, and rerunning the pipeline to reproduce results is not expected.
- All figures and tables reported are preserved in the notebooks as-is. Clicking into individual notebooks to view results is sufficient. Results may vary slightly due to floating-point non-determinism.
- This repository prioritizes **code transparency** rather than code runnability.
- **Do not attempt to rerun** `pair_searching/etf_searching.ipynb`.
- Results of hedge-ratio-based position sizing can be viewed per-model by cloning and changing run_simulation sizing_mode to "hedge" for each respective model in `pair_trading/models/`.

---

## Setup

**Requirements:** Python 3.10+, pandas, numpy, scikit-learn, yfinance, matplotlib, seaborn, statsmodels, pytorch, xgboost, ipykernel

```bash
git clone https://github.com/akshathp0/etf-pairs-trading.git
cd etf-pairs-trading
pip install -r environment.yml
```

---

## Limitations & Future Work

- **Transaction costs:** Slippage and other execution frictions beyond commission estimates are not modeled
- **Fixed holding period:** Dynamic exit upon mean reversion (rather than fixed 10-day horizon) could better capture convergence timing
- **ETF decomposition depth:** Using more than the top 3 holdings with individual weights may better represent ETF-level price movements
- **Model diversity:** Convolutional transformers or other architectures could be explored for trade execution
- **Pair selection alternatives:** Training ML models to directly rank or score pairs, bypassing statistical filters, may better isolate ML's pair-selection value
- **Sentiment features:** Market sentiment data (e.g., VIX) could improve signal quality

---

## Citations

Please cite the associated paper to reference this work.

---

## Contact
 
**Akshath Pasam**
[akshath@pasam.com] · [GitHub](https://github.com/akshathp0)
