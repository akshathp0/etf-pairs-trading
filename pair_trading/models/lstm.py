import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ClassificationLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, input_size):
        super(ClassificationLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.linear(out)
        return out.squeeze()

def zscore_calc_test(series, mean, std):
    return (series - mean) / std

def zscore_calc_train(series):
    return (series - series.mean()) / series.std(), series.mean(), series.std()

def rolling_zscore(series, window = 60):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def predict_mean_reversion_label(
    spread,
    z_window = 60,
    horizon = 10,
    entry_z = 1.0,
    min_pnl = 0.0,
):

    z = rolling_zscore(spread, window=z_window)
    direction = -np.sign(z)
    future_spread = spread.shift(-horizon)
    spread_change = future_spread - spread
    pnl = direction * spread_change

    cond_entry = z.abs() >= entry_z
    labels = ((cond_entry) & (pnl > min_pnl)).astype(float)
    labels = labels.iloc[:-horizon]
    labels = labels.dropna()
    return labels

def create_features(spread, window):
    df = pd.DataFrame({'spread': spread})

    df['z-score'] = rolling_zscore(df['spread'])

    df['z-score_lag1'] = df['z-score'].shift(1)
    df['spread_lag1'] = df['spread'].shift(1)
    df['rolling_mean'] = df['spread'].rolling(window).mean()
    df['volatility'] = df['spread'].pct_change().rolling(window).std()
    df['z-score_lag2'] = df['z-score'].shift(2)
    df['z-score_lag5'] = df['z-score'].shift(5)

    df = df.dropna()
    return df
    
def find_beta(asset1, asset2):
    X = sm.add_constant(asset2)
    model = sm.OLS(asset1, X).fit()
    beta = model.params.iloc[1]
    return beta

def build_dicts(etf_pairs, training_data, testing_data):
    training_spreads = {}
    testing_spreads = {}

    X_train_dict = {}
    y_train_dict = {}
    X_test_dict = {}
    y_test_dict = {}

    pair_betas = {}

    window = 5

    for etf1, etf2 in etf_pairs:

        pair_name = f'{etf1}_{etf2}'

        train_z1, mean_z1, std_z1 = zscore_calc_train(training_data[etf1])
        train_z2, mean_z2, std_z2 = zscore_calc_train(training_data[etf2])

        pair_betas[pair_name] = find_beta(training_data[etf1], training_data[etf2])

        test_z1 = zscore_calc_test(testing_data[etf1], mean_z1, std_z1)
        test_z2 = zscore_calc_test(testing_data[etf2], mean_z2, std_z2)

        training_spread, testing_spread = train_z1 - train_z2, test_z1 - test_z2
        training_spreads[pair_name], testing_spreads[pair_name] = training_spread, testing_spread

        train_features = create_features(training_spread, window = window)
        test_features  = create_features(testing_spread, window = window)

        train_labels = predict_mean_reversion_label(
            training_spread,
            z_window = 60,
            horizon = 10,
            entry_z = 1.0,
            min_pnl = 0.0,
        )

        test_labels = predict_mean_reversion_label(
            testing_spread,
            z_window = 60,
            horizon = 10,
            entry_z = 1.0,
            min_pnl = 0.0,
        )

        train_idx = train_features.index.intersection(train_labels.index)
        test_idx  = test_features.index.intersection(test_labels.index)

        train_features = train_features.loc[train_idx].copy()
        test_features  = test_features.loc[test_idx].copy()

        train_labels = train_labels.loc[train_idx]
        test_labels  = test_labels.loc[test_idx]

        train_features['label'] = train_labels
        test_features['label']  = test_labels

        train_features = train_features.dropna(subset=['label'])
        test_features  = test_features.dropna(subset=['label'])

        X_train_dict[pair_name] = train_features.drop(columns=['label'])
        y_train_dict[pair_name] = train_features['label']
        X_test_dict[pair_name]  = test_features.drop(columns=['label'])
        y_test_dict[pair_name]  = test_features['label']

    for pair_name in y_train_dict:
        print(pair_name, 
            "train_label_mean:", y_train_dict[pair_name].mean(), 
            "test_label_mean:",  y_test_dict[pair_name].mean())
        
    training_spreads_df = pd.DataFrame(training_spreads)

    return training_spreads_df, X_train_dict, y_train_dict, X_test_dict, y_test_dict, pair_betas

def create_sequences(X, y, window_size):
    X_seq, y_seq  = [], []

    for i in range(len(X) - window_size):
        X_seq.append(X.iloc[i : i + window_size].values)
        y_seq.append(y.iloc[i + window_size])

    return np.array(X_seq), np.array(y_seq)

def train_model(model, train_loader, criterion, optimizer, epochs = 30, device = 'cpu'):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")

def evaluate_model(model, pair_name, test_loader, device = 'cpu'):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu())
            all_true.append(yb.cpu())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_true).numpy()
    y_pred_label = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_label)
    print("Test Accuracy: ", accuracy)
    return y_true, y_pred, y_pred_label

def run_pair(pair_name, X_train_dict, y_train_dict, X_test_dict, y_test_dict, epochs, device = 'cpu'):

    hidden_size = 32
    lr = 0.01
    num_layers = 1
    window_size = 5
    batch_size = 32

    X_df = X_train_dict[pair_name]
    y_series = y_train_dict[pair_name]

    X_seq, y_seq = create_sequences(X_df, y_series, window_size = window_size)

    X_train = torch.tensor(X_seq, dtype = torch.float32)
    y_train = torch.tensor(y_seq, dtype = torch.float32).unsqueeze(-1)

    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)

    X_test_df = X_test_dict[pair_name]
    y_test_series = y_test_dict[pair_name]

    X_test_seq, y_test_seq = create_sequences(X_test_df, y_test_series, window_size = window_size)
    
    X_test = torch.tensor(X_test_seq, dtype = torch.float32)
    y_test = torch.tensor(y_test_seq, dtype = torch.float32).unsqueeze(-1)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    model = ClassificationLSTM(input_size = 8, hidden_size = hidden_size, num_layers = num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    print(f"\nTraining model for pair: {pair_name}")
    train_model(model, train_loader, criterion, optimizer, epochs = epochs, device = device)
    
    print(f"\nEvaluating model for pair: {pair_name}")
    y_true, y_pred, y_pred_label = evaluate_model(model, pair_name, test_loader, device = device)

    plot_labels(X_test_df, y_pred_label, pair_name)

    print_confusion_matrix(y_true, y_pred_label)

    print_f1(y_true, y_pred_label)
    
    return model, y_true, y_pred

def aggregate_results(training_spreads_df, X_train_dict, y_train_dict, X_test_dict, y_test_dict):
    results = {}

    for pair_name in training_spreads_df.columns:
        model, y_true, y_pred = run_pair(
            pair_name,
            X_train_dict,
            y_train_dict,
            X_test_dict,
            y_test_dict,
            epochs = 20,
            device = 'cpu'
        )
        results[pair_name] = {'model': model, 'y_true': y_true, 'y_pred': y_pred}

    return results

def build_prices(pair_list, testing_data, X_test_dict):
    test_prices_dict = {}

    for pair in pair_list:
        etf1, etf2 = pair.split('_')

        price_df = testing_data[[etf1, etf2]].copy()
        price_df.columns = ['price_long', 'price_short']

        test_index = X_test_dict[pair].index
        aligned_prices = price_df.loc[test_index]

        test_prices_dict[pair] = aligned_prices

    return test_prices_dict

def simulate_portfolio(
    X_df, probs, y_pred_label, prices_df, pair_betas, pair_name,
    initial_cash_per_pair = 100.0,
    holding_period = 10,
    entry_threshold = 0.8,
    sizing_mode = "dollar"
):
    df = X_df.copy()
    df = df.iloc[-len(y_pred_label):].copy()

    df['prob'] = probs
    df['label'] = y_pred_label
    df = df.join(prices_df)

    df['beta'] = float(pair_betas[pair_name])

    prob_threshold = np.quantile(probs, 0.85)

    cash = initial_cash_per_pair
    equity = pd.Series(index = df.index, dtype = float)
    equity.iloc[0] = cash
    trades = []

    i = 0
    n = len(df)

    while i < n - holding_period:
        row = df.iloc[i]
        prob = row['prob']
        zscore = row['z-score']
        price_long_in = row['price_long']
        price_short_in = row['price_short']

        if (prob >= prob_threshold and abs(zscore) >= entry_threshold):
            entry_idx = i
            exit_idx = i + holding_period
            entry_date = df.index[entry_idx]
            exit_date = df.index[exit_idx]

            price_long_out = df.iloc[exit_idx]['price_long']
            price_short_out = df.iloc[exit_idx]['price_short']

            notional = cash

            if sizing_mode == "dollar":
                long_dollars, short_dollars = notional * 0.5, notional * 0.5

                if zscore > 0:
                    shares_short_long = short_dollars / price_long_in
                    shares_long_short = long_dollars / price_short_in

                    pnl = (shares_long_short * (price_short_out - price_short_in)) + \
                        (shares_short_long * (price_long_in - price_long_out))
                    direction = "Short ETF1 / Long ETF2"
                else:
                    shares_long_long = long_dollars / price_long_in
                    shares_short_short = short_dollars / price_short_in

                    pnl = (shares_long_long * (price_long_out - price_long_in)) + \
                        (shares_short_short * (price_short_in - price_short_out))
                    direction = "Long ETF1 / Short ETF2"
            elif sizing_mode == "hedge":
                beta = abs(float(df["beta"].iloc[0]))
                q = notional / (price_long_in + beta * price_short_in)

                if zscore > 0:
                    shares_short_long = q
                    shares_long_short = beta * q

                    pnl = (shares_long_short * (price_short_out - price_short_in)) + \
                          (shares_short_long * (price_long_in - price_long_out))
                    direction = "Short long_leg / Long beta*short_leg"
                else:
                    shares_long_long   = q
                    shares_short_short = beta * q

                    pnl = (shares_long_long * (price_long_out - price_long_in)) + \
                          (shares_short_short * (price_short_in - price_short_out))
                    direction = "Long long_leg / Short beta*short_leg"

            cash += pnl
            equity.loc[exit_date] = cash

            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "prob": prob,
                "direction": direction,
                "pnl": pnl,
                "notional": notional
            })

            i = exit_idx + 1
        else:
            equity.iloc[i] = cash
            i += 1

    equity.ffill(inplace = True)
    return equity, trades

def run_simulation(results, X_test_dict, testing_data, pair_betas):
    pair_list = list(X_test_dict.keys())
    test_prices_dict = build_prices(pair_list, testing_data, X_test_dict)

    total_cash = 0
    combined_equity = pd.Series(dtype=float)
    pair_trade_counts = {}
    pair_pnls = {}
    pair_sharpes = {}
    pair_drawdowns = {}
    pair_equities = {}

    sns.set_theme(style = "whitegrid", context = "notebook")

    plt.figure(figsize = (12, 6))
    plt.xlabel("Date", fontsize = 12)
    plt.ylabel("Portfolio Value ($)", fontsize = 12)
    plt.grid(True)
    plt.tight_layout()

    for pair_name in pair_list:
        model_info = results[pair_name]
        model = model_info['model']
        y_true = model_info['y_true']
        y_pred = model_info['y_pred']

        probs = torch.sigmoid(torch.tensor(y_pred)).squeeze().numpy()
        y_pred_label = (probs > 0.5).astype(int)

        prices_df = test_prices_dict[pair_name]
        X_test_df = X_test_dict[pair_name]

        equity, trades = simulate_portfolio(X_test_df, probs, y_pred_label, prices_df, pair_betas, pair_name, sizing_mode = "dollar")

        pair_equities[pair_name] = equity
        max_dd, _ = compute_max_drawdown(equity)
        pair_drawdowns[pair_name] = max_dd

        pair_returns = equity.pct_change().dropna()

        if pair_returns.std() != 0:
            sharpe = (pair_returns.mean() / pair_returns.std()) * np.sqrt(252)
        else:
            sharpe = np.nan

        pair_sharpes[pair_name] = sharpe

        total_cash += equity.iloc[-1]
        pair_trade_counts[pair_name] = len(trades)
        pair_pnls[pair_name] = [t['pnl'] for t in trades]

        if combined_equity.empty:
            equity.name = pair_name
            combined_equity = equity.to_frame()
        else:
            equity.name = pair_name
            combined_equity = pd.concat([combined_equity, equity], axis=1)

        sns.lineplot(x = equity.index, y = equity.values, label = pair_name, linewidth = 2, alpha = 0.9)

    plt.legend(title = "Pair", frameon = True, loc = "center left", bbox_to_anchor = (0, 0.64))
    plt.savefig("/Users/ak/Downloads/LSTM_equity.pdf", dpi = 400, bbox_inches = "tight")
    plt.show()

    combined_equity = combined_equity.fillna(method='ffill').fillna(method='bfill')
    combined_equity['Total'] = combined_equity.sum(axis=1)

    print(f"\nFinal Portfolio Value: ${total_cash:,.2f}")
    print("\nTrade Summary:")
    for pair, count in pair_trade_counts.items():
        total_pnl = sum(pair_pnls[pair])
        avg_pnl = np.mean(pair_pnls[pair]) if pair_pnls[pair] else 0
        print(f"  {pair}: {count} trades | Total PnL: ${total_pnl:.2f} | Avg PnL: ${avg_pnl:.2f}")

    print("\nSharpe Ratios by Pair:")
    for pair, sharpe in pair_sharpes.items():
        print(f"{pair}: Sharpe Ratio = {sharpe:.2f}")

    combined_returns = combined_equity['Total'].pct_change().dropna()
    sharpe_ratio = (combined_returns.mean() / combined_returns.std()) * np.sqrt(252)
    print(f"\nTotal Sharpe Ratio: {sharpe_ratio:.2f}")

    print("\nMax Drawdowns by Pair:")
    for pair, drawdown in pair_drawdowns.items():
        print(f"{pair}: Max Drawdown = {drawdown:.2%}")

    max_dd, drawdowns = compute_max_drawdown(combined_equity['Total'])
    print(f"\nTotal Max Drawdown: {max_dd:.2%}")

    plot_combined_equity_and_drawdown(combined_equity['Total'])

    cumulative_equity = combined_equity['Total']
    cumulative_equity.to_csv("equity_curves/lstm_equity.csv", header = ["Portfolio Value"])

def compute_max_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1.0
    max_drawdown = drawdowns.min()
    
    return max_drawdown, drawdowns

def plot_combined_equity_and_drawdown(equity):
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(equity, label='Combined Equity')
    ax[0].set_title("Combined Equity Curve")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(drawdown, color='red', label='Drawdown')
    ax[1].set_title("Drawdown Over Time")
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    plt.show()

def plot_labels(X_df, y_pred_label, pair_name):
    spread = X_df['spread']
    
    aligned_spread = spread[-len(y_pred_label):]
    predicted_points = aligned_spread[y_pred_label == 1]

    plt.figure(figsize=(12, 5))
    sns.lineplot(x = aligned_spread.index, y = aligned_spread.values, label = 'Spread', color = 'black')
    sns.scatterplot(x = predicted_points.index, y = predicted_points.values, color = 'green', label = 'Predicted Reversion', s = 100, marker = 'o')
    plt.title(f"Predicted Mean Reversions of {pair_name}")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_confusion_matrix(y_true, y_pred_label):
    cm = confusion_matrix(y_true, y_pred_label, labels = [0, 1])
    print("Confusion Matrix:")
    print(cm)

def print_f1(y_true, y_pred_label):
    f1 = f1_score(y_true, y_pred_label)
    print(f"F1 Score: {f1:.4f}")