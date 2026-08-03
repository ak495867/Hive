"""
DISCLAIMER:
This software is for educational and research purposes only. It is not financial,
investment, or trading advice. Past simulated performance does not guarantee
future results. All trading involves risk – you may lose money. Use at your own
risk. The authors assume no liability for any losses or damages. By using this
code, you agree to these terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import math
import warnings

warnings.filterwarnings("ignore")


def get_ring_neighbours(num_neurons, k=5):
    half = k // 2
    neighbours = torch.tensor(
        [
            [(i + j) % num_neurons for j in range(-half, half + 1)]
            for i in range(num_neurons)
        ],
        dtype=torch.long,
    )
    return neighbours, half


class NeuronGroup(nn.Module):
    def __init__(self, num_neurons, d_x, d_state=32, tau=0.1, k_neighbours=5, d_k=16):
        super().__init__()
        self.num_neurons = num_neurons
        self.d_state = d_state
        self.tau = tau
        self.k = k_neighbours
        self.d_k = d_k
        self.W_in = nn.Linear(d_x, d_state, bias=False)
        self.W_rec = nn.Linear(d_state, d_state, bias=False)
        self.b = nn.Parameter(torch.zeros(d_state))
        self.C_left = nn.Linear(d_state, d_state, bias=False)
        self.C_right = nn.Linear(d_state, d_state, bias=False)
        self.u = nn.Parameter(torch.randn(2 * d_state))
        self.readout_weight = nn.Parameter(torch.randn(num_neurons, d_state) * 0.1)
        self.readout_bias = nn.Parameter(torch.zeros(num_neurons))
        self.W_q = nn.Linear(d_state, d_k, bias=False)
        self.W_k = nn.Linear(d_state, d_k, bias=False)
        neighbours, _ = get_ring_neighbours(num_neurons, k_neighbours)
        self.register_buffer("neighbour_idx", neighbours)

    def forward(self, x, h_prev):
        batch, N, d = h_prev.shape
        h_neighbours = h_prev[:, self.neighbour_idx, :]
        h_self_exp = h_prev.unsqueeze(2).expand(-1, -1, self.k, -1)
        cat_states = torch.cat([h_self_exp, h_neighbours], dim=-1)
        scores = torch.einsum("d,bnkd->bnk", self.u, cat_states)
        w_local = F.softmax(scores, dim=-1)

        transformed = self.C_left(h_neighbours) * self.C_right(h_neighbours)
        coupling = torch.einsum("bnk,bnkd->bnd", w_local, transformed)

        in_signal = self.W_in(x).unsqueeze(1).expand(-1, N, -1)
        rec_signal = self.W_rec(h_prev)
        z = in_signal + rec_signal + coupling + self.b
        h_new = (1 - self.tau) * h_prev + self.tau * torch.tanh(z)
        return h_new, w_local

    def compute_asset_score(self, h, w_local):
        p_raw = (h * self.readout_weight).sum(dim=2) + self.readout_bias
        p_neighbours = p_raw[:, self.neighbour_idx]
        p_local = (w_local * p_neighbours).sum(dim=2)
        c_g = h.mean(dim=(0, 1))
        q = self.W_q(c_g)
        k = self.W_k(h.mean(dim=0))
        attn_scores = (q @ k.T) / math.sqrt(self.d_k)
        alpha = F.softmax(attn_scores, dim=0)
        score = (alpha.unsqueeze(0) * p_local).sum(dim=1)
        return score


class HiveVolPortfolio(nn.Module):
    def __init__(self, n_assets, d_x, d_state=32, n_strategies=30, temperature=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.temperature = temperature
        self.group = NeuronGroup(n_strategies, d_x, d_state)

    def forward(self, x_seq):
        batch, n_assets, seq_len, d_x = x_seq.shape
        N = self.group.num_neurons
        d = self.group.d_state
        h_list = [
            torch.zeros(batch, N, d, device=x_seq.device) for _ in range(n_assets)
        ]
        w_local_list = [None] * n_assets

        for t in range(seq_len):

            for a in range(n_assets):
                x_t = x_seq[:, a, t, :]
                h_a = h_list[a]
                h_new, w_local = self.group(x_t, h_a)
                h_list[a] = h_new
                w_local_list[a] = w_local

        scores = []
        for a in range(n_assets):
            score_a = self.group.compute_asset_score(h_list[a], w_local_list[a])
            scores.append(score_a)
        scores = torch.stack(scores, dim=1)

        weights = F.softmax(scores / self.temperature, dim=1)
        return weights


TICKERS = ["^VIX", "UVXY", "SVXY", "VXX", "VIXY", "VIXM"]
LOOKBACK = 20
START = "2011-01-01"
END = "2024-01-01"


def compute_asset_features(close_series):
    close = close_series.astype(np.float32)
    log_ret = np.diff(np.log(close), prepend=np.nan)
    log_ret[0] = 0.0
    delta = np.diff(close, prepend=close[0])
    up = np.maximum(delta, 0)
    down = -np.minimum(delta, 0)
    avg_up = pd.Series(up).rolling(10).mean().fillna(0).values
    avg_down = pd.Series(down).rolling(10).mean().fillna(0).values
    rsi = 100 - 100 / (1 + avg_up / (avg_down + 1e-9))
    rsi = rsi / 100.0
    ma_20 = pd.Series(close).rolling(20).mean().fillna(method="bfill").values
    std_20 = pd.Series(close).rolling(20).std().fillna(0).values
    bb_upper = (close - (ma_20 + 2 * std_20)) / (std_20 + 1e-9)
    bb_lower = (close - (ma_20 - 2 * std_20)) / (std_20 + 1e-9)
    features = np.stack([log_ret, rsi, bb_upper, bb_lower], axis=1)
    return features


def build_multivol_dataset(start=START, end=END, lookback=LOOKBACK):
    all_close = pd.DataFrame()
    for t in TICKERS:
        ticker_data = yf.download(t, start=start, end=end, progress=False)
        all_close[t] = ticker_data["Close"]
    all_close = all_close.dropna()
    print(f"Common dates: {len(all_close)} days")

    sequences = []
    target_returns = []
    for i in range(lookback, len(all_close) - 1):
        seq = np.zeros((len(TICKERS), lookback, 4), dtype=np.float32)
        ret = np.zeros(len(TICKERS), dtype=np.float32)
        for a, t in enumerate(TICKERS):
            close_arr = all_close[t].values
            feat_arr = compute_asset_features(close_arr)
            seq[a] = feat_arr[i - lookback + 1 : i + 1]
            ret[a] = (close_arr[i + 1] - close_arr[i]) / (close_arr[i] + 1e-8)
        sequences.append(seq)
        target_returns.append(ret)

    normalized_seq = []
    for day_idx, seq in enumerate(sequences):
        norm_seq = np.zeros_like(seq)
        for a in range(len(TICKERS)):
            if day_idx == 0:
                past_data = seq[a].reshape(-1, 4)
            else:
                past_data = np.concatenate(
                    [s[a] for s in sequences[:day_idx]] + [seq[a]], axis=0
                )
            scaler = StandardScaler()
            scaler.fit(past_data)
            norm_seq[a] = scaler.transform(seq[a])
        normalized_seq.append(norm_seq)

    X_tensors = [torch.FloatTensor(s) for s in normalized_seq]
    y_tensors = [torch.FloatTensor(r) for r in target_returns]
    return X_tensors, y_tensors


if __name__ == "__main__":
    print("Downloading and preparing multi‑asset volatility data...")
    X_list, y_list = build_multivol_dataset()
    total_days = len(X_list)
    print(f"Total trading days: {total_days}")

    train_end = int(0.7 * total_days)
    val_end = int(0.85 * total_days)
    train_X = X_list[:train_end]
    train_y = y_list[:train_end]
    val_X = X_list[train_end:val_end]
    val_y = y_list[train_end:val_end]
    test_X = X_list[val_end:]
    test_y = y_list[val_end:]

    n_assets, lookback, d_x = X_list[0].shape
    print(f"Assets: {n_assets}, Lookback: {lookback}, Features: {d_x}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HiveVolPortfolio(
        n_assets=n_assets, d_x=d_x, d_state=32, n_strategies=30, temperature=0.1
    ).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val_sharpe = -np.inf

    for epoch in range(100):
        model.train()
        indices = np.random.permutation(len(train_X))
        batch_size = 64
        epoch_loss = 0.0
        for i in range(0, len(indices), batch_size):
            idx = indices[i : i + batch_size]
            batch_X = torch.stack([train_X[j] for j in idx]).to(device)
            batch_y = torch.stack([train_y[j] for j in idx]).to(device)

            weights = model(batch_X)
            port_returns = (weights * batch_y).sum(dim=1)

            mean_ret = port_returns.mean()
            std_ret = port_returns.std() + 1e-8
            sharpe = np.sqrt(252) * mean_ret / std_ret
            loss = -sharpe

            concentration = (weights**2).sum(dim=1).mean()
            loss = loss + 0.01 * concentration

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= len(indices)

        model.eval()
        with torch.no_grad():
            val_batch_X = torch.stack(val_X).to(device)
            val_batch_y = torch.stack(val_y).to(device)
            val_weights = model(val_batch_X)
            val_port_returns = (val_weights * val_batch_y).sum(dim=1)
            val_mean = val_port_returns.mean()
            val_std = val_port_returns.std() + 1e-8
            val_sharpe = np.sqrt(252) * val_mean / val_std
        print(
            f"Epoch {epoch+1:3d} | Train Loss: {epoch_loss:.4f} | Val Sharpe: {val_sharpe.item():.3f}"
        )
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            torch.save(model.state_dict(), "model/hive.pt")
            print("  -> Best model saved.")

    model.load_state_dict(torch.load("model/hive.pt"))
    model.eval()
    with torch.no_grad():
        test_batch_X = torch.stack(test_X).to(device)
        test_batch_y = torch.stack(test_y).to(device)
        test_weights = model(test_batch_X)
        test_port_returns = (test_weights * test_batch_y).sum(dim=1)
        test_mean = test_port_returns.mean()
        test_std = test_port_returns.std() + 1e-8
        test_sharpe = np.sqrt(252) * test_mean / test_std
        print(f"\nTest Sharpe: {test_sharpe.item():.3f}")
        cum_ret = torch.cumprod(1 + test_port_returns, 0) - 1
        max_dd = (cum_ret.cummax(0)[0] - cum_ret).max()
        print(f"Max Drawdown: {max_dd.item():.4f}")
