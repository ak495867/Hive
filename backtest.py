import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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
        k = self.group.k
        h = torch.zeros(batch, n_assets, N, d, device=x_seq.device)
        for t in range(seq_len):
            x_t = x_seq[:, :, t, :]
            h_merged = h.view(batch * n_assets, N, d)
            x_merged = x_t.reshape(batch * n_assets, d_x)
            h_new_merged, w_local_merged = self.group(x_merged, h_merged)
            h = h_new_merged.view(batch, n_assets, N, d)
        w_local = w_local_merged.view(batch, n_assets, N, k)
        scores = self._batch_compute_asset_scores(h, w_local)
        weights = F.softmax(scores / self.temperature, dim=1)
        return weights

    def _batch_compute_asset_scores(self, h, w_local):
        group = self.group
        p_raw = (h * group.readout_weight).sum(dim=-1) + group.readout_bias
        neighbour_idx = group.neighbour_idx
        p_neighbours = p_raw[:, :, neighbour_idx]
        p_local = (w_local * p_neighbours).sum(dim=-1)
        c_g = h.mean(dim=(0, 2))
        q = group.W_q(c_g)
        h_mean = h.mean(dim=0)
        k = group.W_k(h_mean)
        attn_scores = torch.einsum("ad,anD->an", q, k) / math.sqrt(group.d_k)
        alpha = F.softmax(attn_scores, dim=1)
        score = (alpha.unsqueeze(0) * p_local).sum(dim=-1)
        return score


TICKERS = [
    "BTC-USD",
    "BKLN",
    "ACWV",
    "NET",
    "DHR",
    "MPWR",
    "AMZN",
    "ITOT",
    "DPZ",
    "QAI",
    "DGRO",
    "GSK",
    "BSV",
    "VFMO",
    "VCSH",
    "ARES",
    "TSLA",
    "SPHD",
    "ARKW",
    "KO",
    "ARB",
    "GLDM",
    "BAB",
    "MA",
    "BND",
    "VMAX",
    "MINV",
    "IEF",
    "XAR",
    "BIV",
    "IRM",
    "VIG",
    "XSW",
    "RWL",
    "MSFT",
    "XHS",
    "PRF",
    "NOBL",
    "HD",
    "DLR",
    "FLOT",
    "GLD",
    "VQT",
    "XMLV",
    "FTLS",
    "V",
    "XLC",
    "WMT",
    "GOOGL",
    "MXN",
    "MUB",
    "LGLV",
    "APO",
    "WBTC-USD",
    "RSP",
    "TIP",
    "FNDX",
    "ARKQ",
    "ABT",
    "BNB-USD",
    "SPY",
    "SPLV",
    "AIQ",
    "VOO",
    "SCHD",
    "LQD",
    "EFAV",
    "IEI",
    "DDOG",
    "VCIT",
    "NTES",
    "CSCO",
    "KKR",
    "PLTR",
    "CRWD",
    "WELL",
    "RDVY",
    "AGG",
    "BLK",
    "BDGS",
    "ABBV",
    "SHY",
    "NVDA",
    "PFM",
    "IAU",
    "HDV",
    "GBTC",
    "JNJ",
    "USMV",
    "SCHX",
    "NFLX",
    "META",
    "UNH",
    "AVGO",
    "XOM",
    "VGIT",
    "VTIP",
    "XT",
]

LOOKBACK = 20
START = "2017-01-01"
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
    print("Downloading data...")
    raw_data = {}
    failed_tickers = []
    for t in TICKERS:
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if df.empty:
                failed_tickers.append(t)
            else:
                raw_data[t] = df["Close"]
        except Exception as e:
            print(f"Failed to download {t}: {e}")
            failed_tickers.append(t)
    if failed_tickers:
        print(
            f"Warning: {len(failed_tickers)} ticker(s) had no data and will be filled with constant 1.0: {failed_tickers}"
        )
    if not raw_data and failed_tickers:
        raise ValueError("No data for any ticker. Check internet or tickers.")
    bdays = pd.bdate_range(start=start, end=end)
    all_close = pd.DataFrame(index=bdays, columns=TICKERS)
    for t, series in raw_data.items():
        all_close[t] = series.reindex(bdays)
    for t in failed_tickers:
        all_close[t] = 1.0
    all_close = all_close.ffill().bfill()
    all_close = all_close.dropna(how="all")
    print(
        f"Data ready: {all_close.shape[1]} tickers, {all_close.shape[0]} business days."
    )
    T = len(all_close)
    n_assets = len(TICKERS)
    features = np.zeros((T, n_assets, 4), dtype=np.float32)
    for a, t in enumerate(TICKERS):
        close_arr = all_close[t].values.astype(np.float32)
        feat_arr = compute_asset_features(close_arr)
        features[:, a, :] = feat_arr
    num_days = T - lookback - 1
    sequences = np.zeros((num_days, n_assets, lookback, 4), dtype=np.float32)
    target_returns = np.zeros((num_days, n_assets), dtype=np.float32)
    for i in range(lookback, T - 1):
        day_idx = i - lookback
        sequences[day_idx] = features[i - lookback + 1 : i + 1].transpose(1, 0, 2)
        target_returns[day_idx] = (
            all_close.iloc[i + 1].values - all_close.iloc[i].values
        ) / (all_close.iloc[i].values + 1e-8)
    flat_features = features.reshape(T, -1)
    cumsum = np.cumsum(flat_features, axis=0)
    cumsum_sq = np.cumsum(flat_features**2, axis=0)
    normalized_seq = []
    for i in range(lookback, T - 1):
        n = i + 1
        mean = cumsum[i] / n
        var = cumsum_sq[i] / n - mean**2
        std = np.sqrt(var + 1e-8)
        window_flat = sequences[i - lookback].reshape(lookback, -1)
        norm_flat = (window_flat - mean) / std
        normalized_seq.append(norm_flat.reshape(n_assets, lookback, 4))
    print(f"Total trading days: {num_days}")
    X_tensors = [torch.FloatTensor(s) for s in normalized_seq]
    y_tensors = [torch.FloatTensor(r) for r in target_returns]
    return X_tensors, y_tensors


def apply_risk_budget(weights, max_per_asset=0.3):
    clamped = torch.clamp(weights, max=max_per_asset)
    clamped = clamped / clamped.sum(dim=1, keepdim=True)
    return clamped


def backtest_with_costs(
    model,
    test_X,
    test_y,
    device,
    commission=0.001,
    slippage=0.0005,
    latency=True,
    fill_rate=0.9,
    fill_random=True,
    starting_capital=1.0,
    max_per_asset=0.3,
):
    model.eval()
    n_days = len(test_X)
    capital = starting_capital
    equity = [capital]
    daily_rets = []
    turnovers = []
    prev_weights = torch.zeros(len(TICKERS), device=device)
    with torch.no_grad():
        for t in range(n_days):
            x = test_X[t].unsqueeze(0).to(device)
            y = test_y[t].to(device)
            raw_weights = model(x).squeeze(0)
            raw_weights = apply_risk_budget(
                raw_weights.unsqueeze(0), max_per_asset
            ).squeeze(0)
            if latency and t > 0:
                target_weights = target_weights_yesterday
            else:
                target_weights = raw_weights
            target_weights_yesterday = raw_weights
            trade_size = (target_weights - prev_weights).abs().sum() * 0.5
            if fill_rate < 1.0:
                if fill_random:
                    fill = np.random.uniform(0.7 * fill_rate, 1.3 * fill_rate)
                    fill = np.clip(fill, 0.0, 1.0)
                else:
                    fill = fill_rate
                trade_size *= fill
                executed_weights = prev_weights + fill * (target_weights - prev_weights)
            else:
                executed_weights = target_weights
            cost = (commission + slippage) * trade_size
            asset_rets = y
            gross_return = (executed_weights * asset_rets).sum()
            net_return = gross_return - cost
            capital *= 1 + net_return.item()
            equity.append(capital)
            daily_rets.append(net_return.item())
            turnovers.append(trade_size.item())
            prev_weights = executed_weights
    return np.array(equity), np.array(daily_rets), np.array(turnovers)


def compute_metrics(daily_returns):
    if len(daily_returns) < 2:
        return {}
    cumulative = np.cumprod(1 + daily_returns) - 1
    mean_daily = np.mean(daily_returns)
    std_daily = np.std(daily_returns, ddof=1)
    sharpe = np.sqrt(252) * mean_daily / std_daily if std_daily != 0 else 0.0
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = (
        np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.0
    )
    sortino = np.sqrt(252) * mean_daily / downside_std if downside_std != 0 else 0.0
    cumulative_curve = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(cumulative_curve)
    drawdown = (cumulative_curve - running_max) / running_max
    max_dd = np.min(drawdown)
    positive_trades = daily_returns[daily_returns > 0]
    negative_trades = daily_returns[daily_returns < 0]
    gross_profit = np.sum(positive_trades) if len(positive_trades) > 0 else 0
    gross_loss = np.abs(np.sum(negative_trades)) if len(negative_trades) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    return {
        "total_return": cumulative[-1],
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "cumulative_curve": cumulative_curve,
        "drawdown_curve": drawdown,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    model = HiveVolPortfolio(
        n_assets=len(TICKERS), d_x=4, d_state=32, n_strategies=30, temperature=0.1
    ).to(device)
    model.load_state_dict(torch.load("hive.pt", map_location=device))
    model.eval()
    print("Preparing data for expanded tickers...")
    X_list, y_list = build_multivol_dataset()
    test_X = X_list
    test_y = y_list
    equity, daily_rets, turnovers = backtest_with_costs(
        model,
        test_X,
        test_y,
        device,
        commission=0.001,
        slippage=0.0005,
        latency=True,
        fill_rate=0.9,
        fill_random=True,
        starting_capital=1.0,
        max_per_asset=0.3,
    )
    metrics = compute_metrics(daily_rets)
    print("\n--- Zero-Shot Backtest with Risk Budget (30% cap) ---")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Average Daily Turnover: {np.mean(turnovers)*100:.2f}%")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(equity, label="Equity (after costs)")
    ax1.set_title("Hive 3.0 – Zero-Shot with Risk Budget (30% cap)")
    ax1.set_xlabel("Trading Day")
    ax1.set_ylabel("Portfolio Value")
    ax1.legend()
    ax2.fill_between(
        range(len(metrics["drawdown_curve"])),
        0,
        metrics["drawdown_curve"] * 100,
        color="red",
        alpha=0.3,
        label="Drawdown %",
    )
    ax2.set_title("Drawdown")
    ax2.set_xlabel("Trading Day")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("logs/results/hive_zeroshot.png", dpi=150)
    plt.show()
    pd.DataFrame({"daily_return": daily_rets, "turnover": turnovers}).to_csv(
        "logs/results/hive_results.csv", index=False
    )
    print(
        "Results saved to logs/results/hive_results.csv and logs/results/hive_zeroshot.png"
    )
