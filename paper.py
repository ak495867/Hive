import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math
import warnings
from datetime import datetime

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
DATA_START = "2016-01-01"
PAPER_START = "2023-01-01"
PAPER_END = "2024-08-01"


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


def download_data(start, end):
    print(f"Downloading data from {start} to {end}...")
    raw_data = {}
    failed = []
    for t in TICKERS:
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if df.empty:
                failed.append(t)
            else:
                raw_data[t] = df["Close"]
        except Exception as e:
            print(f"Failed {t}: {e}")
            failed.append(t)
    if failed:
        print(f"Warning: {len(failed)} tickers filled with 1.0: {failed}")
    bdays = pd.bdate_range(start=start, end=end)
    df_close = pd.DataFrame(index=bdays, columns=TICKERS)
    for t, s in raw_data.items():
        df_close[t] = s.reindex(bdays)
    for t in failed:
        df_close[t] = 1.0
    df_close = df_close.ffill().bfill().dropna(how="all")
    return df_close


def precompute_features(df_close):
    T = len(df_close)
    n_assets = len(TICKERS)
    feat = np.zeros((T, n_assets, 4), dtype=np.float32)
    for a, t in enumerate(TICKERS):
        close_arr = df_close[t].values.astype(np.float32)
        feat[:, a, :] = compute_asset_features(close_arr)
    return feat


def smooth_weights(new_w, prev_w, factor=0.3):
    if prev_w is None:
        return new_w
    return factor * new_w + (1 - factor) * prev_w


def get_next_business_day_index(index, date_str):
    target = pd.Timestamp(date_str)
    if target in index:
        return index.get_loc(target)
    pos = index.get_indexer([target], method="bfill")[0]
    if pos < 0:
        raise KeyError(f"No business day found on or after {date_str}")
    return pos


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    model = HiveVolPortfolio(
        n_assets=len(TICKERS), d_x=4, d_state=32, n_strategies=30, temperature=0.1
    ).to(device)
    model.load_state_dict(torch.load("model/hive.pt", map_location=device))
    model.eval()

    df_close = download_data(DATA_START, PAPER_END)
    features_all = precompute_features(df_close)
    n_assets = len(TICKERS)
    T = len(df_close)

    paper_start_idx = get_next_business_day_index(df_close.index, PAPER_START)
    paper_end_idx = get_next_business_day_index(df_close.index, PAPER_END)

    capital = 1.0
    equity_curve = [capital]
    daily_returns = []
    turnovers = []

    prev_weights = torch.zeros(n_assets, device=device)
    prev_smoothed = None
    prev_target = None

    params = {
        "temperature": 0.5,
        "smoothing": 0.3,
        "turnover_limit": 0.15,
        "vol_target": 0.15,
        "commission": 0.001,
        "slippage": 0.0005,
        "latency": True,
        "fill_rate": 0.9,
        "fill_random": True,
        "max_per_asset": 0.3,
    }

    cumsum = np.cumsum(features_all.reshape(T, -1), axis=0)
    cumsum_sq = np.cumsum((features_all.reshape(T, -1)) ** 2, axis=0)

    print(f"\n=== LIVE PAPER-TRADING SIMULATION ===\n")
    print(
        f"Start: {df_close.index[paper_start_idx].date()}  End: {df_close.index[paper_end_idx].date()}"
    )
    print(
        f"Costs: comm={params['commission']}, slip={params['slippage']}, fill_rate={params['fill_rate']}"
    )
    print(
        f"Latency: {params['latency']}  Turnover cap: {params['turnover_limit']}  Vol target: {params['vol_target']}"
    )
    print("-" * 70)

    for i in range(paper_start_idx, paper_end_idx):
        if i < LOOKBACK:
            continue

        current_date = df_close.index[i]
        x_window = features_all[i - LOOKBACK + 1 : i + 1].transpose(1, 0, 2)
        n = i + 1
        mean = cumsum[i] / n
        var = cumsum_sq[i] / n - mean**2
        std = np.sqrt(var + 1e-8)
        norm_window = (x_window.reshape(LOOKBACK, -1) - mean) / std
        norm_window = norm_window.reshape(n_assets, LOOKBACK, 4)
        x_tensor = torch.FloatTensor(norm_window).unsqueeze(0).to(device)

        with torch.no_grad():
            model.temperature = params["temperature"]
            raw_weights = model(x_tensor).squeeze(0)

        raw_weights = torch.clamp(raw_weights, max=params["max_per_asset"])
        raw_weights /= raw_weights.sum()

        raw_weights = smooth_weights(raw_weights, prev_smoothed, params["smoothing"])
        prev_smoothed = raw_weights.clone()

        if len(daily_returns) >= 20:
            port_vol = np.std(daily_returns[-20:], ddof=1) * np.sqrt(252)
            if port_vol > 0:
                scale = params["vol_target"] / port_vol
                raw_weights = raw_weights * scale

        w_sum = raw_weights.sum()
        if w_sum > 1.0:
            raw_weights /= w_sum

        if params["latency"] and prev_target is not None:
            target_weights = prev_target
        else:
            target_weights = raw_weights
        prev_target = raw_weights.clone()

        desired_trade = (target_weights - prev_weights).abs().sum() * 0.5

        if desired_trade > params["turnover_limit"]:
            scale = params["turnover_limit"] / desired_trade
            executed = prev_weights + scale * (target_weights - prev_weights)
            trade_size = params["turnover_limit"]
        else:
            executed = target_weights
            trade_size = desired_trade.item()

        if params["fill_rate"] < 1.0:
            if params["fill_random"]:
                fill = np.random.uniform(
                    0.7 * params["fill_rate"], 1.3 * params["fill_rate"]
                )
                fill = np.clip(fill, 0.0, 1.0)
            else:
                fill = params["fill_rate"]
            trade_size *= fill
            executed = prev_weights + fill * (executed - prev_weights)

        cost = (params["commission"] + params["slippage"]) * trade_size

        if i + 1 >= T:
            break
        tomorrow_close = df_close.iloc[i + 1].values.astype(np.float32)
        today_close = df_close.iloc[i].values.astype(np.float32)
        asset_rets = (tomorrow_close - today_close) / (today_close + 1e-8)
        asset_rets = torch.FloatTensor(asset_rets).to(device)

        gross_ret = (executed * asset_rets).sum()
        net_ret = gross_ret - cost
        capital *= 1 + net_ret.item()
        equity_curve.append(capital)
        daily_returns.append(net_ret.item())
        turnovers.append(trade_size)

        held = (executed.abs() > 1e-6).sum().item()
        print(
            f"{current_date.date()} | Value: {capital:,.4f} | Return: {net_ret*100:+.3f}% | "
            f"Trade: {trade_size*100:.2f}% | Assets held: {held} | Cost: {cost*100:.3f}%"
        )

        prev_weights = executed.clone()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== PAPER-TRADING RESULTS ===")
    daily_arr = np.array(daily_returns)
    if len(daily_arr) > 0:
        cum_ret = np.cumprod(1 + daily_arr) - 1
        mean_d = np.mean(daily_arr)
        std_d = np.std(daily_arr, ddof=1)
        sharpe = np.sqrt(252) * mean_d / std_d if std_d > 0 else 0.0
        downside = daily_arr[daily_arr < 0]
        sortino = (
            np.sqrt(252)
            * mean_d
            / (np.std(downside, ddof=1) if len(downside) > 1 else 1e-8)
        )
        curve = np.cumprod(1 + daily_arr)
        dd = (curve - np.maximum.accumulate(curve)) / np.maximum.accumulate(curve)
        max_dd = np.min(dd) * 100
        pos = daily_arr[daily_arr > 0]
        neg = daily_arr[daily_arr < 0]
        pf = np.sum(pos) / np.abs(np.sum(neg)) if len(neg) > 0 else np.inf

        print(f"Total Return: {cum_ret[-1]*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.3f}")
        print(f"Sortino Ratio: {sortino:.3f}")
        print(f"Max Drawdown: {max_dd:.2f}%")
        print(f"Profit Factor: {pf:.2f}")
        print(f"Avg Turnover: {np.mean(turnovers)*100:.2f}%")

        log_df = pd.DataFrame(
            {
                "date": df_close.index[paper_start_idx:paper_end_idx],
                "equity": equity_curve[1:],
                "return": daily_arr,
                "turnover": turnovers,
            }
        )
        log_df.to_csv("logs/results/paper_trading_log.csv", index=False)
        print("Log saved to logs/results/paper_trading_log.csv")

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(log_df["date"], equity_curve[1:])
        plt.title("Paper-Trading Equity Curve")
        plt.subplot(2, 1, 2)
        plt.fill_between(log_df["date"], 0, dd * 100, color="red", alpha=0.3)
        plt.title("Drawdown (%)")
        plt.tight_layout()
        plt.savefig("logs/results/paper_trading_plot.png", dpi=150)
        plt.show()
    else:
        print("No trades executed.")
