import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x  # 无 tqdm 时也能跑

def scan_param_three_agents(
    run_simulation,
    param_name, values, default_params,
    n_runs=100, tail_window=500, use_delta=False, normalize=True, seed_base=0
):
    seeds = [seed_base + i for i in range(n_runs)]
    mean_logs = {0: [], 1: [], 2: []}

    for v in tqdm(values, desc=f"Scan {param_name}"):
        pnl_runs = {0: [], 1: [], 2: []}
        for s in seeds:
            kw = default_params.copy(); kw['seed'] = s; kw[param_name] = v
            market, agents = run_simulation(**kw)
            for i, ag in enumerate(agents):
                ph = ag.profit_history
                pnl = (ph[-1] - ph[-1 - tail_window]) if use_delta else ph[-1]
                pnl_runs[i].append(pnl)

        for i in [0, 1, 2]:
            arr = np.array(pnl_runs[i], float)
            shift = (-arr.min() + 1e-6) if arr.min() <= 0 else 0.0
            logs = np.log(arr + shift)
            mean_logs[i].append(np.nanmean(np.where(np.isfinite(logs), logs, np.nan)))

    if normalize:
        all_logs = np.concatenate([np.array(mean_logs[0]), np.array(mean_logs[1]), np.array(mean_logs[2])]).reshape(-1,1)
        mask = np.isfinite(all_logs.flatten()); all_clean = all_logs[mask].reshape(-1,1)
        if all_clean.size:
            scaler = MinMaxScaler(); all_scaled = scaler.fit_transform(all_clean).flatten()
            full_scaled = np.full(all_logs.shape[0], np.nan); full_scaled[mask] = all_scaled
            L = len(mean_logs[0])
            mean_logs[0] = full_scaled[0:L]; mean_logs[1] = full_scaled[L:2*L]; mean_logs[2] = full_scaled[2*L:3*L]
        else:
            L = len(mean_logs[0]); mean_logs = {i: np.full(L, np.nan) for i in [0,1,2]}
    else:
        for i in [0,1,2]: mean_logs[i] = np.array(mean_logs[i])
    return mean_logs

def plot_scan_three_agents(param_name, values, mean_logs, use_delta=False, normalize=True, outdir="results/plots_vs"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,6))
    labels = ['MarketMaker', 'Arbitrageur', 'ShortTermReversalTrader']
    for i in [0,1,2]:
        x = np.array(values); y = np.array(mean_logs[i]); mask = np.isfinite(y)
        plt.plot(x[mask], y[mask], marker='o', label=labels[i])
    ylabel = ("Min-Max Normalized " if normalize else "") + ("Mean Log PnL (Δ tail)" if use_delta else "Mean Log PnL")
    plt.xlabel(param_name); plt.ylabel(ylabel); plt.title(f"PnL vs {param_name}")
    plt.grid(True); plt.legend(); plt.tight_layout()
    fname = os.path.join(outdir, f"scan_{param_name}_all_agents.png")
    plt.savefig(fname, dpi=300); plt.close(); print(f"Saved → {fname}")
