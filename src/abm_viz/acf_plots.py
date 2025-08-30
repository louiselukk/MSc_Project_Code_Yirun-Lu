import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def acf_time_avg(x, max_lag):
    x = np.asarray(x, float); N = len(x)
    if N == 0: return np.full(max_lag+1, np.nan)
    mu = x.mean(); denom = np.sum((x - mu)**2); acf = np.empty(max_lag+1)
    for k in range(max_lag+1):
        if k >= N or denom == 0: acf[k] = np.nan
        else: acf[k] = np.sum((x[:N-k]-mu)*(x[k:]-mu)) / denom
    return acf

def compute_clustering_strength(log_returns, max_lag=5):
    acf_abs = acf_time_avg(np.abs(log_returns), max_lag)
    return float(np.nanmean(acf_abs[1:max_lag+1]))

def collect_clustering_by_param_and_noise(run_simulation, param_name, param_values, noise_values,
                                          num_runs=10, num_steps=1000, max_lag=5):
    results = {}
    for noise in noise_values:
        means, stds = [], []
        for val in param_values:
            scores = []
            for run in range(num_runs):
                kwargs = dict(num_steps=num_steps, seed=1000+run+int(noise*1000),
                              noise_std=noise, alpha=0.2, beta=0.05, window_size=50, spread=0.05, arb_threshold=0.002)
                kwargs[param_name] = val
                market, _ = run_simulation(**kwargs)
                log_r = np.diff(np.log(np.array(market.price_history)))
                if np.std(log_r) > 0: log_r = log_r / np.std(log_r)
                scores.append(compute_clustering_strength(log_r, max_lag))
            means.append(np.mean(scores)); stds.append(np.std(scores))
        results[noise] = (means, stds)
    return results

def plot_clustering_with_errorbars(param_values, results, param_name, out="results/clustering"):
    os.makedirs(out, exist_ok=True)
    plt.figure(figsize=(10,6))
    for noise, (means, stds) in results.items():
        plt.errorbar(param_values, means, yerr=stds, fmt='o--', capsize=4, label=f"σ={noise:.3f}")
    plt.title(f"Volatility Clustering vs {param_name}")
    plt.xlabel(param_name); plt.ylabel("Mean ACF(|log return|)"); plt.grid(True); plt.legend(); plt.tight_layout()
    fname = os.path.join(out, f"clustering_vs_{param_name}.png")
    plt.savefig(fname); plt.close(); print(f"Saved → {fname}")
