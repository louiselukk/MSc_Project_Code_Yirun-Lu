import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            
import matplotlib.pyplot as plt
from abm_market.market import Market
from abm_market.agents import MarketMaker, Arbitrageur, ShortTermReversalTrader


def acf_time_avg(x, max_lag):
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N == 0:
        return np.full(max_lag + 1, np.nan)
    mu = x.mean()
    denom = np.sum((x - mu) ** 2)
    acf = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        if k >= N or denom == 0:
            acf[k] = np.nan
        else:
            acf[k] = np.sum((x[:N - k] - mu) * (x[k:] - mu)) / denom
    return acf


def plot_timeavg_acf_subplots_compare(param_name, values, base_params,
                                      num_steps=1000, max_lag=20,
                                      seed_base=100,
                                      save_dir="plots_timeavg_acf_clean",
                                      subfig_size=(3, 3)):
    out_dir = os.path.join(save_dir, param_name + "_compare")
    os.makedirs(out_dir, exist_ok=True)

    n = len(values)
    if n <= 5:
        rows, cols = 1, n
    else:
        cols = int(np.ceil(n / 2))
        rows = 2

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(subfig_size[0] * cols, subfig_size[1] * rows),
        squeeze=False
    )
    axes = axes.flatten()

    for i, val in enumerate(values):
        ax = axes[i]
        seed = seed_base + i
        params = base_params.copy()
        params[param_name] = val

        market = run_simulation_no_print(
            num_steps=num_steps,
            seed=seed,
            **params
        )
        prices = np.array(market.price_history)
        log_returns = np.diff(np.log(prices))
        abs_log_returns = np.abs(log_returns)

        acf_log = acf_time_avg(log_returns, max_lag)
        acf_abs = acf_time_avg(abs_log_returns, max_lag)

        acf_log = np.clip(acf_log, 0, 1)
        acf_abs = np.clip(acf_abs, 0, 1)

        lags = np.arange(max_lag + 1)
        ax.plot(lags, acf_log, '-', label='log return', color='C0', linewidth=2)
        ax.plot(lags, acf_abs, '--', label='|log return|', color='C1', linewidth=2)

        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlim(0, max_lag)
        ax.set_ylim(-0.1, 1)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title(f"{param_name}={val}")
        ax.grid(True, linestyle=':', linewidth=0.5)
        ax.legend()

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    out_file = os.path.join(out_dir, f"timeavg_acf_{param_name}_compare.png")
    plt.savefig(out_file)
    plt.close(fig)
    print(f"Saved ACF vs |ACF| comparison for {param_name} → {out_file}")


def run_simulation_no_print(
    num_steps=1000,
    seed=42,
    initial_price=100,
    long_term_mean=100,
    spread=0.1,
    alpha=0.1,
    beta=0.05,
    noise_std=0.1,
    window_size=10,
    delta_threshold=0.01,
    mm_initial_capital=10000,
    mm_order_size=10,
    mm_target_inventory=5,
    mm_tolerance=2,
    arb_initial_capital=10000,
    arb_order_size=5,
    arb_threshold=0.001,
    strt_initial_capital=10000,
    strt_order_size=5,
    strt_lookback=5
):
    np.random.seed(seed)

    market = Market(
        initial_price=initial_price,
        long_term_mean=long_term_mean,
        spread=spread,
        alpha=alpha,
        beta=beta,
        noise_std=noise_std,
        window_size=window_size
    )

    mm = MarketMaker(market, mm_initial_capital, mm_order_size, mm_target_inventory, mm_tolerance)
    arb = Arbitrageur(market, arb_initial_capital, arb_order_size, arb_threshold)
    st = ShortTermReversalTrader(market, strt_initial_capital, strt_order_size, strt_lookback)

    agents = [mm, arb, st]

    for step in range(num_steps):
        avg_pnl = np.mean([a.profit_history[-1] for a in agents])
        for a in agents:
            a.action(avg_pnl, step, delta_threshold)
        market.step_advance()

    return market
