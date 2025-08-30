import matplotlib
matplotlib.use("Agg")       
import matplotlib.pyplot as plt
import numpy as np
from abm_market.metrics import build_equity_series, sharpe_from_equity

def plot_combined_figures(market, agents, fname="results/figures/combined_plots.png"):
    T = len(market.step_price_history)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_price, ax_pnl, ax_inv, ax_sharp = axes.flatten()

    ax_price.plot(market.price_history); ax_price.set_title("Market Price"); ax_price.set_ylabel("Price"); ax_price.grid(True); ax_price.set_xlim(0, T-1)

    for ag in agents:
        ax_pnl.plot(ag.profit_history, label=ag.__class__.__name__)
    ax_pnl.set_title("Agent Profit/Loss"); ax_pnl.set_ylabel("P&L"); ax_pnl.legend(); ax_pnl.grid(True); ax_pnl.set_xlim(0, T-1)

    for ag in agents:
        ax_inv.plot(ag.inventory_history, label=ag.__class__.__name__)
    ax_inv.set_title("Agent Inventory"); ax_inv.set_ylabel("Inventory"); ax_inv.legend(); ax_inv.grid(True); ax_inv.set_xlim(0, T-1)

    for ag in agents:
        eq = build_equity_series(ag)
        sr = sharpe_from_equity(eq, steps_per_year=252, risk_free_annual=0.0, use_log=False)
        if len(sr) < T:
            sr = np.concatenate([sr, np.full(T - len(sr), np.nan)])
        else:
            sr = sr[:T]
        k0 = int(np.argmax(np.isfinite(sr)))
        if np.isfinite(sr[k0]): sr[:k0] = sr[k0]
        ax_sharp.plot(sr, label=ag.__class__.__name__)
    ax_sharp.set_title("Annualized Sharpe (Expanding)"); ax_sharp.set_ylabel("Sharpe"); ax_sharp.set_xlim(0, T-1); ax_sharp.grid(True); ax_sharp.legend()

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight"); plt.close(fig)
