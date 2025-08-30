import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from abm_market.simulate import run_simulation, run_simulation_fixed_strategy
from abm_viz.sharpe_plots import plot_combined_figures
from abm_viz.pnl_vs_params import scan_param_three_agents, plot_scan_three_agents
from abm_viz.acf_plots import collect_clustering_by_param_and_noise, plot_clustering_with_errorbars
from abm_viz.payoff_ternary import run_regime_matrix_rowsum, plot_ternary_rowsum
from abm_market.directional_accuracy import direction_stats_all

BASE = dict(num_steps=1000, seed=42, initial_price=100, long_term_mean=100,
            spread=0.1, alpha=0.25, beta=0.1, noise_std=0.1, window_size=10,
            delta_threshold=0.01,
            mm_initial_capital=10000, mm_order_size=10, mm_target_inventory=5, mm_tolerance=2,
            arb_initial_capital=10000, arb_order_size=5, arb_threshold=0.001,
            strt_initial_capital=10000, strt_order_size=5, strt_lookback=5)

if __name__ == "__main__":
    # Simulation plots
    m, ags = run_simulation(**BASE)
    plot_combined_figures(m, ags, "results/figures/combined.png")

    values = [0,0.001,0.005,0.01,0.05,0.1]
    mean_logs = scan_param_three_agents(run_simulation, "noise_std", values, BASE, n_runs=50, use_delta=True, normalize=True)
    plot_scan_three_agents("noise_std", values, mean_logs, use_delta=True, normalize=True)

    # ACF
    res = collect_clustering_by_param_and_noise(run_simulation, "alpha", [0,0.1,0.2,0.3,0.4,0.5], [0,0.001,0.01,0.05,0.1])
    plot_clustering_with_errorbars([0,0.1,0.2,0.3,0.4,0.5], res, "α (impact factor)")

    # Triangle plots
    REGIMES = {'baseline': dict(spread=0.10, noise_std=0.10, window_size=10)}
    df = run_regime_matrix_rowsum(run_simulation_fixed_strategy, BASE, REGIMES['baseline'])
    plot_ternary_rowsum(df, title="Baseline – Ternary (Row-Sum)", fname="results/figures/baseline_ternary.png")

    # Directional Accuracy
    print(direction_stats_all(run_simulation, BASE, noise_std=0.1, seeds=range(10)))
