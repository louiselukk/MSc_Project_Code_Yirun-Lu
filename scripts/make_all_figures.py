import os, pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
ROOT = pathlib.Path(__file__).resolve().parents[1]
(FIG_DIR := ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
(PVS_DIR := ROOT / "results" / "plots_vs").mkdir(parents=True, exist_ok=True)
(ACF_DIR := ROOT / "results" / "plots_timeavg_acf_clean").mkdir(parents=True, exist_ok=True)
from abm_market.simulate import run_simulation, run_simulation_fixed_strategy
from abm_viz.sharpe_plots import plot_combined_figures
from abm_viz.pnl_vs_params import scan_param_three_agents, plot_scan_three_agents
from abm_viz.acf_plots import plot_timeavg_acf_subplots_compare
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
    alpha_values  = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    beta_values   = [0, 0.05, 0.1, 0.15, 0.2, 0.3]
    noise_values  = [0, 0.001, 0.005, 0.01, 0.05, 0.1]
    spread_values = [0, 0.01, 0.05, 0.1, 0.2, 0.3]
    window_values = [5, 10, 20, 30, 50]

    num_steps = 1000
    max_lag   = 20

    plot_timeavg_acf_subplots_compare('alpha',       alpha_values,  BASE, num_steps, max_lag, 100, str(save_dir))
    plot_timeavg_acf_subplots_compare('beta',        beta_values,   BASE, num_steps, max_lag, 200, str(save_dir))
    plot_timeavg_acf_subplots_compare('noise_std',   noise_values,  BASE, num_steps, max_lag, 300, str(save_dir))
    plot_timeavg_acf_subplots_compare('spread',      spread_values, BASE, num_steps, max_lag, 400, str(save_dir))
    plot_timeavg_acf_subplots_compare('window_size', window_values, BASE, num_steps, max_lag, 500, str(save_dir))


    # Triangle plots
    REGIMES = {'baseline': dict(spread=0.10, noise_std=0.10, window_size=10)}
    df = run_regime_matrix_rowsum(run_simulation_fixed_strategy, BASE, REGIMES['baseline'])
    plot_ternary_rowsum(df, title="Baseline – Ternary (Row-Sum)", fname="results/figures/baseline_ternary.png")

    # Directional Accuracy
    print(direction_stats_all(run_simulation, BASE, noise_std=0.1, seeds=range(10)))
