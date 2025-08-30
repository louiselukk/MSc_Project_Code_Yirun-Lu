import numpy as np
from .market import Market
from .agents import MarketMaker, Arbitrageur, ShortTermReversalTrader

def run_simulation(
    num_steps=2000, seed=42,
    initial_price=100, long_term_mean=100, spread=0.1, alpha=0.25, beta=0.1,
    noise_std=0.1, window_size=10,
    delta_threshold=0.01,
    mm_initial_capital=10000, mm_order_size=10, mm_target_inventory=5, mm_tolerance=2,
    arb_initial_capital=10000, arb_order_size=5, arb_threshold=0.001,
    strt_initial_capital=10000, strt_order_size=5, strt_lookback=5,
    mm_initial_strategy='passive', arb_initial_strategy='hf', strt_initial_strategy='mean',
    verbose=False
):
    np.random.seed(seed)

    market = Market(initial_price, long_term_mean, spread, alpha, beta, noise_std, window_size)
    mm  = MarketMaker(market, mm_initial_capital,  mm_order_size,  mm_target_inventory, mm_tolerance);  mm.strategy  = mm_initial_strategy
    arb = Arbitrageur (market, arb_initial_capital, arb_order_size, arb_threshold);                       arb.strategy = arb_initial_strategy
    strt= ShortTermReversalTrader(market, strt_initial_capital, strt_order_size, strt_lookback);         strt.strategy= strt_initial_strategy
    agents = [mm, arb, strt]

    for step in range(num_steps):
        avg_pnl = float(np.mean([a.profit_history[-1] for a in agents]))
        for a in agents:
            a.action(avg_pnl, step, delta_threshold)
        market.step_advance()

        if verbose and step in {int(num_steps/3), int(2*num_steps/3), num_steps - 1}:
            print(f"[t={step}] price={market.price:.4f}, vol={market.get_volatility():.4f}")

    return market, agents

def run_simulation_fixed_strategy(**kwargs):
    market, agents = run_simulation(**kwargs)
    for a in agents:
        a.evaluate_strategy_switch = lambda avg_pnl, step, th: None
    return market, agents
