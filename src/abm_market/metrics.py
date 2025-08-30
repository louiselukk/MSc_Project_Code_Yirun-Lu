import numpy as np

def build_equity_series(agent):
    prices = np.asarray(agent.market.step_price_history, float)
    cap    = np.asarray(agent.capital_history, float)
    inv    = np.asarray(agent.inventory_history, float)
    n = min(len(prices), len(cap), len(inv))
    return cap[:n] + inv[:n] * prices[:n]

def sharpe_from_equity(equity, steps_per_year=252, risk_free_annual=0.0, use_log=False):
    E = np.asarray(equity, float)
    n = len(E)
    sr = np.full(n, np.nan)
    if n < 3: return sr
    rf_step = (1.0 + risk_free_annual)**(1.0/steps_per_year) - 1.0
    r = np.log(E[1:]/E[:-1]) if use_log else (E[1:]/E[:-1] - 1.0)
    ex = r - rf_step
    mean = 0.0; M2 = 0.0
    for k in range(1, len(ex)+1):
        x = ex[k-1]
        delta = x - mean
        mean += delta / k
        M2   += delta * (x - mean)
        if k >= 2:
            var = M2 / (k - 1); sd = np.sqrt(var)
            if sd > 0: sr[k] = (mean / sd) * np.sqrt(steps_per_year)
    return sr

def last_valid(x):
    arr = np.asarray(x, float).ravel()
    ok = np.isfinite(arr)
    return float(arr[ok][-1]) if ok.any() else np.nan
