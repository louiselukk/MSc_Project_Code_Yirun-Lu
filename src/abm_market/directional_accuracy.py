import numpy as np

def direction_stats_all(run_simulation, default_params, noise_std, seeds=range(10)):
    def collect_for(class_name):
        wrong_rates, weighted_wrongs = [], []
        for s in seeds:
            kw = default_params.copy(); kw['noise_std'] = noise_std; kw['seed'] = s
            market, agents = run_simulation(**kw)
            agent = next(a for a in agents if a.__class__.__name__ == class_name)

            P = np.array(market.step_price_history, float)
            if len(P) < 3: continue
            r = np.diff(np.log(P))
            I = np.array(agent.inventory_history, float)

            L = min(len(I) - 2, len(r) - 1)
            if L <= 0: continue
            I_t = I[1:1+L]; r_next = r[1:1+L]

            m = (I_t != 0) & (r_next != 0)
            if not np.any(m): continue
            wrong = (np.sign(I_t[m]) != np.sign(r_next[m])).astype(float)
            wrong_rate = float(wrong.mean())

            w = np.abs(r_next[m]); wsum = w.sum()
            if wsum == 0: continue
            weighted_wrong = float((wrong * (w/wsum)).sum())

            wrong_rates.append(wrong_rate); weighted_wrongs.append(weighted_wrong)

        mm = float(np.mean(wrong_rates)) if wrong_rates else np.nan
        mw = float(np.mean(weighted_wrongs)) if weighted_wrongs else np.nan
        return mm, mw

    return collect_for('MarketMaker'), collect_for('Arbitrageur'), collect_for('ShortTermReversalTrader')
