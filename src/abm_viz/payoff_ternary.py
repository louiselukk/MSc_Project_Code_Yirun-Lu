import numpy as np
import pandas as pd
from itertools import product
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run_regime_matrix_raw(run_simulation_fixed_strategy, base_params, regime_overrides,
                          mm_axis=('passive','adaptive'),
                          arb_axis=('hf','lf'),
                          str_axis=('mean','momentum')):
    params = base_params.copy(); params.update(regime_overrides)
    rows = []
    for mm_s, arb_s, str_s in product(mm_axis, arb_axis, str_axis):
        mkt, agents = run_simulation_fixed_strategy(**params,
                                                    mm_initial_strategy=mm_s,
                                                    arb_initial_strategy=arb_s,
                                                    strt_initial_strategy=str_s)
        mm_pnl, arb_pnl, str_pnl = [a.calculate_pnl() for a in agents]
        rows.append({"MM Strategy": mm_s, "Arb Strategy": arb_s, "STR Strategy": str_s,
                     "MM PnL": mm_pnl, "Arb PnL": arb_pnl, "STR PnL": str_pnl})
    return pd.DataFrame(rows)

def _rowwise_sum_norm(triple, eps=1e-12):
    m,a,s = map(float, triple); tot = m+a+s
    if (m<0) or (a<0) or (s<0) or (tot<=eps):
        shift = max(0.0, -min(m,a,s)) + 1e-12
        m,a,s = m+shift, a+shift, s+shift; tot = m+a+s
    return [m/tot, a/tot, s/tot]

def run_regime_matrix_rowsum(run_simulation_fixed_strategy, base_params, regime_overrides,
                             mm_axis=('passive','adaptive'),
                             arb_axis=('hf','lf'),
                             str_axis=('mean','momentum')):
    df_raw = run_regime_matrix_raw(run_simulation_fixed_strategy, base_params, regime_overrides, mm_axis, arb_axis, str_axis)
    rows = []
    for _, r in df_raw.iterrows():
        m,a,s = _rowwise_sum_norm([r["MM PnL"], r["Arb PnL"], r["STR PnL"]])
        rows.append({"MM Strategy": r["MM Strategy"], "Arb Strategy": r["Arb Strategy"],
                     "STR Strategy": r["STR Strategy"], "MM PnL": m, "Arb PnL": a, "STR PnL": s})
    return pd.DataFrame(rows)

def plot_ternary_rowsum(df_rowsum, title="Ternary (Row-Sum)", fname="results/ternary_rowsum.png", dpi=300):
    fig, ax = plt.subplots(figsize=(7,6.5))
    Ax,Ay = 0.0,0.0; Bx,By = 1.0,0.0; Cx,Cy = 0.5, np.sqrt(3)/2
    ax.plot([Ax,Bx,Cx,Ax], [Ay,By,Cy,Ay], lw=1.6, color='black')
    ax.set_aspect('equal', adjustable='box'); ax.axis('off')
    ax.text(Ax-0.03, Ay-0.04, "MM", fontsize=12)
    ax.text(Bx+0.02, By-0.04, "Arb", fontsize=12)
    ax.text(Cx, Cy+0.02, "STR", ha='center', fontsize=12)

    cmap = plt.colormaps["tab10"]
    combos = df_rowsum[["MM Strategy","Arb Strategy","STR Strategy"]].drop_duplicates().values
    strat_to_color = {tuple(t): cmap(i % 10) for i,t in enumerate(combos)}

    for _, row in df_rowsum.iterrows():
        m,a,s = row["MM PnL"], row["Arb PnL"], row["STR PnL"]
        x = m*Ax + a*Bx + s*Cx; y = m*Ay + a*By + s*Cy
        vals = np.array([m,a,s]); edge = 'black' if np.isclose(vals, vals.max()).sum()==1 else None
        color = strat_to_color[(row["MM Strategy"], row["Arb Strategy"], row["STR Strategy"])]
        ax.scatter(x, y, s=70, marker='o', color=color, edgecolors=edge, linewidths=1.2 if edge else 0)

    handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markeredgecolor='black',
                           markersize=8, label=f"MM={mm}, Arb={ab}, STR={st}")
               for (mm,ab,st), c in strat_to_color.items()]
    ax.legend(handles=handles, bbox_to_anchor=(1.05,1), loc='upper left', frameon=False)
    ax.set_title(title, pad=12)
    plt.savefig(fname, dpi=dpi, bbox_inches='tight'); plt.close(fig)
