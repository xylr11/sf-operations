import matplotlib
matplotlib.use("Agg") # Don't interactively plot, since the supercomputer won't enjoy that
import faulthandler
faulthandler.enable()
import add_signals
import numpy as np
import get_signal_weights
import polars as pl
import datetime as dt
from matplotlib import pyplot as plt
import argparse
from scipy.stats import linregress
import sf_quant.optimizer as sfo

def plot_active_risk_vs_gamma(signal='momentum', n=10, min=-1, max=2.5, start=dt.date.fromisoformat('2013-06-01'), end=dt.date.fromisoformat('2013-06-30'), historical=True):
    df = pl.read_parquet('../data/russell_3000_daily.parquet').sort('barrid', 'date') # .filter(pl.col('price').gt(5)) Temporarily removing price filter
    signals = add_signals.add_signals(df)
    returns = signals.lazy().with_columns(pl.col('return').shift(-1).alias('fwd_return')).filter(
        (pl.col('date') >= start) &
        (pl.col('date') <= end) &
        (pl.col(f'{signal}_alpha').is_not_null())
        ).select(['date', 'barrid', f'{signal}_alpha', 'predicted_beta', 'fwd_return'
        ])
    domain = np.logspace(min, max, n)
    active_risk = []
    for gamma in domain:
        if signal == 'bab':
            gamma_weights = get_signal_weights.get_signal_weights(returns, signal, start, end, gamma=gamma, constraints=[sfo.UnitBeta()])
        else:
            gamma_weights = get_signal_weights.get_signal_weights(returns, signal, start, end, gamma=gamma)

        if historical:
            gamma_returns = get_signal_weights.get_returns_from_weights(gamma_weights.join(returns.select(['date', 'barrid', 'fwd_return']).collect(), on=['date', 'barrid']))
            active = gamma_returns.filter(pl.col('portfolio') == "active").with_columns(pl.col('return').truediv(100.))
            active_risk.append(active.select(pl.col("return").std(ddof=1)).item() * np.sqrt(252.))
        else:
            active_weights = get_signal_weights.get_active_weights_from_weights(gamma_weights)
            # M: Finish this here by taking the sandwich product with the covariance matrix on each day, and averaging the active risks. 
            # Then append to active_risk, and the plotting should just work as is.
            # The formula is active risk = \sqrt{w_a^T \Sigma w_a}, I believe (and then annualize by multiplying by \sqrt{252}).

    print_dict = {gamma: active_risk[i] for i, gamma in enumerate(domain)}
    print(print_dict)
    
    (m, b, r, p, err) = linregress(np.log(domain[n // 2:]), np.log(active_risk[n // 2:]))
    print(m, b, r, p, err)

    plt.scatter(domain, active_risk, label="Backtested points")
    plt.plot(
        domain[n // 2:], np.exp(m * np.log(domain[n // 2:]) + b),
        label=f"$\\log{{\\gamma}}$ regression: $m={m:.4g},\\ b={b:.4g}$\n"
            f"$r={r:.4g},\\ p={p:.4g},\\ err={err:.4g}$")
    plt.title(f"Active Risk vs. Gamma ({signal}, {start}, {n})")
    plt.xlabel("Gamma")
    plt.ylabel("Active Risk")
    plt.legend()
    plt.savefig(f'plots/log_regression_active_risk_vs_gamma_{signal}_{start}_{n}.png')
    plt.clf()
    plt.close("all")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Active Risk vs. Gamma"
    )
    parser.add_argument("signal", help="Signal name (str)")
    parser.add_argument("n", help="Number of points to plot (int)")
    parser.add_argument("start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end", help="End date (YYYY-MM-DD)")

    args = parser.parse_args()
    print(f'Running plot_active_risk_vs_gamma with args={args}...')

    plot_active_risk_vs_gamma(signal=args.signal, n=int(args.n), start=dt.date.fromisoformat(args.start), end=dt.date.fromisoformat(args.end))