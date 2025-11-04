import add_signals
import numpy as np
import get_signal_weights
import polars as pl
import sf_quant.backtester as sfb
import sf_quant.performance as sfp
import seaborn as sns
import datetime as dt
from matplotlib import pyplot as plt
import argparse

def plot_TE_vs_gamma(signal='momentum', n=10, min=-1, max=2, start=dt.date.fromisoformat('2013-06-01'), end=dt.date.fromisoformat('2013-06-30')):
    df = pl.read_parquet('../data/russell_3000_daily.parquet').sort('barrid', 'date').filter(pl.col('price').gt(5))
    signals = add_signals.add_signals(df)
    returns = signals.with_columns(pl.col('return').shift(-1).alias('fwd_return')).filter(
        (pl.col('date') >= start) &
        (pl.col('date') <= end) &
        (pl.col(f'{signal}_alpha').is_not_null())
        ).select(['date', 'barrid', f'{signal}_alpha', 'predicted_beta', 'fwd_return'
        ])
    domain = np.logspace(min, max, n)
    tracking_error = []
    for gamma in domain:
        gamma_weights = get_signal_weights.get_signal_weights(returns.lazy(), signal, start, end, gamma=gamma)
        gamma_returns = get_signal_weights.get_returns_from_weights(gamma_weights.join(returns.select(['date', 'barrid', 'fwd_return']), on=['date', 'barrid']))
        
        # gamma_table = sfp.generate_summary_table(gamma_returns)
        # print(f"For gamma={gamma}:\n{gamma_table}")
        # tracking_error.append(gamma_table.item(0, "Volatility (%)") / 100)
        active = gamma_returns.filter(pl.col('portfolio') == "active")
        tracking_error.append(active.select(pl.col("return").std(ddof=1)).item() * np.sqrt(252))

    print_dict = {gamma: tracking_error[i] for i, gamma in enumerate(domain)}
    print(print_dict)
    
    plt.plot(domain, tracking_error)
    plt.title(f"Tracking Error vs. Gamma ({signal}, {start}, {n})")
    plt.xlabel("Gamma")
    plt.ylabel("Tracking Error")
    plt.savefig(f'plots/TE_vs_gamma_{signal}_{start}_{n}.png')
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot TE vs. Gamma"
    )
    parser.add_argument("signal", help="Signal name (str)")
    parser.add_argument("n", help="Number of points to plot (int)")
    parser.add_argument("start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end", help="End date (YYYY-MM-DD)")

    args = parser.parse_args()
    print(f'Running plot_TE_vs_gamma with args={args}...')

    plot_TE_vs_gamma(signal=args.signal, n=int(args.n), start=dt.date.fromisoformat(args.start), end=dt.date.fromisoformat(args.end))
    
