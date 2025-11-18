import polars as pl
from numpy.linalg import pinv
import numpy as np
import sf_quant.data as sfd
import datetime as dt
from sf_quant.data._factors import factors
from sf_quant.data.covariance_matrix import _construct_factor_covariance_matrix
from time import perf_counter
from add_signals import add_signals

def load_factor_covariances(start, end):
    pass # At some point this should probably be written

def construct_covariance_pinv_matrix(date, timer=False):
    if timer:
        start = perf_counter()

    factor_exposures = sfd.load_exposures(date, date, True, ["date", "barrid"] + factors)
    specific_risk = sfd.load_assets_by_date(date, True, ["date", "barrid", "specific_risk"]) # The sfd load functions have their kwargs out of order :skull:
    A = np.diag(specific_risk.sort("barrid").fill_nan(0).fill_null(0).select('specific_risk').to_numpy().flatten()) / 1e4
    U = factor_exposures.sort("barrid").fill_nan(0).fill_null(0).select(factors).to_numpy()
    C = _construct_factor_covariance_matrix(date).select(factors).to_numpy() / 1e4
    cov = U @ C @ U.T + A

    if timer:
        end = perf_counter()
        print(f"[INFO] Loading data for construct_covariance_pinv_matrix on {date} took {end - start} seconds.")
        start = perf_counter()
        
    A_pinv = pinv(A)
    cov_pinv = A_pinv - A_pinv @ U @ pinv(pinv(C) + U.T @ A_pinv @ U) @ U.T @ A_pinv
    I_p = cov @ cov_pinv

    if not np.allclose(I_p, np.eye(I_p.shape[0])):
        raise Warning(f"[WARNING] {date}: cov_pinv is not a true inverse (this can generally be ignored).")
    
    if timer:
        end = perf_counter()
        print(f"[INFO] Computing the inverse on {date} took {end - start} seconds.")

    return cov_pinv

def get_gamma_from_alphas(alpha: pl.DataFrame, target: float, date):
    beta = sfd.load_assets_by_date(date, True, ["date", "barrid", "predicted_beta"]).sort("barrid").select("predicted_beta").to_numpy()
    constraints = np.hstack([np.ones_like(beta), beta]).T
    print(constraints)

if __name__ == "__main__":
    df = pl.read_parquet('../data/russell_3000_daily.parquet').sort('barrid', 'date').filter(pl.col('price').gt(5))
    signals = add_signals(df)
    alpha = signals.filter(pl.col("date").eq(dt.date.fromisoformat('2013-06-03'))).select(["date", "barrid", "momentum_alpha"])
    print(get_gamma_from_alphas(alpha, 0.05, dt.date.fromisoformat('2013-06-03')))