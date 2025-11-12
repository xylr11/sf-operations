import polars as pl
import sf_quant.data as sfd
import datetime as dt
from sf_quant.data._factors import factors
from sf_quant.data.covariance_matrix import _construct_factor_covariance_matrix

def load_factor_covariances(start, end):
    pass

def get_inv_cov_mat(start, end):
    print(factors)
    factor_exposures = sfd.load_exposures(start, end, True, ["date", "barrid"] + factors)
    factors_covariances = None
    specific_risk = sfd.load_assets(start, end, True, ["date", "barrid", "specific_risk"])
    dates = factor_exposures["date"].unique().sort().to_list()
    for date in dates:
        B = factor_exposures.filter(pl.col("date").eq(date)).sort("barrid").fill_nan(0).fill_null(0).select(factors).to_numpy()
        print(B)

        break

if __name__ == "__main__":
    print(get_inv_cov_mat(dt.date.fromisoformat('2013-06-01'), dt.date.fromisoformat('2013-06-30')))