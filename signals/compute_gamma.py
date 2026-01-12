import polars as pl
import numpy as np
import sf_quant.data as sfd
import datetime as dt
from sf_quant.data._factors import factors
from sf_quant.data.covariance_matrix import _construct_factor_covariance_matrix
from time import perf_counter
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import minres
from tqdm import tqdm

def iter_factor_data(start, end):
    """
    Stream factor-model inputs between two dates (inclusive).

    Yields (date, data) tuples where ``data`` is a dict containing:
        - ``B``: factor exposure matrix (n_assets x n_factors)
        - ``F``: factor covariance matrix (n_factors x n_factors)
        - ``D``: specific risk variances (length n_assets)
        - ``barrid``: asset identifiers ordered consistently with ``B`` rows
    """
    exposures = (
        sfd.load_exposures(start, end, True, ["date", "barrid"] + factors)
        .fill_nan(0)
        .fill_null(0)
    )
    specific_risk = (
        sfd.load_assets(start, end, ["date", "barrid", "specific_risk"], in_universe=True)
        .fill_nan(0)
        .fill_null(0)
    )

    dates = sorted(exposures.select("date").unique().to_series().to_list())

    for date in tqdm(dates, desc="Loading factor data"):
        exp_date = exposures.filter(pl.col("date").eq(date)).sort("barrid")
        if exp_date.is_empty():
            continue

        # Align specific risk to exposure ordering
        sr_date = (
            exp_date.select("barrid")
            .join(
                specific_risk.filter(pl.col("date").eq(date)).select(
                    "barrid", "specific_risk"
                ),
                on="barrid",
                how="left",
            )
            .fill_null(0)
            .fill_nan(0)
            .sort("barrid")
        )

        barrids = exp_date.select("barrid").to_series().to_list()
        B = exp_date.select(factors).to_numpy()
        F = (
            _construct_factor_covariance_matrix(date)
            .fill_nan(0)
            .fill_null(0)
            .select(factors)
            .to_numpy()
            / 1e4
        )
        D = np.square(sr_date.select("specific_risk").to_numpy().flatten()) / 1e4

        yield date, {"B": B, "F": F, "D": D, "barrid": np.array(barrids)}

def load_factor_data(start, end):
    """
    Materialize factor-model inputs between two dates (inclusive) into a dict.

    This wraps :func:`iter_factor_data` for callers that want everything in memory.
    """
    return {date: data for date, data in iter_factor_data(start, end)}

def build_signal_factor_inputs(start, end, signal, df):
    """
    Combine signal alphas with factor inputs for each date.

    Parameters
    ----------
    start, end : datetime-like
        Date range to include (inclusive).
    signal : str
        Name of the signal; the alpha column is expected to be ``{signal}_alpha``.
    df : pl.DataFrame or pl.LazyFrame
        Data containing ``date``, ``barrid`` and the signal alpha column.

    Returns
    -------
    dict
        Keys are dates; values are dicts with ``alpha`` (np.ndarray aligned to
        the factor exposure ordering) and the ``B``, ``F``, ``D`` matrices from
        :func:`load_factor_data`.
    """
    alpha_col = f"{signal}_alpha"

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    if alpha_col not in df.columns:
        raise ValueError(f"Column '{alpha_col}' not found in provided dataframe.")

    filtered = (
        df.filter(
            (pl.col("date") >= start)
            & (pl.col("date") <= end)
            & pl.col(alpha_col).is_not_null()
        )
        .select(["date", "barrid", alpha_col])
    )

    alpha_by_date = {}
    for frame in filtered.partition_by("date", maintain_order=True):
        date_value = frame["date"][0]
        alpha_by_date[date_value] = dict(
            zip(frame["barrid"].to_list(), frame[alpha_col].to_list())
        )

    factor_data = load_factor_data(start, end)
    combined = {}

    for date, data in factor_data.items():
        barrids = data.get("barrid")
        if barrids is None:
            raise ValueError("Factor data missing barrid ordering; rerun load_factor_data.")

        alpha_map = alpha_by_date.get(date, {})
        alpha_vec = np.array([alpha_map.get(b, 0.0) for b in barrids])

        combined[date] = {
            "alpha": alpha_vec,
            "B": data["B"],
            "F": data["F"],
            "D": data["D"],
        }

    return combined

def iter_factor_mvos(start, end, signal, df, A=None, b=None, L=None, d=None, d_floor=1e-8):
    """
    Stream per-date FactorMVO instances with alphas aligned to factor exposures.

    Parameters
    ----------
    start, end : datetime-like
        Date range to include (inclusive).
    signal : str
        Signal name; alpha column must be ``{signal}_alpha``.
    df : pl.DataFrame or pl.LazyFrame
        Source data containing alphas.
    A, b, L, d : optional
        Constraint matrices/vectors to pass into :class:`FactorMVO`.
    d_floor : float
        Lower bound for specific risk diagonal entries in the optimizer.

    Yields
    ------
    tuple
        (date, FactorMVO) for each available date.
    """
    alpha_col = f"{signal}_alpha"

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    if alpha_col not in df.columns:
        raise ValueError(f"Column '{alpha_col}' not found in provided dataframe.")

    # Keep only relevant slice of alpha data up front
    alpha_slice = (
        df.filter(
            (pl.col("date") >= start)
            & (pl.col("date") <= end)
            & pl.col(alpha_col).is_not_null()
        )
        .select(["date", "barrid", alpha_col])
    )

    alpha_by_date = {}
    for frame in alpha_slice.partition_by("date", maintain_order=True):
        date_value = frame["date"][0]
        alpha_by_date[date_value] = dict(
            zip(frame["barrid"].to_list(), frame[alpha_col].to_list())
        )

    for date, data in iter_factor_data(start, end):
        barrids = data["barrid"]
        alpha_map = alpha_by_date.get(date, {})
        alpha_vec = np.array([alpha_map.get(b, 0.0) for b in barrids])

        yield date, FactorMVO(alpha_vec, data["B"], data["F"], data["D"], A=A, b=b, L=L, d=d, d_floor=d_floor)

class FactorMVO:
    """
    Factor-model mean-variance optimizer that applies the factor covariance as a LinearOperator
    (no explicit nxn covariance materialization).

    Objective: maximize alpha^T w - (gamma / 2) * w^T Cov w,  Cov = B F B^T + diag(D)
    Constraints: A w = b, optional L w <= d (not implemented here), or target active risk.
    """

    def __init__(self, alpha, B, F, D, A=None, b=None, L=None, d=None, d_floor=1e-8):
        self.alpha = alpha          # n
        self.B = B                  # n × k
        self.F = F                  # k × k
        self.D = np.maximum(D, d_floor)  # n, floor to keep H SPD enough for Krylov
        self.A = A                  # m × n  (equality constraints, e.g. UnitBeta, ZeroBeta, FullInvestment)
        self.b = b                  # m
        self.L = L                  # p × n  (inequality constraints, e.g. LongOnly. Must be less than version, hence L)
        self.d = d                  # p

        self.n = len(alpha)
        self.k = F.shape[0]
    
    def _make_H_operator(self, gamma):
        """Return LinearOperator for Hessian gamma*Cov without forming Cov."""
        n, k = self.B.shape

        def matvec(x):
            # x: length n
            Btx = self.B.T @ x            # R^k
            F_Btx = self.F @ Btx         # R^k
            B_F_Btx = self.B @ F_Btx  # R^n
            return gamma * (B_F_Btx + self.D * x)

        return LinearOperator(
            shape=(n, n),
            matvec=matvec,
            dtype=float
        )

    def _make_KKT_operator(self, gamma):
        """
        KKT matrix:
            [ H   A^T ]
            [ A    0  ]
        implemented as a LinearOperator.
        """
        n = self.B.shape[0]
        m = self.A.shape[0]

        # Build H as a LinearOperator
        H_op = self._make_H_operator(gamma)

        def matvec(z):
            # z = [x, y]
            x = z[:n]
            y = z[n:]

            top = H_op.matvec(x) + self.A.T @ y
            bottom = self.A @ x

            return np.concatenate([top, bottom])

        return LinearOperator(
            shape=(n + m, n + m),
            matvec=matvec,
            dtype=float
        )

    def cov_times(self, w):
        """Compute Cov * w without forming Cov explicitly."""
        Bw = self.B.T @ w              # k
        F_Bw = self.F @ Bw             # k
        return self.B @ F_Bw + self.D * w

    def risk(self, w):
        """Compute sqrt( w^T Cov w )."""
        return np.sqrt(w @ self.cov_times(w))

    def kkt_residuals(self, w, gamma):
        """Return (||A w - b||_2, ||g + A^T lambda||_2) with g = -alpha + gamma*Cov w."""
        g = -self.alpha + gamma * self.cov_times(w)

        if self.A is None:
            primal = 0.0
            dual = np.linalg.norm(g)
            return primal, dual

        primal_vec = self.A @ w - self.b
        # Solve for lambda in (A A^T) lambda = A (-g) to get best dual residual
        ATA = self.A @ self.A.T
        rhs = self.A @ (-g)
        lam, *_ = np.linalg.lstsq(ATA, rhs, rcond=None)
        dual_vec = g + self.A.T @ lam

        return np.linalg.norm(primal_vec), np.linalg.norm(dual_vec)

    def solve(self, gamma, active_risk_target=None, max_iter=50, tol=1e-8, debug=False):
        """
        Solve:
            maximize alpha^T w - (gamma / 2) * w^T Cov w
        with constraints.
        active_risk_target: impose sqrt(w^T Cov w) = active_risk_target via bisection on gamma.
        """

        # If active_risk_target is given, use the _solve_for_risk version
        if active_risk_target is not None:
            return self._solve_for_risk(active_risk_target, gamma_init=gamma, debug=debug)

        # KKT solve for fixed gamma
        return self._solve_fixed_gamma(gamma, max_iter=max_iter, tol=tol, debug=debug)

    def _solve_fixed_gamma(self, gamma, max_iter=500, tol=1e-8, debug=False, w0=None):
        """
        Newton solve for fixed gamma with equality constraints via KKT MINRES.
        """

        if debug:
            start = perf_counter()
            print(f'[INFO] Started optimizer with gamma={gamma}.')

        n = self.n
        w = np.zeros(n) if w0 is None else w0.copy()

        A = self.A
        b = self.b
        m = 0 if A is None else A.shape[0]

        lin_maxiter = max(500, 5 * self.n)

        converged = False
        # Newton loop
        for _ in range(max_iter): # Might want to track and return this

            # Gradient: g = - alpha + gamma Cov w   (negative because maximizing)
            g = -self.alpha + gamma * self.cov_times(w)

            # Hessian operator: H v = gamma Cov v
            # We apply Cov v via cov_times.

            # Build KKT system:
            #
            # [ H   Aᵀ ] [ Δw ] = - [ g ]
            # [ A    0 ] [ Δgamma ]     [ r ]
            #
            # where r = A w - b
            #
            # This is solved with block elimination:
            #
            # Solve  H Δw + A^T Δµ = -g
            #        A Δw         = -(Aw - b) What

            if m == 0:
                H_op = self._make_H_operator(gamma)
                dw, info = minres(H_op, -g, rtol=1e-10, maxiter=lin_maxiter)
                if info != 0:
                    raise RuntimeError(f'minres did not converge for unconstrained solve (info={info}).')
                w += dw

                cur_norm = np.linalg.norm(dw)
                if debug: print(f'[INFO] Current norm is {cur_norm}.')
                if cur_norm < tol:
                    converged = True
                    break

            else:
                K = self._make_KKT_operator(gamma)

                rhs = np.concatenate([-g, -(A @ w - b)])
                sol, info = minres(K, rhs, rtol=1e-10, maxiter=lin_maxiter)
                if info != 0:
                    raise RuntimeError(f'minres did not converge for constrained solve (info={info}).')

                dw = sol[:n]
                w += dw

                cur_norm = np.linalg.norm(dw)
                if debug: print(f'[INFO] Current norm is {cur_norm}.')
                if cur_norm < tol:
                    converged = True
                    break

        if debug:
            end = perf_counter()
            print(f'[INFO] Optimizer took {(end - start):.4g} seconds to finish.')

        if not converged:
            print(f"[WARN] FactorMVO _solve_fixed_gamma did not reach tol={tol}. "
                  f"Final step norm={cur_norm:.3e}, iter={max_iter}, gamma={gamma}.")

        return w

    def _solve_for_risk(self, target, gamma_init, tol=1e-8, debug=False):
        """
        Approximate sqrt(w^T Cov w) = target by bisection on gamma.
        """
        gamma_low = 1e-1
        gamma_high = 1e4
        gamma = gamma_init

        if debug:
            start = perf_counter()
            print(f'[INFO] Started optimizer with target active risk {target:.4g}.')

        w_init = np.zeros(self.n)
        reached = False
        for i in range(40):
            w = self._solve_fixed_gamma(gamma, w0=w_init)
            r = self.risk(w)
            w_init = w  # warm start the next solve

            if np.abs(r - target) < tol:
                reached = True
                break
            elif r < target:
                gamma_high = gamma
            else:
                gamma_low = gamma

            gamma = 0.5 * (gamma_low + gamma_high)

            if debug: print(f'[INFO] Finished iteration {i} with risk {r}.')

        if debug:
            end = perf_counter()
            print(f'[INFO] Optimizer with arget risk tuning took {(end - start):.4g} seconds to finish.')

        if not reached:
            print(f"[WARN] FactorMVO _solve_for_risk did not reach target risk within tol={tol}. "
                  f"Final risk={r:.3e}, target={target}, last gamma={gamma}.")

        return self._solve_fixed_gamma(gamma, w0=w_init)
    
if __name__ == "__main__":
    # Benchmark using real data + add_signals-derived alphas.
    from add_signals import add_signals

    signal = "momentum"
    start_date = dt.date(2005, 1, 1)
    end_date = dt.date(2005, 3, 31)
    gamma = 3.0

    print(f"[BENCHMARK] Loading base data from data/russell_3000_daily.parquet...")
    raw = pl.read_parquet("data/russell_3000_daily.parquet")
    df_with_signals = add_signals(raw)

    print(f"[BENCHMARK] Computing MVO for {signal} from {start_date} to {end_date}...")
    total_start = perf_counter()

    per_date_times = []
    solved = 0
    for date, mvo in iter_factor_mvos(start_date, end_date, signal, df_with_signals):
        date_start = perf_counter()
        _ = mvo.solve(gamma, max_iter=80, tol=1e-8)
        per_date_times.append(perf_counter() - date_start)
        solved += 1

    total_elapsed = perf_counter() - total_start

    if solved == 0:
        print("[BENCHMARK] No dates solved; check data coverage for the chosen window/signal.")
    else:
        print(f"[BENCHMARK] Solved {solved} dates in {total_elapsed:.3f}s "
              f"(avg {np.mean(per_date_times):.3f}s/date, median {np.median(per_date_times):.3f}s/date).")
