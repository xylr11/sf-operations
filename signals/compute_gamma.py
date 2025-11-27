import polars as pl
from numpy.linalg import pinv
import numpy as np
import sf_quant.data as sfd
import datetime as dt
from sf_quant.data._factors import factors
from sf_quant.data.covariance_matrix import _construct_factor_covariance_matrix
import sf_quant.backtester as sfb
from time import perf_counter
from add_signals import add_signals
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import minres
from scipy.sparse.linalg import cg
import get_signal_weights
import sf_quant.optimizer as sfo

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

class FactorMVO:
    """
    Factor-model mean-variance optimizer without inverting anything and hopefully without computing cov.

    Objective: maximize   alpha^T w  - gamma * w^T Cov w
    Cov = B F B^T + diag(D)

    Constraints:
        A w = b
        L w <= d (this is element-wise)
        sqrt(w^T Cov w) <= active_risk_target (same here)
    """

    def __init__(self, alpha, B, F, D, A=None, b=None, L=None, d=None):
        self.alpha = alpha          # n
        self.B = B                  # n × k
        self.F = F                  # k × k
        self.D = D                  # n
        self.A = A                  # m × n  (equality constraints, e.g. UnitBeta, ZeroBeta, FullInvestment)
        self.b = b                  # m
        self.L = L                  # p × n  (inequality constraints, e.g. LongOnly. Must be less than version, hence L)
        self.d = d                  # p

        self.n = len(alpha)
        self.k = F.shape[0]
    
    def _make_H_operator(self, gamma):
        n, k = self.B.shape

        def matvec(x):
            # x: length n
            Btx = self.B.T @ x            # R^k
            F_Btx = self.F @ Btx         # R^k
            B_F_Btx = self.B @ F_Btx  # R^n
            return 2 * gamma * (B_F_Btx + self.D * x)

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
        """Compute Cov * w without forming Cov explicitly.
        Some scratch work shows this is about 10x faster than the naive method with n~3000, k~300, more generally n/k times speedup
        """
        Bw = self.B.T @ w              # k
        F_Bw = self.F @ Bw             # k
        return self.B @ F_Bw + self.D * w

    def risk(self, w):
        """Compute sqrt( w^T Cov w )."""
        return np.sqrt(w @ self.cov_times(w))

    def solve(self, gamma, active_risk_target=None, max_iter=50, tol=1e-8, debug=False):
        """
        Solve:
            maximize alpha^T w - gamma * w^T Cov w
        with constraints (these will need to be made somewhat accessible, not sure about what the best form is).
        active_risk_target: impose sqrt(w^T Cov w) = active_risk_target via bisection on gamma.
        """

        # If active_risk_target is given, use the _solve_for_risk version
        if active_risk_target is not None:
            return self._solve_for_risk(active_risk_target, gamma_init=gamma, debug=debug)

        # KKT solve for fixed gamma
        return self._solve_fixed_gamma(gamma, max_iter=max_iter, tol=tol, debug=debug)

    def _solve_fixed_gamma(self, gamma, max_iter=50, tol=1e-8, debug=False):
        """
        Solve with KKT I think:
            maximize alpha^T w - gamma * w^T Cov w
            A w = b      
            L w <= d     (handled via projected Newton; probably there is a better method I don't know about)
        """

        if debug:
            start = perf_counter()
            print(f'[INFO] Started optimizer with gamma={gamma}.')

        n = self.n
        w = np.zeros(n)

        A = self.A
        b = self.b
        m = 0 if A is None else A.shape[0]

        # Newton loop
        for _ in range(max_iter): # Might want to track and return this

            # Gradient: g = - alpha + 2 gamma Cov w   (negative because maximizing)
            g = -self.alpha + 2 * gamma * self.cov_times(w)

            # Hessian operator: H v = 2 gamma Cov v
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
                dw, _ = cg(H_op, -g, rtol=1e-10)
                w += dw

                cur_norm = np.linalg.norm(dw)
                if debug: print(f'[INFO] Current norm is {cur_norm}.')
                if cur_norm < tol:
                    break

            else:
                K = self._make_KKT_operator(gamma)

                rhs = np.concatenate([-g, -(A @ w - b)])
                sol, _ = minres(K, rhs, rtol=1e-8)

                dw = sol[:n]

                dw = sol[:self.n]
                w += dw

                cur_norm = np.linalg.norm(dw)
                if debug: print(f'[INFO] Current norm is {cur_norm}.')
                if cur_norm < tol:
                    break

        if debug:
            end = perf_counter()
            print(f'[INFO] Optimizer took {(end - start):.4g} seconds to finish.')

        return w

    def _solve_for_risk(self, target, gamma_init, tol= 1e-8, debug=False):
        """
        Approx sqrt(w^T Cov w) = target by bisection.
        """
        gamma_low = 1e-1
        gamma_high = 1e4
        gamma = gamma_init

        if debug:
            start = perf_counter()
            print(f'[INFO] Started optimizer with target active risk {target:.4g}.')

        for i in range(40):
            w = self._solve_fixed_gamma(gamma)
            r = self.risk(w)

            if np.abs(r - target) < tol:
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

        return self._solve_fixed_gamma(gamma)
    
def target_active_risk_weights(alpha, date):
    beta = sfd.load_assets_by_date(date, True, ["date", "barrid", "predicted_beta"]).sort("barrid").select("predicted_beta").to_numpy() / 1e2
    A = np.hstack([np.ones_like(beta), beta]).T    
    factor_exposures = sfd.load_exposures(date, date, True, ["date", "barrid"] + factors)
    specific_risk = sfd.load_assets_by_date(date, True, ["date", "barrid", "specific_risk"]) # The sfd load functions have their kwargs out of order :skull:
    D = specific_risk.sort("barrid").fill_nan(0).fill_null(0).select('specific_risk').to_numpy().flatten() / 1e4
    B = factor_exposures.sort("barrid").fill_nan(0).fill_null(0).select(factors).to_numpy()
    F = _construct_factor_covariance_matrix(date).fill_nan(0).fill_null(0).select(factors).to_numpy() / 1e4
    b = np.zeros((2))

    solver = FactorMVO(alpha, B, F, D, A, b)
    return solver.solve(gamma=100, active_risk_target=None, debug=True)

if __name__ == "__main__":
    df = pl.read_parquet('../data/russell_3000_daily.parquet').sort('barrid', 'date')
    signals = add_signals(df)

    start = perf_counter()
    alpha = signals.filter(pl.col("date").eq(dt.date.fromisoformat('2013-06-03')))["momentum_alpha"].fill_nan(0).fill_null(0).to_numpy().flatten()
    print(target_active_risk_weights(alpha, dt.date.fromisoformat('2013-06-03')))
    end = perf_counter()
    print(f'Total time for mine: {end - start} seconds')

    start = perf_counter()
    print(get_signal_weights.get_signal_weights(signals.lazy().filter(pl.col('date').eq(dt.date.fromisoformat('2013-06-03'))), 'momentum', dt.date.fromisoformat('2013-06-03'), dt.date.fromisoformat('2013-06-03'), gamma=50, constraints=[sfo.ZeroBeta(), get_signal_weights.NetZeroInvestment()]))
    end = perf_counter()
    print(f'Total time for old: {end - start} seconds')
    