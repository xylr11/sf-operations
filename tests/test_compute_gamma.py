import datetime as dt
import numpy as np
import polars as pl
import pytest

from signals import compute_gamma as cg


def _setup_factor_mocks(monkeypatch):
    # Use a small factor set for testing.
    monkeypatch.setattr(cg, "factors", ["f1", "f2"])

    d1 = dt.date(2020, 1, 1)
    d2 = dt.date(2020, 1, 2)

    exposures = pl.DataFrame(
        {
            "date": [d1, d1, d2, d2],
            "barrid": ["B", "A", "A", "B"],  # intentionally unsorted
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [5.0, 6.0, 7.0, 8.0],
        }
    )
    specific_risk = pl.DataFrame(
        {
            "date": [d1, d1, d2, d2],
            "barrid": ["B", "A", "A", "B"],
            "specific_risk": [0.1, 0.2, 0.3, 0.4],
        }
    )
    # Build a covariance that becomes identity after the /1e4 scaling.
    cov = pl.DataFrame({"f1": [1e4, 0.0], "f2": [0.0, 1e4]})

    monkeypatch.setattr(cg.sfd, "load_exposures", lambda *_, **__: exposures)
    monkeypatch.setattr(cg.sfd, "load_assets", lambda *_, **__: specific_risk)
    monkeypatch.setattr(cg, "_construct_factor_covariance_matrix", lambda *_: cov)

    return d1, d2


def test_iter_factor_data_alignment(monkeypatch):
    d1, d2 = _setup_factor_mocks(monkeypatch)

    items = list(cg.iter_factor_data(d1, d2))
    assert [date for date, _ in items] == [d1, d2]

    # First date: barrids sorted A, B
    _, data1 = items[0]
    np.testing.assert_array_equal(data1["barrid"], np.array(["A", "B"]))
    np.testing.assert_allclose(data1["B"], np.array([[2.0, 6.0], [1.0, 5.0]]))
    np.testing.assert_allclose(data1["F"], np.eye(2))
    np.testing.assert_allclose(
        data1["D"], np.square(np.array([0.2, 0.1])) / 1e4
    )

    # Second date: barrids sorted A, B, check shapes
    _, data2 = items[1]
    assert data2["B"].shape == (2, 2)
    assert data2["F"].shape == (2, 2)
    assert data2["D"].shape == (2,)


def test_iter_factor_mvos_alpha_alignment(monkeypatch):
    d1, d2 = _setup_factor_mocks(monkeypatch)

    alpha_df = pl.DataFrame(
        {
            "date": [d1, d1, d2],
            "barrid": ["A", "B", "B"],
            "momentum_alpha": [0.1, 0.2, 0.3],
        }
    )

    mvos = list(cg.iter_factor_mvos(d1, d2, "momentum", alpha_df))
    assert len(mvos) == 2

    # Date 1 has both alphas; check alignment to sorted barrids A, B.
    date1, mvo1 = mvos[0]
    assert date1 == d1
    np.testing.assert_allclose(mvo1.alpha, np.array([0.1, 0.2]))

    # Date 2 only has B alpha; A should be zero-filled.
    date2, mvo2 = mvos[1]
    assert date2 == d2
    np.testing.assert_allclose(mvo2.alpha, np.array([0.0, 0.3]))
