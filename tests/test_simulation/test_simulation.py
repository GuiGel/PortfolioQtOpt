"""This module test that the `Simulation` callable is working as expected"""
import numpy as np
import pandas as pd
from pytest import fixture

from portfolioqtopt.simulation.simulation import Simulation
from portfolioqtopt.assets import Assets


@fixture
def assets() -> Assets:
    prices = np.array(
        [
            [100, 101.5, 103.2, 102.6, 101.1],
            [10, 10.2, 10.4, 10.5, 10.4],
            [50, 51.1, 52.2, 52.5, 52.6],
            # [1.0, 1.02, 1.01, 1.05, 1.03],
        ],
        dtype=np.float64,
    ).T
    df = pd.DataFrame(prices, columns=["a", "b", "c"])
    return Assets(df=df)


@fixture
def simulation(assets) -> Simulation:
    return Simulation(assets, {"b": 0.3, "c": 0.15, "a": 0.1}, 10)


class TestSimulation:

    def test_get_random_daily_returns(self, simulation: Simulation) -> None:

        expected_covariance = np.diag(np.ones(3))

        random_daily_returns = simulation._get_random_unit_cov()
        obtained_covariance = np.cov(random_daily_returns)

        np.testing.assert_almost_equal(obtained_covariance, expected_covariance)

    def test_correlate(self, simulation: Simulation) -> None:
        daily_returns = simulation.correlate()

        expected_covariance = simulation.assets.cov
        obtained_covariance = np.cov(daily_returns)

        np.testing.assert_almost_equal(obtained_covariance, expected_covariance)

    def test_er(self, simulation: Simulation) -> None:
        expected_er = np.array([0.1, 0.3, 0.15])
        np.testing.assert_equal(simulation.er, expected_er)

    def test_log_taylor_series(self) -> None:
        pass
