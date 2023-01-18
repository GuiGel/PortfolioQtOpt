import numpy as np
import pandas as pd
from pytest import fixture

from portfolioqtopt.simulation.simulation import Simulation
from portfolioqtopt.simulation.stocks import Stocks


@fixture
def stocks() -> Stocks:
    prices = np.array(
        [
            [100, 101.5, 103.2, 102.6, 101.1],
            [10, 10.2, 10.4, 10.5, 10.4],
            [50, 51.1, 52.2, 52.5, 52.6],
            # [1.0, 1.02, 1.01, 1.05, 1.03],
        ],
        dtype=np.float64,
    ).T
    df = pd.DataFrame(prices, columns="a b c".split())
    stocks = Stocks(df=df)
    return stocks


class TestSimulation:
    def test_get_random_daily_returns(self, stocks) -> None:

        simulation = Simulation(stocks, {"a": 0.1, "b": 0.3, "c": 0.1}, 10)
        expected_covariance = np.diag(np.ones(3))

        random_daily_returns = simulation._get_random_daily_returns()
        obtained_covariance = np.cov(random_daily_returns)

        np.testing.assert_almost_equal(obtained_covariance, expected_covariance)
