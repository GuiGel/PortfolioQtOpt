"""This module contains tests for the Expand_Prices module."""
import numpy as np
import numpy.typing as npt
import pytest

from portfolioqtopt.Expand_Prices import get_expand_prices_opt


@pytest.mark.parametrize(
    "prices, slices_list, budget, expected_expand_prices, reversed",
    [
        (
            np.array([[100, 12, 10], [50, 30, 5], [10, 60, 1]]),
            np.array([1, 0.5]),
            1,
            np.array(
                [
                    [10.0, 5.0, 0.2, 0.1, 10.0, 5.0],
                    [5.0, 2.5, 0.5, 0.25, 5.0, 2.5],
                    [1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
                ]
            ),
            False,
        ),
        (
            np.array([[100, 12, 10], [50, 30, 5], [10, 60, 1]]),
            np.array([1, 0.5]),
            1,
            np.array(
                [
                    [1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
                    [0.5, 0.25, 2.5, 1.25, 0.5, 0.25],
                    [0.1, 0.05, 5.0, 2.5, 0.1, 0.05],
                ]
            ),
            True,
        ),
    ],
)
def test_get_expand_prices_opt(
    prices: npt.NDArray[np.float64],
    slices_list: npt.NDArray[np.float64],
    budget: int,
    expected_expand_prices: npt.NDArray[np.float64],
    reversed: bool,
) -> None:
    expand_prices = get_expand_prices_opt(
        prices, slices_list, budget, reversed=reversed
    )
    np.testing.assert_equal(expand_prices, expected_expand_prices)
