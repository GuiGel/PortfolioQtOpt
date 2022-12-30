"""This module calculates the expected return (based on estimates from Modern Portfolio
Theory) from historical price data.

TODO: This code seems to be more correct than the one in the module ``ExpectedReturn_calculator`` but need a verification by Tecnalia.
"""
import typing

import numpy as np
import numpy.typing as npt

from portfolioqtopt.expand_prices import get_slices_list


def get_granular_mean_daily_returns(
    prices: npt.NDArray[np.floating[typing.Any]], slices_num: int
) -> npt.NDArray[np.floating[typing.Any]]:
    """Obtain the granular mean daily returns of each funds.

    Example:

        >>> prices = np.array([[100, 50, 10, 5, 1], [10, 5, 1, 0.5, 0.1]]).T
        >>> slices_num = 3
        >>> get_granular_mean_daily_returns(prices, slices_num)  # doctest: +NORMALIZE_WHITESPACE
        array([-0.65 , -0.325 , -0.1625, -0.65 , -0.325 , -0.1625])

    Args:
        prices (npt.NDArray[np.floating[typing.Any]]):  The fund prices. Shape (m, n)
            where m is the prices number and n the funds number.
        slices_num (int): The number of slices is the granularity that we are
            going to give to each fund. That is, the amount of the budget we
            will be able to invest.

    Returns:
        npt.NDArray[np.floating[typing.Any]]: Granular mean daily returns.
    """
    _, n = prices.shape  # n is the number of funds
    p = n * slices_num
    daily_returns = ((prices[1:] - prices[:-1]) / prices[:-1]).mean(axis=0)
    pw = get_slices_list(slices_num)
    granular_daily_returns = (daily_returns[:, np.newaxis] * pw[np.newaxis, :]).reshape(
        -1, p
    )
    expected_returns_ = granular_daily_returns.mean(axis=0)
    return expected_returns_
