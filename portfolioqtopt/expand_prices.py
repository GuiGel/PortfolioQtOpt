"""Create columns of historical price data representing various percentages of the
budget.

For example, if the budget is 20 and the price of a fund is 100, you could analyze
various percentages of the fund based on the budget: 5, 10, 15 and 20 to find the best
option.

NOTE:: This of course increases the search space.
"""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class ExpandPrices:
    data: npt.NDArray[np.float64]  # (m, p)
    reversed_data: npt.NDArray[np.float64]  # (m, p)

    @property
    def last(self) -> npt.NDArray[np.float64]:
        return self.data[-1, :]  # (p, )


def get_slices_list(slices: int) -> npt.NDArray[np.float64]:
    """Compute the possible proportions of the budget that we can allocate to each fund.

    Example:

    >>> get_slices_list(5)
    array([1.    , 0.5   , 0.25  , 0.125 , 0.0625])

    Args:
        slices (int): The number of slices is the granularity that we are
            going to give to each fund. That is, the amount of the budget we
            will be able to invest.

    Returns:
        npt.NDArray[np.float64]: List of slices values.
    """
    return np.power(0.5, np.arange(slices))


def get_expand_prices_opt(
    prices: npt.NDArray[np.float64],
    slices_list: npt.NDArray[np.float64],
    budget: float = 1.0,
    reversed: bool = False,
) -> npt.NDArray[np.float64]:
    """Optimized version of get_expand_prices.
    Speedup of 50X with the original ``get_expand_prices`` code.

    Examples:

        >>> prices = np.array([[100, 50, 10, 5], [10, 5, 1, 0.5]]).T
        >>> get_expand_prices_opt(prices, [1, 0.5, 0.25], 1)  # doctest: +NORMALIZE_WHITESPACE
        array([[20.  , 10.  ,  5.  , 20.  , 10.  ,  5.  ],
               [10.  ,  5.  ,  2.5 , 10.  ,  5.  ,  2.5 ],
               [ 2.  ,  1.  ,  0.5 ,  2.  ,  1.  ,  0.5 ],
               [ 1.  ,  0.5 ,  0.25,  1.  ,  0.5 ,  0.25]])

    Args:
        prices (npt.NDArray[np.float64]): The fund prices with shape
            (prices number, funds number).
        slices_list (npt.NDArray[np.float64]): Granularity slice list.
        budget (int, optional): The initial budget. Defaults to 1.

    Returns:
        npt.NDArray[np.float64]: The expanded prices.
    """
    # norm_price_factor is the value we will use to normalize the purchase values of each
    # asset

    # TODO ensure that prices values must be > 0 and not np.Nan

    factor = prices[-1, :]
    if reversed:
        factor = prices[0, :]

    norm_price_factor = np.divide(budget, factor, dtype=np.float64, casting="unsafe")
    all_assert_prices = (
        np.expand_dims(prices, axis=2) * slices_list * norm_price_factor.reshape(-1, 1)
    )
    _, num_cols, num_slices = all_assert_prices.shape
    asset_prices = all_assert_prices.reshape(-1, num_cols * num_slices)
    return asset_prices.astype(np.float64)


def get_expand_prices(
    prices: npt.NDArray[np.float64],
    slices: int,
    budget: float = 1.0,
) -> ExpandPrices:
    """Expand the prices with the slices and normalized them.

    Example:

        >>> slices = 3
        >>> budget = 1.0
        >>> prices = np.array([[100, 50, 10], [10, 5, 1]]).T
        >>> prices.shape
        (3, 2)
        >>> get_expand_prices(prices, budget, slices)  # doctest: +NORMALIZE_WHITESPACE
        ExpandPrices(data=array([[30., 30.], [15., 15.], [ 3.,  3.]]),
        reversed_data=array([[3. , 3. ], [1.5, 1.5], [0.3, 0.3]]))

    Args:
        prices (npt.NDArray[np.float64]): The fund prices. Shape (m, n)
            where m is the prices number and n the funds number.
        slices (int): The number of slices is the granularity that we are
            going to give to each fund. That is, the amount of the budget we
            will be able to invest.
        budget (int, optional): The initial budget. Defaults to 1.

    Returns:
        ExpandPrices: A ExpandPrices ``dataclass``.
    """
    slices_list = get_slices_list(slices)

    data = get_expand_prices_opt(prices, slices_list, budget)

    reversed_data = get_expand_prices_opt(
        prices,
        slices_list,
        budget,
        reversed=True,
    )

    return ExpandPrices(data, reversed_data)
