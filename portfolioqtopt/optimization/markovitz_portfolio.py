"""Preliminary calculations.

In this module, we calculate all preliminary results that depend only on prices,
granularity depth and budget.
"""
from __future__ import annotations

import typing
from functools import cached_property

import numpy as np
import numpy.typing as npt


def get_partitions(w: int) -> npt.NDArray[np.floating[typing.Any]]:
    """Compute the possible proportions of the budget that we can allocate to each fund.

    Example:

        >>> get_partitions(5)
        array([1.    , 0.5   , 0.25  , 0.125 , 0.0625])

    Args:
        w (int): The partitions number that determine the granularity that we are
            going to give to each fund. That is, the amount of the budget we
            will be able to invest.

    Returns:
        npt.NDArray[np.floating[typing.Any]]: List of fraction values.
    """
    return np.power(0.5, np.arange(w))


class Selection:
    """Class with various attributes that help in the
    :class:`portfolioqtopt.optimization.qubo.Qubo` creation.

    Args:
        prices (npt.NDArray[np.floating[typing.Any]]): The stock prices.
        w (int): The granularity depth. The partitions number that determine the
            granularity that we are going to give to each fund. That is, the amount of
            the budget we will be able to invest.
        budget (float): The initial budget.

    Attributes:
        prices (str): The stock prices.
        w (int): The granularity depth.
        b (float): The allocated budget.
        m (int): The number of stocks.
        p (int): The product :math:`w*p`
    """

    def __init__(
        self, prices: npt.NDArray[np.floating[typing.Any]], w: int, budget: float
    ) -> None:
        self.prices = prices  # (n, m)
        self.w = w
        self.b = budget
        _, self.m = prices.shape  # The number of assets
        self.p = self.m * w

    @cached_property
    def granularity(self) -> npt.NDArray[np.floating[typing.Any]]:
        """Compute the possible proportions of the budget :math:`w_i` that we can
        allocate to each fund defined as
        :math:`\\forall i \\in [0, w-1], w_{i} = \\frac{1}{2^i}`

        Returns:
            npt.NDArray[np.floating[typing.Any]]: List of fraction values.
        """
        return get_partitions(self.w)

    def _get_expand_prices(
        self, reversed: bool = False
    ) -> npt.NDArray[np.floating[typing.Any]]:
        """Optimized version of get_expand_prices.

        Examples:

            >>> prices = np.array([[100, 50, 10, 5], [10, 5, 1, 0.5]]).T
            >>> selection = Selection(prices, 3, 1)
            >>> selection._get_expand_prices()
            array([[20.  , 10.  ,  5.  , 20.  , 10.  ,  5.  ],
                   [10.  ,  5.  ,  2.5 , 10.  ,  5.  ,  2.5 ],
                   [ 2.  ,  1.  ,  0.5 ,  2.  ,  1.  ,  0.5 ],
                   [ 1.  ,  0.5 ,  0.25,  1.  ,  0.5 ,  0.25]])
            >>> selection._get_expand_prices(reversed=True)
            array([[1.    , 0.5   , 0.25  , 1.    , 0.5   , 0.25  ],
                   [0.5   , 0.25  , 0.125 , 0.5   , 0.25  , 0.125 ],
                   [0.1   , 0.05  , 0.025 , 0.1   , 0.05  , 0.025 ],
                   [0.05  , 0.025 , 0.0125, 0.05  , 0.025 , 0.0125]])

        Args:
            reversed (bool, optional): Normalized each asset prices by it's beginning
                price or if it's False by it's first price. Defaults to False.

        Returns:
            npt.NDArray[np.floating[typing.Any]]: The partitions of the normalized prices.
        """
        factor = self.prices[-1, :]
        if reversed:
            factor = self.prices[0, :]

        normalized_prices = np.divide(
            self.b, factor, dtype=np.float64, casting="unsafe"
        )
        all_assert_prices = (
            self.prices[:, :, np.newaxis]
            * self.granularity
            * normalized_prices.reshape(-1, 1)
        )
        asset_prices = all_assert_prices.reshape(-1, self.m * self.w)
        return typing.cast(npt.NDArray[np.floating[typing.Any]], asset_prices)

    def __getitem__(self, key: typing.Any) -> Selection:
        """Get new Selection object from self.prices[:, key]

        Example:

        >>> selection = Selection(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).T, 6, 1)
        >>> new_selection = selection[1, -1]
        >>> new_selection.prices
        array([[4, 7],
               [5, 8],
               [6, 9]])
        >>> new_selection.npp.shape
        (3, 12)

        Args:
            key (typing.Any): The selected index.

        Returns:
            Selection: A new Selection instance.
        """
        new_prices = self.prices[:, key]
        return Selection(new_prices, self.w, self.b)

    @cached_property
    def npp(self) -> npt.NDArray[np.floating[typing.Any]]:
        """Compute each partitions of the normalize prices.

        Normalize all au,l by au,Nf to keep all asset prices to a similar range. The
        acronym n.p.p stands for normalized price partitions.

        Examples:

            >>> prices = np.array([[100, 50, 10, 5], [10, 5, 1, 0.5]]).T
            >>> selection = Selection(prices, 3, 1)
            >>> selection.npp  # doctest: +NORMALIZE_WHITESPACE
            array([[20.  , 10.  ,  5.  , 20.  , 10.  ,  5.  ],
                [10.  ,  5.  ,  2.5 , 10.  ,  5.  ,  2.5 ],
                [ 2.  ,  1.  ,  0.5 ,  2.  ,  1.  ,  0.5 ],
                [ 1.  ,  0.5 ,  0.25,  1.  ,  0.5 ,  0.25]])

        Returns:
            npt.NDArray[np.floating[typing.Any]]:  The partitions of the normalized
                prices. shape (p, )
        """
        return self._get_expand_prices(reversed=False)

    @cached_property
    def npp_rev(self) -> npt.NDArray[np.floating[typing.Any]]:
        return self._get_expand_prices(reversed=True)

    @cached_property
    def npp_last(self) -> npt.NDArray[np.floating[typing.Any]]:
        """Get the partitions of the last normalized prices for each assets.

        Example:

            >>> prices = np.array([[100, 50, 10, 5], [10, 5, 1, 0.5]]).T
            >>> selection = Selection(prices, 6, 1)
            >>> selection.npp_last  # doctest: +NORMALIZE_WHITESPACE
            array([1. , 0.5 , 0.25 , 0.125 , 0.0625 , 0.03125,
                   1. , 0.5 , 0.25 , 0.125 , 0.0625 , 0.03125])

        Returns:
            npt.NDArray[np.floating[typing.Any]]: Partitions of the last normalized
                prices.
        """
        return self.npp[-1, :]  # (p, )

    @cached_property
    def expected_returns(self) -> npt.NDArray[np.floating[typing.Any]]:
        """Obtain the partition of the mean daily returns of each funds.

        Example:

            >>> prices = np.array([[100, 50, 10, 5, 1], [10, 5, 1, 0.5, 0.1]]).T
            >>> w, b = 3, 1.0
            >>> selection = Selection(prices, w, b)
            >>> selection.expected_returns  # doctest: +NORMALIZE_WHITESPACE
            array([-0.65 , -0.325 , -0.1625, -0.65 , -0.325 , -0.1625])

        Returns:
            npt.NDArray[np.floating[typing.Any]]: Granular mean daily returns.
        """
        daily_returns = ((self.prices[1:] - self.prices[:-1]) / self.prices[:-1]).mean(
            axis=0
        )
        granular_daily_returns = (
            daily_returns[:, np.newaxis] * self.granularity[np.newaxis, :]
        ).reshape(-1, self.p)
        expected_returns = granular_daily_returns.mean(axis=0)
        return typing.cast(npt.NDArray[np.floating[typing.Any]], expected_returns)


# https://stackoverflow.com/questions/69178071/cached-property-doctest-is-not-detected
__test__ = {
    "Selection.pnn": Selection.npp,
    "Selection.expected_returns": Selection.expected_returns,
    "Selection.npp_last": Selection.npp_last,
}

if __name__ == "__main__":
    prices = np.array([[2, 4, 8, 16], [2, 4, 8, 16]]).T  # , [10, 5, 1, 0.5, 0.1]]).T
    w, b = 3, 1.0
    selection = Selection(prices, w, b)
    print(selection.npp)
    print(selection.npp_rev)
    print(selection.expected_returns)
