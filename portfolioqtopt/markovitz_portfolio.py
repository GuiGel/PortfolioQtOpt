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
        return get_partitions(self.w)

    def _get_expand_prices(
        self, reversed: bool = False
    ) -> npt.NDArray[np.floating[typing.Any]]:
        """Optimized version of get_expand_prices.

        Examples:

            >>> prices = np.array([[100, 50, 10, 5], [10, 5, 1, 0.5]]).T
            >>> selection = Selection(prices, 3, 1)
            >>> selection._get_expand_prices()  # doctest: +NORMALIZE_WHITESPACE
            array([[20.  , 10.  ,  5.  , 20.  , 10.  ,  5.  ],
                [10.  ,  5.  ,  2.5 , 10.  ,  5.  ,  2.5 ],
                [ 2.  ,  1.  ,  0.5 ,  2.  ,  1.  ,  0.5 ],
                [ 1.  ,  0.5 ,  0.25,  1.  ,  0.5 ,  0.25]])

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
        return asset_prices.astype(np.float64)

    @cached_property
    def ppn(self) -> npt.NDArray[np.floating[typing.Any]]:
        """Compute the each partitions of the normalize prices.

        Normalize all au,l by au,Nf to keep all asset prices to a similar range.

        Examples:

            >>> prices = np.array([[100, 50, 10, 5], [10, 5, 1, 0.5]]).T
            >>> selection = Selection(prices, 3, 1)
            >>> selection.ppn  # doctest: +NORMALIZE_WHITESPACE
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
    def ppn_rev(self) -> npt.NDArray[np.floating[typing.Any]]:
        return self._get_expand_prices(reversed=True)

    @cached_property
    def ppn_last(self) -> npt.NDArray[np.floating[typing.Any]]:
        return self.ppn[-1, :]  # (p, )

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
        return expected_returns


# https://stackoverflow.com/questions/69178071/cached-property-doctest-is-not-detected
__test__ = {
    "Selection.pnn": Selection.ppn,
    "Selection.expected_returns": Selection.expected_returns,
}

if __name__ == "__main__":
    prices = np.array(
        [
            [100, 104, 102, 104, 100],
            [10, 10.2, 10.4, 10.5, 10.4],
            [50, 51, 52, 52.5, 52],
            [1.0, 1.02, 1.04, 1.05, 1.04],
        ]
    ).T
    selection = Selection(prices, 6, 1.0)
    print(f"{selection.granularity=}")
    print(f"{selection.ppn=}")
    print(f"{selection.ppn_last=}")
    print(f"{selection.expected_returns=}")
