import typing
from functools import cache, cached_property

import numpy as np
import numpy.typing as npt

from portfolioqtopt.dwave_solver import SolverTypes, solve_dwave_advantage_cubo
from portfolioqtopt.qubo import Qubo, get_qubo_dict
from portfolioqtopt.symmetric_to_triangular import get_upper_triangular


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
        return typing.cast(npt.NDArray[np.floating[typing.Any]], asset_prices)

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

    @cache
    def get_qubo(self, theta1: float, theta2: float, theta3: float) -> Qubo:
        """Compute the qubo matrix and it's corresponding dictionary.

        Args:
            theta1 (float): First Lagrange multiplier.
            theta2 (float): Second Lagrange multiplier
            theta3 (float): Third Lagrange multiplier

        Returns:
            Qubo: A dataclass that have the qubo matrix and the qubo index dictionary
                as attributes.
        """
        # Obtenemos los valores asociados al riesgo, es decir, la covariance
        qubo_covariance = np.cov(self.npp.T)  # (p, p)

        # ----- SHAPING THE VALUES OF THE QUBO

        # We generate a diagonal matrix with the returns, this matrix will be used later
        # with the value of theta1.
        qubo_returns = np.diag(self.expected_returns)  # (p, p)

        # We generate a diagonal matrix with the possible prices * 2. This will be
        # related to the returns.
        qubo_prices_linear = 2.0 * self.b * np.diag(self.npp_last)  # (p, p)

        # We generate a symmetric matrix also related to the possible prices. This will
        # be related to diversity.
        qubo_prices_quadratic = np.outer(self.npp_last, self.npp_last)  # (p, p)

        # ----- Final QUBO formation, with bias and penalty values included

        # We form the diagonal values, related to return and prices.
        qi = -theta1 * qubo_returns - theta2 * qubo_prices_linear  # (p, p)

        # We now form the quadratic values, related to diversity.
        qij = theta2 * qubo_prices_quadratic + theta3 * qubo_covariance  # (p, p)

        qubo = typing.cast(npt.NDArray[np.floating[typing.Any]], qi + qij)
        qubo_matrix = get_upper_triangular(qubo)
        qubo_dict = get_qubo_dict(qubo_matrix)
        return Qubo(qubo_matrix, qubo_dict)

    def solve(
        self,
        theta1: float,
        theta2: float,
        theta3: float,
        token: str,
        solver: SolverTypes,
    ) -> npt.NDArray[np.int8]:
        qubo = self.get_qubo(theta1, theta2, theta3)
        sampleset = solve_dwave_advantage_cubo(qubo, solver, token)
        qubits: npt.NDArray[np.int8] = sampleset.record.sample
        return qubits


# https://stackoverflow.com/questions/69178071/cached-property-doctest-is-not-detected
__test__ = {
    "Selection.pnn": Selection.npp,
    "Selection.expected_returns": Selection.expected_returns,
    "Selection.npp_last": Selection.npp_last,
}


if __name__ == "__main__":
    prices = np.array(
        [
            [100, 104, 102, 104, 100],
            [10, 10.2, 10.4, 10.5, 10.4],
            [50, 51, 52, 52.5, 52],
            [1.0, 1.02, 1.04, 1.05, 1.04],
        ],
        dtype=np.floating,
    ).T
    selection = Selection(prices, 6, 1.0)
    print(f"{selection.granularity=}")
    print(f"{selection.npp=}")
    print(f"{selection.npp_last=}")
    print(f"{selection.expected_returns=}")
    print(f"{selection.get_qubo(0.3, 0.2, 0.2)=}")
    # print(f"{selection.solve(0.3, 0.2, 0.2, 'aa', SolverTypes.hybrid_solver)=}")

    # How to simulate def Selection.solve output?
    runs = 10

    from portfolioqtopt.interpreter import get_selected_funds_indexes

    w, b = 6, 1.0
    selection = Selection(prices, w, b)
    for i in range(runs):
        qbits = selection.solve(0.3, 0.2, 0.2, "aa", SolverTypes.hybrid_solver)
        indexes = get_selected_funds_indexes(qbits, w)
