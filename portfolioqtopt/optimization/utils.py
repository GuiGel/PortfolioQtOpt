"""Preliminary calculations.

In this module, we calculate all preliminary results that depend only on prices,
granularity depth and budget.
"""
from __future__ import annotations

import typing
from functools import cache

import numpy as np
import numpy.typing as npt
from dimod.typing import Bias, Variable

Array = npt.NDArray[np.floating[typing.Any]]

Q = typing.Mapping[typing.Tuple[Variable, Variable], Bias]


@cache
def get_partitions_granularity(w: int) -> Array:
    """Compute the possible proportions of the budget that we can allocate to each fund.

    For :math:`k` in :math:`[1, w]`,  :math:`p_{k}=1/2^{k-1}`.  

    Example:

        >>> get_partitions(5)
        array([1.    , 0.5   , 0.25  , 0.125 , 0.0625])

    Args:
        w (int): The partitions number :math:`w` that determine the precision of the \
granularity partition.
    Returns:
        npt.NDArray[np.floating[typing.Any]]: The List of fraction values :math:`p_{w}`.
    """
    return np.power(0.5, np.arange(w))


def get_partitions_granularity_broadcast(
    partitions_granularity: Array, m: int
) -> Array:
    # (p10, .., p1w-1, ..., pm0 ... pmw-1)
    broadcast_array = np.zeros((m, 1)) + partitions_granularity  # (m, w)
    return broadcast_array.flatten()  # (p,)


def get_qubo_prices_linear(partitions_granularity_broadcast, b: float) -> Array:
    return 2.0 * b * np.diag(partitions_granularity_broadcast)  # (p, p)


def get_qubo_prices_quadratic(partitions_granularity_broadcast, b: float) -> Array:
    return np.outer(
        partitions_granularity_broadcast, partitions_granularity_broadcast
    )  # (p, p)


def get_qubo_returns(average_daily_returns_partition: Array) -> Array:
    return np.diag(average_daily_returns_partition)


def get_qubo_covariance(normalized_prices_partition: Array) -> Array:
    return np.cov(normalized_prices_partition.T)  # (p, p)


def get_normalized_prices(prices: Array, b: float) -> Array:
    ":math:`\\bar a`"
    factor = np.divide(b, prices[-1, :], dtype=np.float64, casting="unsafe")
    normalized_prices = prices * factor
    return normalized_prices


def get_normalized_prices_partition(
    normalized_prices: Array, partitions_granularity
) -> Array:
    n = len(normalized_prices)
    _npp = normalized_prices[..., None] * partitions_granularity  # (n, m, w)
    return _npp.reshape(n, -1)  # (n, p)


def get_average_daily_returns(prices: Array) -> Array:
    average_daily_returns = (np.diff(prices, axis=0) / prices[:-1]).mean(axis=0)
    return average_daily_returns  # (m,)


def get_average_daily_returns_partition(
    average_daily_returns: Array, partitions_granularity: Array
) -> Array:
    # average_daily_returns (m, )
    average_daily_returns_partition = (
        average_daily_returns[..., None] * partitions_granularity
    ).flatten()
    # average_daily_returns = granular_daily_returns.mean(axis=0)  # (m - 1)
    return typing.cast(Array, average_daily_returns_partition)


def get_average_daily_returns_partition_tecnalia(
    normalized_prices: Array, partitions_granularity: Array
) -> Array:
    n, _ = normalized_prices.shape  # m the number of "days"
    approximate_average = (normalized_prices[-1, :] - normalized_prices[0, :]) / (n - 1)
    return (approximate_average[..., None] * partitions_granularity).flatten()


def get_anual_returns(prices: Array) -> Array:
    return (prices[-1] - prices[0]) / prices[0]


def get_anual_returns_partition(anual_returns: Array, pw: Array) -> Array:
    return (anual_returns[..., None] * pw).flatten()


def get_upper_triangular(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Extract an upper triangular matrix.

    Example:
        >>> a = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
        >>> get_upper_triangular(a)
        array([[1, 4, 6],
               [0, 1, 8],
               [0, 0, 1]])


    Args:
        a (npt.NDArray[np.float64]): A numpy array.

    Returns:
        npt.NDArray[np.float64]: A numpy array.
    """
    return np.triu(a, 1) + np.triu(a, 0)


def get_lower_triangular(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Extract a lower triangular matrix.

    Example:
        >>> a = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
        >>> get_lower_triangular(a)
        array([[1, 0, 0],
               [4, 1, 0],
               [6, 8, 1]])


    Args:
        a (npt.NDArray[np.float64]): A numpy array.

    Returns:
        npt.NDArray[np.float64]: A numpy array.
    """
    return np.tril(a, -1) + np.tril(a, 0)


def get_qubo_dict(q: npt.NDArray[np.float64]) -> Q:
    """Create a dictionary from a symmetric matrix.

    This function is utilize to generate the qubo dictionary, which we will use to solve
    the problem in DWAVE.

    Example:
        >>> q = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
        >>> q
        array([[1, 2, 3],
               [2, 1, 4],
               [3, 4, 1]])
        >>> get_qubo_dict(q)  # doctest: +NORMALIZE_WHITESPACE
        {(0, 0): 1, (0, 1): 2, (0, 2): 3, (1, 0): 2, (1, 1): 1, (1, 2): 4, (2, 0): 3,
        (2, 1): 4, (2, 2): 1}

    Args:
        q (npt.NDArray[np.float64]): A symmetric matrix. The qubo matrix for example.

    Returns:
        Q: A dict with key the tuple of coordinate (i, j) and value the
            corresponding matrix value q[i, j].
    """
    n = len(q)
    qubo_dict: Q = {(i, j): q[i, j] for i in range(n) for j in range(n)}
    return qubo_dict
