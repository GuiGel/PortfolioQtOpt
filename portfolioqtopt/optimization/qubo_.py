"""Preliminary calculations.

In this module, we calculate all preliminary results that depend only on prices,
granularity depth and budget.
"""
from __future__ import annotations

import typing
from functools import cache

import numpy as np
from dimod.typing import Bias, Variable

from portfolioqtopt.optimization.assets_ import Array, Assets

Q = typing.Mapping[typing.Tuple[Variable, Variable], Bias]


def expand(array: Array, pw: Array, b: float = 1.0) -> Array:
    dim = len(array.shape)
    if dim == 1:
        # array (m, )
        expanded_array = (array[..., None] * pw).flatten()  # m * w
    elif dim == 2:
        n, _ = array.shape
        expanded_ = array[..., None] * b * pw  # (n, m, w)
        expanded_array = expanded_.reshape(n, -1)  # (n, p)
    else:
        raise ValueError("The array must be a Matrix or a Vector.")
    return expanded_array


@cache
def get_pw(w: int) -> Array:
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


def get_pw_broadcast(pw: Array, m: int) -> Array:
    # (p10, .., p1w-1, ..., pm0 ... pmw-1)
    broadcast_array = np.zeros((m, 1)) + pw  # (m, w)
    return broadcast_array.flatten()  # (p,)


def get_qubo_prices_linear(pw_broadcast: Array, b: float) -> Array:
    return 2.0 * b * np.diag(pw_broadcast)  # (p, p)


def get_qubo_prices_quadratic(pw_broadcast: Array) -> Array:
    return np.outer(pw_broadcast, pw_broadcast)  # (p, p)


def get_qubo_returns(average_daily_returns_partition: Array) -> Array:
    return np.diag(average_daily_returns_partition)


def get_qubo_covariance(normalized_prices_partition: Array) -> Array:
    return np.cov(normalized_prices_partition.T)  # (p, p)


def get_normalized_prices_partition(a: Assets, pw: Array, b: float) -> Array:
    return expand(a.normalized_prices, pw, b)  # (n, p)


def get_average_daily_returns_partition_(a: Assets, pw: Array) -> Array:
    return expand(a.average_daily_returns, pw)  # (m - 1)


def get_average_daily_returns_partition_tecnalia(a: Assets, pw: Array) -> Array:
    return expand(a.normalized_prices_approx, pw)


def get_anual_returns_partition(a: Assets, pw: Array) -> Array:
    return expand(a.anual_returns, pw)


def get_upper_triangular(a: Array) -> Array:
    """Extract an upper triangular matrix.

    Example:
        >>> a = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
        >>> get_upper_triangular(a)
        array([[1, 4, 6],
               [0, 1, 8],
               [0, 0, 1]])


    Args:
        a (Array): A numpy array.

    Returns:
        Array: A numpy array.
    """
    return np.triu(a, 1) + np.triu(a, 0)


def get_lower_triangular(a: Array) -> Array:
    """Extract a lower triangular matrix.

    Example:
        >>> a = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
        >>> get_lower_triangular(a)
        array([[1, 0, 0],
               [4, 1, 0],
               [6, 8, 1]])


    Args:
        a (Array): A numpy array.

    Returns:
        Array: A numpy array.
    """
    return np.tril(a, -1) + np.tril(a, 0)


def get_qubo_dict(q: Array) -> Q:
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
        q (Array): A symmetric matrix. The qubo matrix for example.

    Returns:
        Q: A dict with key the tuple of coordinate (i, j) and value the
            corresponding matrix value q[i, j].
    """
    n = len(q)
    qubo_dict: Q = {(i, j): q[i, j] for i in range(n) for j in range(n)}
    return qubo_dict


def get_qubo(
    a: Assets, b: float, w: int, theta1: float, theta2: float, theta3: float
) -> Q:

    # Compute the granularity partition
    pw = get_pw(w)

    # Compute the  partition of the normalized prices
    npp = expand(a.normalized_prices, pw, b)  # (n, p)

    # Compute the partition of the anual returns
    anual_returns_partitions = expand(a.anual_returns, pw)  # (p,)

    # Compute the partitions of the average daily returns.
    # adrp = expand(a.average_daily_returns, pw)  # (p,)
    adrp = expand(a.normalized_prices_approx, pw)  # (p,) Tecnalia option.

    # Set qubo values
    qubo_covariance = get_qubo_covariance(npp)
    qubo_returns = get_qubo_returns(adrp)
    pw_broadcast = get_pw_broadcast(pw, a.m)
    qubo_linear = get_qubo_prices_linear(pw_broadcast, b)
    qubo_quadratic = get_qubo_prices_quadratic(pw_broadcast)

    # Create qubo.
    qi = -theta1 * qubo_returns - theta2 * qubo_linear  # (p, p).  eq (21a)
    qij = theta2 * qubo_quadratic + theta3 * qubo_covariance  # (p, p). eq (21b)
    qubo_ = typing.cast(Array, qi + qij)
    qubo_matrix = get_upper_triangular(qubo_)
    q = get_qubo_dict(qubo_matrix)

    return q
