"""Preliminary calculations.

In this module, we calculate the qubo dict for a given portfolio represented by a 
:class:`portfolioqtopt.assets_.Assets` object, with a granularity depth :math:`w` 
and a budget :math:`b`.

.. note:: 
    We keep most of the small functions we have implemented as they help to understand 
    the calculations for this project and are an aid to accurate unit testing.
"""
from __future__ import annotations

import typing
from functools import cache

import numpy as np
from dimod.typing import Bias, Variable

from portfolioqtopt.assets import Array, Assets

Q = typing.Mapping[typing.Tuple[Variable, Variable], Bias]


def expand(array: Array, pw: Array, b: float = 1.0) -> Array:
    """Expand a 1D or 2D array by multiplying all its values with :math:`p_w` values.

    Example:

        >>> pw = [1, 2, 3]

        If array is a vector:

        >>> array = np.array([1, 5], dtype=np.float64)
        >>> expand(array, pw)
        array([ 1.,  2.,  3.,  5., 10., 15.])

        If array is a matrix:

        >>> array = np.array([[1, 2], [5, 6]], dtype=np.float64)
        >>> expand(array, pw)
        array([[ 1.,  2.,  3.,  2.,  4.,  6.],
               [ 5., 10., 15.,  6., 12., 18.]])

    Args:
        array (Array): A 1D or 2D array to expand with pw. (m,) or (n, m).
        pw (Array): A granularity partition. (w,)
        b (float, optional):The budget. Defaults to 1.0.

    Raises:
        ValueError: The array is not 1D or 2D.

    Returns:
        Array: If array is (n, m) -> (n, m * w) else (m,) -> (m * w,)
    """
    dim = len(array.shape)
    if dim == 1:
        # array (m, )
        expanded_array = (array[..., None] * pw).flatten()  # p = m * w
    elif dim == 2:
        n, _ = array.shape
        expanded_ = array[..., None] * b * pw  # (n, m, w)
        expanded_array = expanded_.reshape(n, -1)  # (n, m * w) = (n, p)
    else:
        raise ValueError(f"The array must be a 2D or 1D but is {dim}D.")
    return expanded_array


@cache
def get_pw(w: int) -> Array:
    """Compute the possible proportions :math:`p_{k}` of the budget that we can \
    allocate to each fund.

    For :math:`k` in :math:`[1, w]`,  :math:`p_{k}=1/2^{k-1}`.  

    Example:

        >>> get_pw(5)
        array([1.    , 0.5   , 0.25  , 0.125 , 0.0625])

    Args:
        w (int): The partitions number :math:`w` that determine the precision of the \
granularity partition.
    Returns:
        npt.NDArray[np.floating[typing.Any]]: The List of fraction values \
            :math:`p_{w}`. (w,)
    """
    return np.power(0.5, np.arange(w))


def get_pw_broadcast(pw: Array, m: int) -> Array:
    """Broadcast pw by concatenating :math:`p_{w}` m times along 1 dimension.

    Example:
        >>> pw = [1, 0.5, 0.25, 0.125]
        >>> get_pw_broadcast(pw, 2)  #
        array([1.   , 0.5  , 0.25 , 0.125, 1.   , 0.5  , 0.25 , 0.125])

    Args:
        pw (Array): Granularity partition. (w,)
        m (int): The number of assets.

    Returns:
        Array: The broadcasted granular partitions. (p,)
    """
    # (p10, .., p1w-1, ..., pm0 ... pmw-1)
    broadcast_array = np.zeros((m, 1), dtype=np.float64) + pw  # (m, w)
    return broadcast_array.flatten()  # (p,)


def get_quboprices_linear(pw_broadcast: Array, b: float) -> Array:
    """Compute the linear prices part of the qubo.

    Example:

        >>> get_quboprices_linear([1, 2, 1, 2], 3)
        array([[ 6.,  0.,  0.,  0.],
               [ 0., 12.,  0.,  0.],
               [ 0.,  0.,  6.,  0.],
               [ 0.,  0.,  0., 12.]])

    Args:
        pw_broadcast (Array): A 1D array. (p,)
        b (float): The budget.

    Returns:
        Array: A 2D array. (p, p)
    """
    return 2.0 * b * np.diag(pw_broadcast)  # (p, p)


def get_quboprices_quadratic(pw_broadcast: Array) -> Array:
    """Compute the quadratic part of the qubo.

    Example:

        >>> pw_broadcast = [1, 2, 3]
        >>> get_quboprices_quadratic(pw_broadcast)
        array([[1, 2, 3],
               [2, 4, 6],
               [3, 6, 9]])

    Args:
        pw_broadcast (Array): A 1D matrix. (p,)

    Returns:
        Array: The outer product of input vector by himself. (p, p)
    """
    return np.outer(pw_broadcast, pw_broadcast)


def get_quboreturns(average_daily_returns_partition: Array) -> Array:
    """Get the qubo returns part.

    Example:

        >>> get_quboreturns([1, 2, 3])
        array([[1, 0, 0],
               [0, 2, 0],
               [0, 0, 3]])

    Args:
        average_daily_returns_partition (Array): A 1D array. (p,)

    Returns:
        Array: A diagonal array. (p, p)
    """
    return np.diag(average_daily_returns_partition)


def get_qubocovariance(normalized_prices_partition: Array) -> Array:
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


def get_qubodict(q: Array) -> Q:
    """Create a dictionary from a symmetric matrix upper indexes and values.

    This function is utilize to generate the qubo dictionary, which we will use to solve
    the problem in DWAVE.

    Example:
        >>> q = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
        >>> q
        array([[1, 2, 3],
               [2, 1, 4],
               [3, 4, 1]])
        >>> get_qubodict(q)  # doctest: +NORMALIZE_WHITESPACE
        {(0, 0): 1, (0, 1): 2, (0, 2): 3, (1, 0): 2, (1, 1): 1, (1, 2): 4, (2, 0): 3,
        (2, 1): 4, (2, 2): 1}

    Args:
        q (Array): A symmetric matrix. The qubo matrix for example.

    Returns:
        Q: A dict with key the tuple of coordinate (i, j) and value the
            corresponding matrix value q[i, j].
    """
    n = len(q)
    qubodict: Q = {(i, j): q[i, j] for i in range(n) for j in range(n)}
    return qubodict


def get_qubo(
    a: Assets, b: float, w: int, theta1: float, theta2: float, theta3: float
) -> Q:
    """Function that return the qubo ready for passing to the solver.

    Compute the qubo as stand by the equation (20) in :cite:p:`Grant2021`.

    Args:
        a (Assets): Assets with asset prices.
        b (float): The initial budget to allocate.
        w (int): The partition number.
        theta1 (float): The first Lagrange multiplier.
        theta2 (float): The second Lagrange multiplier.
        theta3 (float): The third Lagrange multiplier.

    Returns:
        Q: The qubo as a dict where the keys are a tuple representing the position of
        the corresponding value in the qubo matrix.
    """

    # Compute the granularity partition
    pw = get_pw(w)

    # Compute the  partition of the normalized prices
    npp = expand(a.normalized_prices, pw, b)  # (n, p)

    # Compute the partition of the anual returns
    anual_returns_partitions = expand(a.anual_returns, pw)  # (p,)

    # Compute the partitions of the average daily returns.
    # adrp = expand(a.average_daily_returns, pw)  # (p,)
    adrp = expand(
        a.normalized_prices_approx, pw, b
    )  # (p,) Tecnalia option as describe in the article.

    # Set qubo values
    qubocovariance = get_qubocovariance(npp)
    quboreturns = get_quboreturns(adrp)  # (p, p)
    pw_broadcast = get_pw_broadcast(pw, a.m)
    qubolinear = get_quboprices_linear(pw_broadcast, b)
    quboquadratic = get_quboprices_quadratic(pw_broadcast)

    # Create qubo.
    qi = -theta1 * quboreturns - theta2 * qubolinear  # (p, p).  eq (21a)
    qij = theta2 * quboquadratic + theta3 * qubocovariance  # (p, p). eq (21b)
    qubo = typing.cast(Array, qi + qij)
    qubomatrix = get_upper_triangular(qubo)
    q = get_qubodict(qubomatrix)

    return q
