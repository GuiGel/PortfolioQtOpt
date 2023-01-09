import typing
from typing import Dict, NamedTuple, Tuple

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


QuboDict = Dict[Tuple[int, int], np.floating]


class Qubo(NamedTuple):
    matrix: npt.NDArray[np.float64]
    dictionary: QuboDict


def get_qubo_dict(q: npt.NDArray[np.float64]) -> QuboDict:
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
        QuboDict: A dict with key the tuple of coordinate (i, j) and value the
            corresponding matrix value q[i, j].
    """
    n = len(q)
    qubo_dict: QuboDict = {(i, j): q[i, j] for i in range(n) for j in range(n)}
    return qubo_dict
