"""Module that implement the Qubo creation logic."""
from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt
from dimod.typing import Bias, Variable

from portfolioqtopt.markovitz_portfolio import Selection


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


QuboDict = typing.Mapping[typing.Tuple[Variable, Variable], Bias]


@dataclass
class Qubo:
    """Dataclass that contains the qubo matrix and dictionary. It is accessible by the
    :py:attr:`portfolioqtopt.qubo.QuboFactory.qubo` attribute.

    Attributes:
        matrix (npt.NDArray[np.float64]): The Qubo matrix.
        dictionary (QuboDict): The Qubo dictionary.
    """

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


class QuboFactory:
    """Generates the QUBO matrix from the Lagrange multipliers theta1, theta2, and
    theta3"""

    def __init__(
        self, selection: Selection, theta1: float, theta2: float, theta3: float
    ) -> None:
        self.selection = selection
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3

    def __getitem__(self, val: typing.Any) -> QuboFactory:
        selection = self.selection[val]
        return QuboFactory(selection, self.theta1, self.theta2, self.theta3)

    @cached_property
    def qubo(self) -> Qubo:
        """Compute the qubo matrix and it's corresponding dictionary.

        Example:

            >>> prices = np.array(
            ...     [
            ...         [100, 104, 102],
            ...         [10, 10.2, 10.4],
            ...     ],
            ...     dtype=np.floating,
            ... ).T
            >>> selection = Selection(prices, 2, 1.0)
            >>> qubo_factory = QuboFactory(selection, 0.1, 0.2, 0.3)
            >>> qubo = qubo_factory.qubo
            >>> qubo.matrix
            array([[-0.20092312,  0.20011534,  0.40011312,  0.20005656],
                   [ 0.        , -0.1504904 ,  0.20005656,  0.10002828],
                   [ 0.        ,  0.        , -0.20186945,  0.20011095],
                   [ 0.        ,  0.        ,  0.        , -0.15096246]])
            >>> qubo.dictionary
            {(0, 0): -0.20092312128471299, (0, 1): 0.20011534025374858, (0, 2): \
0.4001131221719457, (0, 3): 0.20005656108597286, (1, 0): 0.0, (1, 1): \
-0.15049039570579364, (1, 2): 0.20005656108597286, (1, 3): 0.10002828054298643, \
(2, 0): 0.0, (2, 1): 0.0, (2, 2): -0.2018694454113006, (2, 3): 0.20011094674556215, \
(3, 0): 0.0, (3, 1): 0.0, (3, 2): 0.0, (3, 3): -0.15096245939204084}

        Returns:
            Qubo: A dataclass that have the qubo matrix and the qubo index dictionary
                as attributes.
        """
        # We obtain the values associated to the risk, i.e. the covariance
        qubo_covariance = np.cov(self.selection.npp.T)  # (p, p)

        # Set the Qubo values
        qubo_returns = np.diag(self.selection.expected_returns)  # (p, p)
        qubo_prices_linear = (
            2.0 * self.selection.b * np.diag(self.selection.npp_last)
        )  # (p, p)
        qubo_prices_quadratic = np.outer(
            self.selection.npp_last, self.selection.npp_last
        )  # (p, p)

        # Final QUBO formation, with bias and penalty values included
        qi = (
            -self.theta1 * qubo_returns - self.theta2 * qubo_prices_linear
        )  # (p, p).  eq (21a)
        qij = (
            self.theta2 * qubo_prices_quadratic + self.theta3 * qubo_covariance
        )  # (p, p). eq (21b)
        qubo = typing.cast(npt.NDArray[np.floating[typing.Any]], qi + qij)

        qubo_matrix = get_upper_triangular(qubo)
        qubo_dict = get_qubo_dict(qubo_matrix)
        return Qubo(qubo_matrix, qubo_dict)


__test__ = {"QuboCreator.qubo": QuboFactory.qubo}
