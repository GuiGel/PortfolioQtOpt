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
    """This class generates the QUBO from the weights (theta1, theta2, and  theta3)"""

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

        Args:
            theta1 (float): First Lagrange multiplier.
            theta2 (float): Second Lagrange multiplier
            theta3 (float): Third Lagrange multiplier

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
