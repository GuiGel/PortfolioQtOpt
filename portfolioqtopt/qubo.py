"""This class generates the QUBO from the weights (theta1, theta2, and 
theta3), the budget, the historical price data for each asset, and the
expected returns of each asset as a matrix.
"""

import itertools as it

import numpy as np
import numpy.typing as npt

from portfolioqtopt.symmetric_to_triangular import get_upper_triangular


class QUBO:
    def __init__(
        self, qi: npt.NDArray[np.float64], qij: npt.NDArray[np.float64]
    ) -> None:
        # Obtenemos las dimensiones del problema,
        # m = la profundidad historica de los datos
        # n = el numero de fondos * el numero de slices
        m, n = qij.shape

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # GENERAMOS EL QUBO
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # En un primer momento generamos un matriz en la que unimos la
        # diagonal, relacionada con los expected returns, y la parte
        # cuadrática, relacionada con las varianzas.
        qubo = qi + qij

        # En un primer momento la matriz es completa, por lo que con este
        # método se obtiene unicamente la parte superior de esta matriz.

        self.qubo = get_upper_triangular(qubo)

        # Generamos el diccionario, que es lo que vamos a emplear para
        # resolver el problema en DWAVE.
        self.qubo_dict = {z: self.qubo[z] for z in it.product(range(n), range(n))}


from typing import Dict, NamedTuple, Tuple

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


def get_qubo(qi: npt.NDArray[np.float64], qij: npt.NDArray[np.float64]) -> Qubo:
    """Compute the qubo matrix and the corresponding dictionary.

    Args:
        qi (npt.NDArray[np.float64]): Diagonal, related to expected returns. Shape
            (n, n) where m is the historical depth of the data and
            n = number of funds * number of slices.
        qij (npt.NDArray[np.float64]): The quadratic part, related to variances.
            Shape (n, n).

    Returns:
        Qubo: Tuple that has the qubo matrix and dictionary as attributes.
    """
    qubo = qi + qij
    qubo_matrix = get_upper_triangular(qubo)
    qubo_dict = get_qubo_dict(qubo_matrix)
    return Qubo(qubo_matrix, qubo_dict)
