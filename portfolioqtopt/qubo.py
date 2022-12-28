"""This class generates the QUBO from the weights (theta_one, theta_two, and 
theta_three), the budget, the historical price data for each asset, and the
expected returns of each asset as a matrix.
"""

import itertools as it

import numpy as np
import numpy.typing as npt

from .symmetric_to_triangular import get_upper_triangular


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

QuboDict = Dict[Tuple[int, int], npt.NDArray[np.float64]]


class Qubo(NamedTuple):
    matrix: npt.NDArray[np.float64]
    dictionary: QuboDict


def get_qubo(qi: npt.NDArray[np.float64], qij: npt.NDArray[np.float64]) -> Qubo:
    """Compute the qubo matrix and the corresponding dictionary.

    Args:
        qi (npt.NDArray[np.float64]): Diagonal, related to expected returns. Shape
            (n, n) where m is the historical depth of the data and
            n = number of funds * number of slices.
        qij (npt.NDArray[np.float64]): The quadratic part, related to variances.
            Shape (n, n).

    Returns:
        Qubo: Dataclass that has the qubo matrix and dictionary as attributes.
    """
    n = len(qij)

    # >>>>>>>>>>>>>>
    # COMPUTE QUBO
    # >>>>>>>>>>>>>>

    qubo = qi + qij

    # At first the matrix is complete, so with this method only the upper part of this
    # matrix is obtained.

    qubo_matrix = get_upper_triangular(qubo)

    # We generate the dictionary, which we will use to solve the problem in DWAVE.
    qubo_dict: QuboDict = {z: qubo_matrix[z] for z in it.product(*(range(n),) * 2)}

    return Qubo(qubo_matrix, qubo_dict)
