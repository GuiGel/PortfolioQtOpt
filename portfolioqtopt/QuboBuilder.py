# coding=utf-8
########################################################################################################################
# Esta clase genera el QUBO a partir de los pesos (theta_one, theta_two, y theta_three), el presupuesto,
# los datos históricos de precios de cada activo, y los rendimientos esperados de cada activo como una matriz.
########################################################################################################################


import itertools as it

import numpy as np
import numpy.typing as npt

from .SymmetricToTriangular import get_upper_triangular


class QUBO:
    def __init__(
        self, qi: npt.NDArray[np.float64], qij: npt.NDArray[np.float64]
    ) -> None:
        self.qi = qi
        self.qij = qij

        # Obtenemos las dimensiones del problema,
        # m = la profundidad historica de los datos
        # n = el numero de fondos * el numero de slices
        m, n = self.qij.shape

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
