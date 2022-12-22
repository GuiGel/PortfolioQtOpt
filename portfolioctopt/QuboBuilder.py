# coding=utf-8
########################################################################################################################
# Esta clase genera el QUBO a partir de los pesos (theta_one, theta_two, y theta_three), el presupuesto,
# los datos históricos de precios de cada activo, y los rendimientos esperados de cada activo como una matriz.
########################################################################################################################

from SymmetricToTriangular import TriangleGenerator

class QUBO:

    def __init__(self, qi, qij):
        ######### Inicializamos los datos de input #########
        self.qi = qi
        self.qij = qij

        ######### Obtenemos las dimensiones del problema, num_rows = la profundidad historica de los datos #########
        ######### num_cols = el numero de fondos * el numero de slices #########
        self.num_rows, self.num_cols = self.qij.shape
        self.n = self.num_cols

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # GENERAMOS EL QUBO
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        ######### En un primer momento generamos un matriz en la que unimos la diagonal, relacionada con los #########
        ######### expected returns, y la parte cuadrática, relacionada con las varianzas #########
        self.qubo = qi + qij

        ######### En un primer momento la matriz es completa, por lo que con este metodo se obtiene unicamente #########
        ######### la parte superior de esta matriz #########
        self.triangle = TriangleGenerator(self.n, self.qubo)
        self.triangle.upper()
        self.qubo = self.triangle.upper_matrix

        ######### Generamos el diccionario, que es lo que vamos a emplear para resolver el problema en DWAVE #########
        self.qubo_dict = {}
        self.qubo_dict.update({(i, j): self.qubo[i][j] for i in range(self.n) for j in range(self.n)})





