# coding=utf-8
#                                                    COVARIANCE CALCULATOR
########################################################################################################################
#   Calcular la matriz de covarianza entre activos utilizando datos históricos de precios.
#   La matriz de covarianza se utiliza para implementar el término de diversidad en el problema de selección de cartera.
########################################################################################################################
# coding=utf-8
import numpy as np


class Covariance:
    def __init__(self, price_data):

        ######### Obtenemos las dimensiones del problema, num_rows = la profundidad historica de los datos #########
        ######### num_cols = el numero de fondos * el numero de slices #########
        self.num_rows, self.num_cols = price_data.shape
        self.price_data = price_data

        ######### Inicializamos la matriz de covarianzas #########
        self.QUBO_covariance = np.zeros((self.num_cols, self.num_cols))
        ######### Calculamos las covarianzas #########
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                self.QUBO_covariance[i, j] = np.cov(self.price_data[:, i], self.price_data[:, j])[0][1]
                # self.QUBO_covariance[i, j] = ((price_data[num_rows - 1, i] - np.mean(price_data[:, i])) * (price_data[num_rows - 1, j] - np.mean(price_data[:, j]))) / (num_cols - 1)