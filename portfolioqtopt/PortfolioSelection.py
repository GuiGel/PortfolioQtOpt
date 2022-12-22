# coding=utf-8
#                                        PORTFOLIO SELECTION
########################################################################################################################
# ESTA ES LA CLASE QUE OBTIENE LOS DATOS Y GENERA LOS VALORES PARA CONFORMAR EL QUBO
########################################################################################################################
import numpy as np
from Covariance_calculator import Covariance
from ExpectedReturn_calculator import ExpectedReturns
from Expand_Prices import ExpandPriceData


class PortfolioSelection:

    def __init__(self, theta_one, theta_two, theta_three, price_data, num_slices):
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # OBTENEMOS LOS VALORES DE INPUT
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        ######### El peso que le damos al retorno #########
        self.theta_one = theta_one
        ######### El peso que le damos al penalty, a la restriccion de no sobrepasar el budget #########
        self.theta_two = theta_two
        ######### El peso que le damos a la covarianza, es decir, a la diversidad #########
        self.theta_three = theta_three
        ######### Este es el budget, que es igual a 1 en todos los casos #########
        self.b = 1
        ######### En este punto de la ejecucion, los precios son los valores de los fondos en formato raw, sin normalizar #########
        self.price_data = price_data
        ######### El numero de slices es la granularidad que le vamos a dar a cada fondo. Es decir, la cantidad del #########
        ######### presupuesto que vamos a ser capaces de invertir. Por ejemplo, un 0.5, un 0.25, un 0.125... #########
        self.num_slices = num_slices

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # HACEMOS LA EXPANSION DE LOS PRECIOS EN FUNCION DE LAS SLIDES
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        ######### En este punto es en el que se normalizan los precios de cada fondo, utilizando como base el ultimo valor #########
        ######### registrado. A raiz de ese valor y en funcion de las slides, se va componiendo el resto de precios #########
        expand = ExpandPriceData(self.b, self.num_slices, self.price_data)

        ######### Se substituye los precios en formato raw por los precios en formato normalizado #########
        self.price_data = expand.price_data_expanded
        self.price_data_reversed = expand.price_data_expanded_reversed

        ######### Obtenemos las dimensiones del problema, num_rows = la profundidad historica de los datos #########
        ######### num_cols = el numero de fondos * el numero de slices #########
        self.num_rows, self.num_cols = self.price_data.shape

        ######### Los precios posibles, esto realmente es una lista de la proporcion del budget que puedes invertir #########
        ######### para cada uno de los fondos. Por ejemplo: 1.0, 0.5, 0.25, 0.125 #########
        self.prices = self.price_data[self.num_rows - 1, :].tolist()

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # OBTENEMOS EL EXPECTED RETURN
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        ######### Obtenemos el retorno esperado utilizando los precios como base #########
        ######### Inicializamos la clase #########
        expected = ExpectedReturns(self.price_data)

        ######### Calculamos el return esperado, utilizando una funcion de average #########
        expected.all_average()
        self.expected_returns = expected.exp_returns

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # OBTENEMOS EL EXPECTED RETURN
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        ######### Obtenemos los valores asociados al riesgo, es decir, la covarianza #########
        ######### Inicializamos la clase, calculamos la covarianza y la asignamos #########
        cov = Covariance(self.price_data)
        self.QUBO_covariance = cov.QUBO_covariance

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # CONFORMACION DE LOS VALORES DEL QUBO
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        ######### Generamos una matriz diagonal con los retornos, esta matriz la usaremos luego con el valor de theta_one #########
        self.QUBO_returns = np.diag(self.expected_returns)

        ######### Generamos una matriz diagonal con los precios posibles * 2. Esto se relacionara con los returns #########
        self.QUBO_prices_linear = np.diag([x * (2 * self.b) for x in self.prices])

        ######### Generamos una matriz simetrica también relacionada con los precios posibles. Esto se relacionara con la diversidad #########
        self.QUBO_prices_quadratic = np.outer(self.prices, self.prices)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # FORMACION DEFINITIVA DEL QUBO, CON LOS VALORES DE BIAS Y PENALIZACION INCLUIDOS
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        ######### Primero conformamos los valores de la diagonal, relacionados con el return y los precios #########
        self.qi = -(self.theta_one * self.QUBO_returns) - (self.theta_two * self.QUBO_prices_linear)

        ######### Ahora conformamos los valores cuadraticos, relacionados con la diversidad ##########
        self.qij = (self.theta_two * self.QUBO_prices_quadratic) + (self.theta_three * self.QUBO_covariance)

