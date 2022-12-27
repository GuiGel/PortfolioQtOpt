# coding=utf-8
#                                        PORTFOLIO SELECTION
########################################################################################################################
# ESTA ES LA CLASE QUE OBTIENE LOS DATOS Y GENERA LOS VALORES PARA CONFORMAR EL QUBO
########################################################################################################################
import numpy as np
import numpy.typing as npt

from .Covariance_calculator import get_prices_covariance
from .Expand_Prices import ExpandPriceData
from .ExpectedReturn_calculator import get_expected_returns


class PortfolioSelection:
    """The PortfolioSelection class.

    Attributes:
        theta_one (float): The weight we give to return.
        theta_two (float): The weight we give to the penalty, to the
                constraint of not exceeding the budget.
        theta_three (float): The weight we give to covariance, i.e. to
                diversity.
        price_data (npt.NDArray[np.float64]): At this point in the
                execution, prices are the values of the funds in raw format,
                without normalizing.
        num_slices (int): The number of slices is the granularity we are
            going to give to each fund. That is, the amount of the budget
            that we will be able to invest. For example, a 0.5, a 0.25, a
            0.125...
        b (int): The budget, which is equal to 1 in all cases
    """

    def __init__(
        self,
        theta_one: float,
        theta_two: float,
        theta_three: float,
        price_data: npt.NDArray[np.float64],
        num_slices: int,
    ) -> None:
        """Initialized the ``PortfolioSelection`` class.

        Args:
            theta_one (float): The weight we give to return.
            theta_two (float): The weight we give to the penalty, to the
                constraint of not exceeding the budget.
            theta_three (float): The weight we give to covariance, i.e. to
                diversity.
            price_data (npt.NDArray[np.float64]): At this point in the
                execution, prices are the values of the funds in raw format,
                without normalizing.
            num_slices (int): The number of slices is the granularity we are
                going to give to each fund. That is, the amount of the budget
                that we will be able to invest. For example, a 0.5, a 0.25, a
                0.125...
        """
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # OBTENEMOS LOS VALORES DE INPUT
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        self.theta_one = theta_one
        self.theta_two = theta_two
        self.theta_three = theta_three
        self.price_data = price_data
        self.num_slices = num_slices

        self.b = 1.0  # Este es el budget, que es igual a 1 en todos los casos

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # HACEMOS LA EXPANSION DE LOS PRECIOS EN FUNCIÓN DE LAS SLIDES
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # En este punto es en el que se normalizan los precios de cada fondo,
        # utilizando como base el ultimo valor registrado. A raíz de ese valor
        # y en función de las slides, se va componiendo el resto de precios
        expand = ExpandPriceData(self.b, self.num_slices, self.price_data)

        # Se substituye los precios en formato raw por los precios en formato
        # normalizado.
        self.price_data = expand.price_data_expanded
        self.price_data_reversed = expand.price_data_expanded_reversed

        # Obtenemos las dimensiones del problema, num_rows = la profundidad historica de los datos
        # num_cols = el numero de fondos * el numero de slices
        self.num_rows, self.num_cols = self.price_data.shape

        # Los precios posibles, esto realmente es una lista de la proporción
        # del budget que puedes invertir para cada uno de los fondos.
        # Por ejemplo: 1.0, 0.5, 0.25, 0.125
        # NOTE: We talk about the final possible prices
        self.prices = self.price_data[self.num_rows - 1, :].tolist()

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # OBTENEMOS EL EXPECTED RETURN
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        ######### Obtenemos el retorno esperado utilizando los precios como base #########

        ######### Calculamos el return esperado, utilizando una función de average #########

        self.expected_returns = get_expected_returns(self.price_data)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # OBTENEMOS EL EXPECTED RETURN
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        ######### Obtenemos los valores asociados al riesgo, es decir, la covariance #########
        self.QUBO_covariance = get_prices_covariance(self.price_data)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # CONFORMACIÓN DE LOS VALORES DEL QUBO
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        ######### Generamos una matriz diagonal con los retornos, esta matriz la usaremos luego con el valor de theta_one #########
        self.QUBO_returns = np.diag(self.expected_returns)

        ######### Generamos una matriz diagonal con los precios posibles * 2. Esto se relacionara con los returns #########
        self.QUBO_prices_linear = np.diag(
            [x * (2 * self.b) for x in self.prices]
        )  # (num_cols, num_cols)

        ######### Generamos una matriz simétrica también relacionada con los precios posibles. Esto se relacionara con la diversidad #########
        self.QUBO_prices_quadratic = np.outer(
            self.prices, self.prices
        )  # (num_cols, num_cols)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # FORMACIÓN DEFINITIVA DEL QUBO, CON LOS VALORES DE BIAS Y PENALIZACIÓN INCLUIDOS
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        ######### Primero conformamos los valores de la diagonal, relacionados con el return y los precios #########
        self.qi = -(self.theta_one * self.QUBO_returns) - (
            self.theta_two * self.QUBO_prices_linear
        )

        ######### Ahora conformamos los valores cuadráticos, relacionados con la diversidad ##########
        self.qij = (self.theta_two * self.QUBO_prices_quadratic) + (
            self.theta_three * self.QUBO_covariance
        )
