# coding=utf-8
#                                                    EXPAND RETURNS
########################################################################################################################
# Con esta clase se crean columnas de datos de precios históricos que representan varios porcentajes del presupuesto.
# Por ejemplo, si el presupuesto es 20 y el precio de un fondo es 100, se podrían analizar varios porcentajes del fondo
# en función del presupuesto: 5, 10, 15 y 20 para encontrar la mejor opción.
# Esto, por supuesto, aumenta el espacio de búsqueda.
########################################################################################################################
# coding=utf-8
import numpy as np
import numpy.typing as npt


def get_slices_list(slices: int) -> npt.NDArray[np.float64]:
    """Generate a list of slices.

    Example:

    >>> get_slices_list(5)
    array([1.    , 0.5   , 0.25  , 0.125 , 0.0625])

    Args:
        slices (int): The number of slices is the granularity that we are
            going to give to each fund. That is, the amount of the budget we
            will be able to invest.

    Returns:
        npt.NDArray[np.float64]: List of slices values.
    """
    return np.power(0.5, np.arange(slices))


class ExpandPriceData:
    def __init__(self, budget, slices, raw_price_data):
        ######### Inicializamos los datos de entrada. El numero de slices es el numero de proporciones consideradas #########
        self.slices = slices
        self.b = budget

        ######### Obtenemos las dimensiones del problema, num_rows = la profundidad historica de los datos #########
        ######### num_cols = el numero de fondos * el numero de slices #########
        num_rows, num_cols = raw_price_data.shape

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # GENERAMOS LAS POSIBLES PROPORCIONES DEL BUDGET QUE PODEMOS ASIGNAR A CADA FONDO
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.slices_list = get_slices_list(slices)

        ######### Inicializamos la variable self.price_data_expanded #########
        self.price_data_expanded = None

        ######### Inicializamos la variable self.price_data_expanded #########
        self.price_data_expanded_reversed = None

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # EN FUNCIÓN DE LOS PRECIOS Y LAS PROPORCIONES, CREAMOS LOS PRECIOS EXPANDIDOS
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        for i in range(num_cols):

            ######### Inicializamos asset_prices #########
            asset_prices = np.zeros((num_rows, self.slices))

            ######### Este es el valor que vamos a usar para normalizar los valores de compra de cada asset. Se hace por slide #########
            norm_price_factor = budget / raw_price_data[num_rows - 1, i]

            ######### Este for va rellenando los precios normalizados por cada asset y slice a lo largo del periodo temporal #########
            for j in range(self.slices):
                for k in range(num_rows):
                    asset_prices[k, j] = (
                        raw_price_data[k, i] * self.slices_list[j] * norm_price_factor
                    )

            ######### se va generando poco a poco price_data_expanded, que incluye todos los precios normalizados #########
            if i == 0:
                self.price_data_expanded = asset_prices
            else:
                self.price_data_expanded = np.append(
                    self.price_data_expanded, asset_prices, 1
                )

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # EN FUNCION DE LOS PRECIOS Y LAS PROPORCIONES, CREAMOS LOS PRECIOS EXPANDIDOS
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        for i in range(num_cols):

            ######### Inicializamos asset_prices #########
            asset_prices = np.zeros((num_rows, self.slices))

            ######### Este es el valor que vamos a usar para normalizar los valores de compra de cada asset. Se hace por slide #########
            norm_price_factor = budget / raw_price_data[0, i]

            ######### Este for va rellenando los precios normalizados por cada asset y slice a lo largo del periodo temporal #########
            for j in range(self.slices):
                for k in range(num_rows):
                    asset_prices[k, j] = (
                        raw_price_data[k, i] * self.slices_list[j] * norm_price_factor
                    )

            ######### se va generando poco a poco price_data_expanded, que incluye todos los precios normalizados #########
            if i == 0:
                self.price_data_expanded_reversed = asset_prices
            else:
                self.price_data_expanded_reversed = np.append(
                    self.price_data_expanded_reversed, asset_prices, 1
                )
