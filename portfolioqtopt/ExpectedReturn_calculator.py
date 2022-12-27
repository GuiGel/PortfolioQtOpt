# coding=utf-8
#                                              EXPECTED RETURN CALCULATOR
########################################################################################################################
# Esta clase calcula la rentabilidad esperada (basada en estimaciones de la Teoría Moderna de Carteras)
# a partir de datos históricos de precios.
########################################################################################################################
# coding=utf-8
import numpy as np
import numpy.typing as npt


class ExpectedReturns:
    def __init__(self, price_data):

        ######### Obtenemos los valores del precio #########
        self.price_data = price_data

        ######### Obtenemos las dimensiones del problema, num_rows = la profundidad historica de los datos #########
        ######### num_cols = el numero de fondos * el numero de slices #########
        self.num_rows, self.num_cols = price_data.shape

        self.exp_returns = np.zeros(self.num_cols)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # LLEVAMOS A CABO LA MEDIA CON EL ORIZONTE TEMPORAL COMPLETO
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def all_average(self):
        self.daily_return = np.zeros((self.num_rows - 1, self.num_cols))
        self.exp_returns = np.zeros(self.num_cols)
        # Calculate daily_return array which contains the daily returns contained in the historical price_data array:
        for i in range(self.num_cols):
            for j in range(self.num_rows - 1):
                self.daily_return[j, i] = (
                    self.price_data[j + 1, i] - self.price_data[j, i]
                )
            self.exp_returns[i] = np.mean(self.daily_return[:, i])


def get_expected_returns(
    normalized_prices: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the mean daily returns of prices.

    Args:
        normalized_prices (npt.NDArray[np.float64]): The normalized prices. shape (n, m)

    Returns:
        npt.NDArray[np.float64]: The mean daily returns of prices. shape (m, )
    """
    mean_daily_return = (normalized_prices[1:, :] - normalized_prices[:-1, :]).mean(
        axis=0
    )
    return mean_daily_return
