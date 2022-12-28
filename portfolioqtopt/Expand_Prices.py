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
    """Compute the possible proportions of the budget that we can allocate to each fund.

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


def get_expand_prices(
    prices,
    slices,
    slices_list,
    budget=1,
):
    num_rows, num_cols = prices.shape

    ######### Inicializamos la variable self.price_data_expanded #########
    price_data_expanded = None

    assert num_cols > 0

    for i in range(num_cols):

        ######### Inicializamos asset_prices #########
        asset_prices = np.zeros((num_rows, slices))

        ######### Este es el valor que vamos a usar para normalizar los valores de compra de cada asset. Se hace por slide #########
        norm_price_factor = budget / prices[num_rows - 1, i]

        ######### Este for va rellenando los precios normalizados por cada asset y slice a lo largo del periodo temporal #########
        for j in range(slices):
            for k in range(num_rows):
                asset_prices[k, j] = prices[k, i] * slices_list[j] * norm_price_factor

        ######### se va generando poco a poco price_data_expanded, que incluye todos los precios normalizados #########
        if i == 0:
            price_data_expanded = asset_prices
        else:
            assert isinstance(price_data_expanded, np.ndarray)
            price_data_expanded = np.append(price_data_expanded, asset_prices, 1)

    return price_data_expanded


def get_expand_prices_opt(
    prices: npt.NDArray[np.float64],
    slices_list: npt.NDArray[np.float64],
    budget: float = 1.0,
    reversed: bool = False,
) -> npt.NDArray[np.float64]:
    """Optimized version of get_expand_prices.
    Speedup of 50X with the original ``get_expand_prices`` code.

    Args:
        prices (npt.NDArray[np.float64]): The fund prices with shape
            (prices number, funds number).
        slices_list (npt.NDArray[np.float64]): Granularity slice list.
        budget (int, optional): The initial budget. Defaults to 1.

    Returns:
        npt.NDArray[np.float64]: The expanded prices.
    """
    # norm_price_factor is the value we will use to normalize the purchase values of each
    # asset

    # TODO ensure that prices values must be > 0 and not np.Nan

    factor = prices[-1, :]
    if reversed:
        factor = prices[0, :]

    norm_price_factor = np.divide(budget, factor, dtype=np.float64, casting="unsafe")
    all_assert_prices = (
        np.expand_dims(prices, axis=2) * slices_list * norm_price_factor.reshape(-1, 1)
    )
    _, num_cols, num_slices = all_assert_prices.shape
    asset_prices = all_assert_prices.reshape(-1, num_cols * num_slices)
    return asset_prices.astype(np.float64)


def get_expand_prices_reversed(raw_price_data, slices, slices_list, budget):

    num_rows, num_cols = raw_price_data.shape

    ######### Inicializamos la variable self.price_data_expanded #########
    price_data_expanded_reversed = None

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # EN FUNCIÓN DE LOS PRECIOS Y LAS PROPORCIONES, CREAMOS LOS PRECIOS EXPANDIDOS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for i in range(num_cols):

        ######### Inicializamos asset_prices #########
        asset_prices = np.zeros((num_rows, slices))

        ######### Este es el valor que vamos a usar para normalizar los valores de compra de cada asset. Se hace por slide #########
        norm_price_factor = budget / raw_price_data[0, i]

        ######### Este for va rellenando los precios normalizados por cada asset y slice a lo largo del periodo temporal #########
        for j in range(slices):
            for k in range(num_rows):
                asset_prices[k, j] = (
                    raw_price_data[k, i] * slices_list[j] * norm_price_factor
                )

        ######### se va generando poco a poco price_data_expanded, que incluye todos los precios normalizados #########
        if i == 0:
            price_data_expanded_reversed = asset_prices
        else:
            price_data_expanded_reversed = np.append(
                price_data_expanded_reversed, asset_prices, 1
            )

    return price_data_expanded_reversed


class ExpandPriceData:
    """Based on the prices and ratios, create the expanded prices."""

    def __init__(self, budget, slices, prices):

        slices_list = get_slices_list(slices)

        self.price_data_expanded = get_expand_prices_opt(prices, slices_list, budget)

        self.price_data_expanded_reversed = get_expand_prices_opt(
            prices,
            slices_list,
            budget,
            reversed=True,
        )
