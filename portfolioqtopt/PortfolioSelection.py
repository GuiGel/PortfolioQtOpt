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
        theta1 (float): The weight we give to return.
        theta2 (float): The weight we give to the penalty, to the constraint of
            not exceeding the budget.
        theta3 (float): The weight we give to covariance, i.e. to diversity.
        price_data (npt.NDArray[np.float64]): At this point in the execution, prices
            are the values of the funds in raw format, without normalizing.
            Shape (m, n) where m is the historical depth of the data and
            n = funds number * slices number.
        num_slices (int): The number of slices is the granularity we are going to
            give to each fund. That is, the amount of the budget that we will be
            able to invest. For example, a 0.5, a 0.25, a 0.125...
        b (int): The budget, which is equal to 1 in all cases
    """

    def __init__(
        self,
        theta1: float,
        theta2: float,
        theta3: float,
        price_data: npt.NDArray[np.float64],
        num_slices: int,
    ) -> None:
        """Initialized the ``PortfolioSelection`` class.

        Args:
            theta1 (float): The weight we give to return.
            theta2 (float): The weight we give to the penalty, to the constraint of
                not exceeding the budget.
            theta3 (float): The weight we give to covariance, i.e. to diversity.
            price_data (npt.NDArray[np.float64]): At this point in the execution, prices
                are the values of the funds in raw format, without normalizing.
                Shape (m, n) where m is the historical depth of the data and
                n = funds number * slices number.
            num_slices (int): The number of slices is the granularity we are going to
                give to each fund. That is, the amount of the budget that we will be
                able to invest. For example, a 0.5, a 0.25, a 0.125...
        """
        # >>>>>>>>>>>>
        # INPUT VALUES
        # >>>>>>>>>>>>
        self.price_data = price_data
        self.num_slices = num_slices

        b = 1.0  # This is the budget, which is equal to 1 in all cases.

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # WE MAKE THE EXPANSION OF THE PRICES ACCORDING TO THE SLIDES
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # It is at this point that the prices of each fund are normalized, using the
        # last value recorded as a basis. Based on this value and depending on the
        # slides, the rest of the prices are composed as follows.

        expand = ExpandPriceData(b, self.num_slices, self.price_data)

        # Prices in raw format are replaced by prices in standardized format.

        # TODO: Change name to standardized_price. p = n * num_slices
        self.price_data = expand.price_data_expanded  # (m, p)
        self.price_data_reversed = expand.price_data_expanded_reversed  # (m, p)

        # Possible prices, this is actually a list of the proportion of the budget you
        # can invest for each of the funds. For example: 1.0, 0.5, 0.25, 0.125
        # NOTE: We talk about the final possible prices

        self.last_prices = self.price_data[-1, :]  # (n * p, )

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # COMPUTE THE EXPECTED RETURN
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # We obtain the expected return using the prices as a basis.

        # Compute the mean of the daily returns.

        self.expected_returns = get_expected_returns(self.price_data)  # (p, )

        # Obtenemos los valores asociados al riesgo, es decir, la covariance
        qubo_covariance = get_prices_covariance(self.price_data)  # (p, p)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # SHAPING THE VALUES OF THE QUBO
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # We generate a diagonal matrix with the returns, this matrix will be used later
        # with the value of theta1.
        qubo_returns = np.diag(self.expected_returns)  # (p, p)

        # We generate a diagonal matrix with the possible prices * 2. This will be
        # related to the returns.
        qubo_prices_linear = 2.0 * b * np.diag(self.last_prices)  # (p, p)

        # We generate a symmetric matrix also related to the possible prices. This will
        # be related to diversity.
        qubo_prices_quadratic = np.outer(self.last_prices, self.last_prices)  # (p, p)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # FINAL QUBO FORMATION, WITH BIAS AND PENALTY VALUES INCLUDED
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # We form the diagonal values, related to return and prices.
        self.qi = -theta1 * qubo_returns - theta2 * qubo_prices_linear  # (p, p)

        # We now form the quadratic values, related to diversity.
        self.qij = theta2 * qubo_prices_quadratic + theta3 * qubo_covariance  # (p, p)
