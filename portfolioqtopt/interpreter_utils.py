"""Module that extract relevant information from the optimization results"""
import typing

import numpy as np
import numpy.typing as npt

from portfolioqtopt.utils import get_partitions


def get_investment(
    dwave_array: npt.NDArray[np.int8],
    slices_nb: int,
) -> npt.NDArray[np.floating[typing.Any]]:
    """Get the investment per fund.

    Example:

        >>> dwave_array = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], \
            dtype=np.int8)
        >>> investment = get_investment(dwave_array, 5)
        >>> investment
        array([0.5 , 0.25, 0.25])

        We can verify that all the budget is invest as expected.
        >>> investment.sum()
        1.0

    Args:
        dwave_array (npt.NDArray[np.int8]): The dwave output array made of 0 and 1.
            Shape (p,).
        slices_nb (int): The depth of granularity.

    Returns:
        npt.NDArray[np.floating[typing.Any]]: The total investment for each funds.
            Shape (p/slices_nb,).
    """
    qbits = dwave_array.reshape(-1, slices_nb)
    slices = get_partitions(slices_nb).reshape(1, -1)
    investments: npt.NDArray[np.floating[typing.Any]] = (qbits * slices).sum(axis=1)

    total = investments.sum()
    assert (
        total == 1
    ), f"All the budget is not invest! The total investment is {total} in spite of 1.0"

    return investments


def get_deviation(
    investments: npt.NDArray[np.floating[typing.Any]],
    prices: npt.NDArray[np.floating[typing.Any]],
) -> float:
    """Compute the deviation.

    Calculate the sum of the square of the standard deviation of each fund weighted by
    its squared investment.

    Example:

        >>> investments = np.array([0.5, 0.25, 0.25, 0.])
        >>> prices = np.array([\
            [100, 104, 102, 104, 100],\
            [10, 10.2, 10.4, 10.5, 10.4],\
            [50, 51, 52, 52.5, 52],\
            [1., 1.02, 1.04, 1.05, 1.04],\
        ]).T
        >>> get_deviation(investments, prices)
        0.852

    Args:
        investments (npt.NDArray[np.floating[typing.Any]]): The investment for each fund.
            Shape (n,).
        prices (npt.NDArray[np.floating[typing.Any]]): The funds prices. Shape (m, n).

    Returns:
        float: The compute deviation.
    """
    deviation = ((np.std(prices, axis=0) ** 2) * (investments**2)).sum()
    return typing.cast(float, deviation)


def get_covariance(
    investments: npt.NDArray[np.floating[typing.Any]],
    prices: npt.NDArray[np.floating[typing.Any]],
) -> float:
    """Compute the covariances. TODO: Add a better documentation.

    Example:
        >>> investments = np.array([0.5, 0.25, 0.25, 0.])
        >>> prices = np.array([\
            [100, 104, 102, 104, 100],\
            [10, 10.2, 10.4, 10.5, 10.4],\
            [50, 51, 52, 52.5, 52],\
            [1., 1.02, 1.04, 1.05, 1.04],\
        ]).T
        >>> get_covariance(investments, prices)
        0.2499999999999999

    Args:
        investments (npt.NDArray[np.floating[typing.Any]]): The investment for each fund.
            Shape (n,).
        prices (npt.NDArray[np.floating[typing.Any]]): The funds prices. Shape (m, n).

    Returns:The compute covariance.
    """
    n = len(investments)
    index = np.triu_indices(n, 1)
    prices_cov = np.cov(prices, rowvar=False)[index]
    investments_prod = np.outer(investments, investments)[index]
    covariance: float = (investments_prod * prices_cov).sum() * 2
    return covariance


def get_returns(
    dwave_array: npt.NDArray[np.int8],
    data_reversed: npt.NDArray[np.floating[typing.Any]],
) -> npt.NDArray[np.floating[typing.Any]]:
    """Get the final return for each fund weighted by the value of each slice.

    Example:

        >>> from portfolioqtopt.markovitz_portfolio import Selection
        >>> prices = np.array([[100, 50, 10, 5], [10, 5, 1, 0.5]]).T
        >>> dwave_array = np.array([0, 1, 1, 0, 0, 1], dtype=np.int8)
        >>> selection = Selection(prices, w=3, budget=1)
        >>> get_returns(dwave_array, selection.npp_rev)  # doctest: +NORMALIZE_WHITESPACE
        array([-0. , -0.475 , -0.2375, -0. , -0. , -0.2375])

    Args:
        dwave_array (npt.NDArray[np.int8]): The dwave output array made of 0 and 1.
            Shape (p,)
        data_reversed (npt.NDArray[np.floating[typing.Any]]): The sliced prices multiplied by the
            ratio between the budget and the first price. Shape (m, p)

    Returns:
        npt.NDArray[np.floating[typing.Any]]: The expected return for each fund. Shape (p,)
    """
    returns = dwave_array * (data_reversed[-1] - data_reversed[0])  # (p,)
    return typing.cast(npt.NDArray[np.floating[typing.Any]], returns)


def get_risk(
    investments: npt.NDArray[np.floating[typing.Any]],
    prices: npt.NDArray[np.floating[typing.Any]],
) -> float:
    """Compute the Risk.

    Example:

        >>> investments = np.array([0.5, 0.25, 0.25, 0.])
        >>> prices = np.array([\
            [100, 104, 102, 104, 100],\
            [10, 10.2, 10.4, 10.5, 10.4],\
            [50, 51, 52, 52.5, 52],\
            [1., 1.02, 1.04, 1.05, 1.04],\
        ]).T
        >>> get_risk(investments, prices)
        1.0497618777608566

    Args:
        investments (npt.NDArray[np.floating[typing.Any]]): The investment for each fund.
            Shape (n,).
        prices (npt.NDArray[np.floating[typing.Any]]): The funds prices. Shape (m, n).

    Returns:
        float: The compute risk.
    """
    deviation = get_deviation(investments, prices)
    covariance = get_covariance(investments, prices)
    risk = np.sqrt(deviation + covariance)
    return typing.cast(float, risk)


def get_sharpe_ratio(
    dwave_array: npt.NDArray[np.int8],
    data_reversed: npt.NDArray[np.floating[typing.Any]],
    prices: npt.NDArray[np.floating[typing.Any]],
    slices_nb: int,
) -> float:
    """Compute the Sharpe Ratio.

    Example:

            >>> from portfolioqtopt.markovitz_portfolio import Selection
            >>> prices = np.array([[100, 50, 10, 5], [10, 5, 1, 0.5]]).T
            >>> dwave_array = np.array([0, 1, 1, 0, 0, 1], dtype=np.int8)
            >>> slices_nb = 3
            >>> selection = Selection(prices, w=slices_nb, budget=1)
            >>> get_sharpe_ratio(dwave_array, selection.npp_rev, prices, slices_nb)
            -3.1810041773302586

    Args:
        dwave_array (npt.NDArray[np.int8]): The dwave output array made of 0 and 1.
            Shape (p,)
        data_reversed (npt.NDArray[np.floating[typing.Any]]): The sliced prices multiplied by the
            ratio between the budget and the first price. Shape (m, p)
        prices (npt.NDArray[np.floating[typing.Any]]): The funds prices. Shape (m, n).
        slices_nb (int): The depth of granularity.

    Returns:
        float: A float representing the Sharpe Ratio.
    """
    returns = get_returns(dwave_array, data_reversed)
    investments = get_investment(dwave_array, slices_nb)
    risk = get_risk(investments, prices)
    try:
        sharpe_ratio: float = 100 * returns.sum() / risk
    except ZeroDivisionError:
        raise
    return sharpe_ratio


def get_selected_funds_indexes(
    dwave_array: npt.NDArray[np.int8], slices_nb: int
) -> npt.NDArray[np.signedinteger[typing.Any]]:
    """Get the positional index of the selected funds in the prices array.

    Example:

        >>> dwave_array = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], \
            dtype=np.int8)
        >>> indexes = get_selected_funds_indexes(dwave_array, 5)
        >>> indexes
        array([0, 1, 2])

    Args:
        dwave_array (npt.NDArray[np.int8]): The dwave output array made of 0 and 1.
            Shape (p,).
        slices_nb (int): The depth of granularity.

    Returns:
        npt.NDArray[np.floating[typing.Any]]: The total investment for each funds.
            Shape (p/slices_nb,).
    """
    investments = get_investment(dwave_array, slices_nb)
    selected_funds = investments.nonzero()[0]  # We know that investment is a 1D array
    return selected_funds


from dataclasses import dataclass


@dataclass
class InterpretData:
    investment: typing.List[float]
    selected_indexes: typing.List[int]
    risk: float
    sharpe_ratio: float

    def __init__(
        self,
        investment: typing.List[float],
        selected_indexes: typing.List[int],
        risk: float,
        sharpe_ratio: float,
    ) -> None:
        self.investment = investment
        self.selected_indexes = selected_indexes
        self.risk = risk
        self.sharpe_ratio = sharpe_ratio
