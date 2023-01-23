"""Module that extract relevant information from the optimization results.

All the functions extract only information from \
:class:`portfolioqtopt.optimization._qubo.QuboData` and \
:type:`portfolioqtopt.optimization._qbits.Qbits`."""
import typing
from typing import NewType

import numpy as np
import numpy.typing as npt

from portfolioqtopt.optimization.utils import Array, get_partitions_granularity

Qbits = NewType("Qbits", npt.NDArray[np.int8])

Indexes = NewType("Indexes", npt.NDArray[np.signedinteger[typing.Any]])


def get_investments(
    qbits: Qbits,
    w: int,
) -> Array:
    """Get the investment per fund.

    Example:

        >>> qbits = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], \
dtype=np.int8)
        >>> investment = get_investments(qbits, 5)
        >>> investment
        array([0.5, 0.5, 0. ])

        We can verify that all the budget is invest as expected.
        >>> investment.sum()
        1.0

    Args:
        qbits (Qbits): The dwave output array made of 0 and 1. (p,).
        w (int): The depth of granularity.

    Returns:
        npt.NDArray[np.floating[typing.Any]]: The total investment for each funds. (m,).
    """
    qbits = qbits.reshape(-1, w)  # type: ignore[assignment]
    pw = get_partitions_granularity(w).reshape(1, -1)
    investments: Array = (qbits * pw).sum(axis=1)
    total = investments.sum()

    assert (
        total == 1
    ), f"All the budget is not invest! The total investment is {total} in spite of 1.0"

    return investments


def get_selected_funds_indexes(qbits: Qbits, w: int) -> Indexes:
    """Get the positional index of the selected funds in the prices array.

    Example:

        >>> qbits = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], \
dtype=np.int8)
        >>> indexes = get_selected_funds_indexes(qbits, 5)
        >>> indexes
        array([0, 1, 2])

    Args:
        qbits (Qbits): The dwave output array made of 0 and 1. (p,).
        w (int): The depth of granularity.

    Returns:
        npt.NDArray[np.floating[typing.Any]]: The total investment for each funds. (m,).
    """
    investments = get_investments(qbits, w)
    selected_funds = investments.nonzero()[0]  # We know that investment is a 1D array
    return typing.cast(Indexes, selected_funds)


def get_investments_nonzero(investments: Array) -> Array:
    """Get the investment per fund that is not null.

    Example:

        >>> investment = np.array([0.5, 0.5, 0.0])
        >>> get_investments_nonzero(investment)
        array([0.5, 0.5])

    Args:
        qbits (Qbits): The dwave output array made of 0 and 1. (p,).
        w (int): The depth of granularity.

    Returns:
        npt.NDArray[np.floating[typing.Any]]: The total investment for each selected \
funds.
    """
    return investments[investments.nonzero()]


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
        >>> prices.shape
        (5, 4)
        >>> get_deviation(investments, prices)
        0.852

    Args:
        investments (npt.NDArray[np.floating[typing.Any]]): The investment for each \
fund. (m,).
        prices (npt.NDArray[np.floating[typing.Any]]): The funds prices. (n, m).

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
        investments (npt.NDArray[np.floating[typing.Any]]): The investment for each \
fund. (m,).
        prices (npt.NDArray[np.floating[typing.Any]]): The funds prices. Shape (n, m).

    Returns:
        float: The compute covariance.
    """
    n = len(investments)
    index = np.triu_indices(n, 1)
    prices_cov = np.cov(prices, rowvar=False)[index]
    investments_prod = np.outer(investments, investments)[index]
    covariance: float = (investments_prod * prices_cov).sum() * 2
    return covariance


def get_returns(
    qbits: Qbits,
    anual_returns_partitions: npt.NDArray[np.floating[typing.Any]],
) -> npt.NDArray[np.floating[typing.Any]]:
    """Get the final return for each fund weighted by the value of each slice.

    Example:

        >>> qbits = np.array([0, 1, 1, 0, 0, 1], dtype=np.int8)
        >>> arp = np.array([-0.95  , -0.475 , -0.2375, -0.95  , -0.475 , -0.2375])
        >>> get_returns(qbits, arp)  # doctest: +NORMALIZE_WHITESPACE
        array([-0. , -0.475 , -0.2375, -0. , -0. , -0.2375])

    Args:
        qbits (Qbits): The dwave output array made of 0 and 1.
            Shape (p,)
        anual_returns_partitions (npt.NDArray[np.floating[typing.Any]]): The sliced prices multiplied by the
            ratio between the budget and the first price. Shape (m, p)

    Returns:
        npt.NDArray[np.floating[typing.Any]]: The expected return for each fund. Shape (p,)
    """
    return qbits * anual_returns_partitions


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
    qbits: Qbits,
    anual_returns_partitions: npt.NDArray[np.floating[typing.Any]],
    prices: npt.NDArray[np.floating[typing.Any]],
    w: int,
) -> float:
    """Compute the Sharpe Ratio.

    Example:

        >>> prices = np.array([[100, 50, 10, 5], [10, 5, 1, 0.5]]).T
        >>> qbits = np.array([0, 1, 1, 0, 0, 1], dtype=np.int8)
        >>> w = 3
        >>> arp = np.array([-0.95  , -0.475 , -0.2375, -0.95  , -0.475 , -0.2375])
        >>> get_sharpe_ratio(qbits, arp, prices, w)
        -3.1810041773302586

    Args:
        qbits (Qbits): The dwave output array made of 0 and 1.
            Shape (p,)
        anual_returns_partitions (npt.NDArray[np.floating[typing.Any]]): The sliced prices multiplied by the
            ratio between the budget and the first price. Shape (m, p)
        prices (npt.NDArray[np.floating[typing.Any]]): The funds prices. Shape (m, n).
        w (int): The depth of granularity.

    Returns:
        float: A float representing the Sharpe Ratio.
    """
    returns = get_returns(qbits, anual_returns_partitions)
    investments = get_investments(qbits, w)
    risk = get_risk(investments, prices)
    try:
        sharpe_ratio: float = 100 * returns.sum() / risk
    except ZeroDivisionError:
        raise
    return sharpe_ratio
