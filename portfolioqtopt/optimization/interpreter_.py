"""Module that extract relevant information from the optimization results.

All the functions extract only information from \
:py:class:`portfolioqtopt.optimization._qubo.QuboData` and \
:py:class:`portfolioqtopt.optimization._qbits.Qbits`."""
import typing
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from loguru import logger

from portfolioqtopt.optimization.assets_ import Array, Assets
from portfolioqtopt.optimization.qubo_ import get_pw

Qbits = npt.NDArray[np.int8]
Indexes = npt.NDArray[np.signedinteger[typing.Any]]


class IvtIdx(typing.NamedTuple):
    """Named Tuple that stores the indexes of the selected assets and their \
corresponding investment.
    """

    investment: Array
    indexes: Indexes


def get_investments(
    qbits: Qbits,
    w: int,
) -> IvtIdx:
    """Get the investment per selected funds and their indexes.

    Example:

        >>> qbits = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], \
dtype=np.int8)
        >>> ivt_idx = get_investments(qbits, 5)
        >>> ivt_idx.investment
        array([0.5, 0.5])

        We can verify that all the budget is invest as expected.
        >>> ivt_idx.investment.sum()
        1.0

        We have the corresponding selected indexes
        >>> ivt_idx.indexes
        array([0, 1])

    Args:
        qbits (Qbits): The dwave output array made of 0 and 1. (p,).
        w (int): The depth of granularity.

    Returns:
        npt.NDArray[np.floating[typing.Any]]: The total investment for each funds. (m,).
    """
    qbits = qbits.reshape(-1, w)  # type: ignore[assignment]
    pw = get_pw(w).reshape(1, -1)
    investments: Array = (qbits * pw).sum(axis=1)
    logger.debug(f"{investments.tolist()=}")

    indexes = typing.cast(Indexes, investments.nonzero()[0])  # selected indexes

    total = investments.sum()

    assert (
        total == 1
    ), f"All the budget is not invest! The total investment is {total} in spite of 1.0"

    return IvtIdx(investments[indexes], indexes)


@dataclass(eq=False)
class Interpretation:
    selected_indexes: Indexes
    investments: Array
    expected_returns: float
    risk: float
    sharpe_ratio: float

    def __eq__(self, other):
        if not isinstance(other, Interpretation):
            return NotImplemented
        return (
            (
                np.testing.assert_equal(self.selected_indexes, other.selected_indexes)
                is None
            )
            and (np.testing.assert_equal(self.investments, other.investments) is None)
            and self.expected_returns == other.expected_returns
            and self.risk == other.risk
            and self.sharpe_ratio == other.sharpe_ratio
        )


def interpret(a: Assets, qbits: Qbits) -> Interpretation:
    """Interpret the optimization results.

    Example:

        Define 5 prices for 4 assets.
 
        >>> prices = np.array([\
[100, 102, 104, 108, 116],\
[10, 10.3, 10.9, 10.27, 10.81],\
[100, 104, 116, 164, 356],\
[100, 101, 102, 103, 104],\
], dtype=np.float64).T

        Create an instance of the :py:class:`portfolioqtopt.optimization.assets_.Assets`
        object.

        >>> assets = Assets(prices)

        Define some fake qbits made of 0 and 1 for w = 6.

        >>> qbits: Qbits = np.array([[0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], \
[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=np.int8).flatten()

        Create an :class:`portfolioqtopt.optimization.interpreter_.Interpretation` object.

        >>> interpret(assets, qbits)
        Interpretation(selected_indexes=array([0, 2, 3]), \
investments=array([0.75 , 0.125, 0.125]), expected_returns=44.5, \
risk=17.153170260916784, sharpe_ratio=2.594272622676201)

    Args:
        a (Assets): The assets.
        qbits (Qbits): The qbits resulting of the optimization.

    Returns:
        Interpretation: The interpretation of the optimization process for a given set \
of assets.
    """
    p = len(qbits)
    w = int(p / a.m)

    investments, selected_indexes = get_investments(qbits, w)

    a = a[selected_indexes]  # Reduce the assets to the selected ones.

    deviation = ((np.std(a.prices, axis=0) ** 2) * (investments**2)).sum()

    # Compute the covariance
    indexes = np.triu_indices(a.m, 1)
    prices_cov = np.cov(a.prices, rowvar=False)[indexes]
    investments_prod = np.outer(investments, investments)[indexes]
    covariance: float = (investments_prod * prices_cov).sum() * 2

    risk = np.sqrt(deviation + covariance)

    returns = a.anual_returns * investments
    expected_returns = 100 * returns.sum()
    logger.info(f"{expected_returns=}")

    try:
        sharpe_ratio: float = expected_returns / risk
    except ZeroDivisionError:
        raise

    return Interpretation(
        selected_indexes=selected_indexes,
        investments=investments,
        expected_returns=expected_returns,
        risk=risk,
        sharpe_ratio=sharpe_ratio,
    )
