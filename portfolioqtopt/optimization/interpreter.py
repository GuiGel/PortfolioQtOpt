"""Module that extract relevant information from the optimization results.

All the functions extract only information from \
:class:`~portfolioqtopt.assets.Assets` and :class:`Qbits`.
"""
import itertools as it
import typing
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

from portfolioqtopt.assets import Array, Assets
from portfolioqtopt.optimization.qubo import get_pw

Qbits = npt.NDArray[np.int8]
"""A typing alias for documentation purpose of an 1D array made of only 0 and 1 
integers."""

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

    Args:
        qbits (:class:`Qbits`): The dwave output array made of 0 and 1. (p,).
        w (int): The depth of granularity.

    Returns:
        :class:`IvtIdx`: A tuple that store the investment for each fund and the
            corresponding indexes.

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
    """
    qbits = qbits.reshape(-1, w)  # type: ignore[assignment]
    pw = get_pw(w).reshape(1, -1)
    investments: Array = (qbits * pw).sum(axis=1)

    indexes = typing.cast(Indexes, investments.nonzero()[0])  # selected indexes

    total = investments.sum()

    assert (
        total == 1
    ), f"All the budget is not invest! The total investment is {total} in spite of 1.0"

    return IvtIdx(investments[indexes], indexes)


@dataclass(eq=False)
class Interpretation:
    """Dataclass that hold the relevant information after the optimization process.

    Example:

        Let's define an :class:`Interpretation` instance.

        >>> interpretation1 = Interpretation(
        ...     selected_indexes=pd.Index(['A', 'C', 'D'], dtype='object'),
        ...     investments=np.array([0.75 , 0.125, 0.125]),
        ...     expected_returns=44.5,
        ...     risk=17.153170260916784,
        ...     sharpe_ratio=2.594272622676201,
        ...     )

        We can verify that both interpretation object are the same or not. For this we
        define an interpretation with a slightly different risk value.

        >>> interpretation2 = Interpretation(
        ...     selected_indexes=pd.Index(['A', 'C', 'D'], dtype='object'),
        ...     investments=np.array([0.75 , 0.125, 0.125]),
        ...     expected_returns=44.5,
        ...     risk=17.153170260916479,
        ...     sharpe_ratio=2.594272622676201,
        ...     )
        >>> interpretation2 == interpretation1
        False

    """

    selected_indexes: pd.Index
    """The funds that have been selected."""
    investments: Array
    """The proportion of the budget allocated to each selected fund."""
    expected_returns: float
    """The expected returns of the selected funds. This is the sum of the annual return
    of each fund weighted by its corresponding investment.
    """
    risk: float
    """The risk of the selected funds."""
    sharpe_ratio: float
    """The sharpe ratio of the selected funds."""

    def __eq__(self, other):
        if not isinstance(other, Interpretation):
            return NotImplemented
        return (
            (
                pd.testing.assert_index_equal(
                    self.selected_indexes, other.selected_indexes
                )
                is None
            )
            and (np.testing.assert_equal(self.investments, other.investments) is None)
            and self.expected_returns == other.expected_returns
            and self.risk == other.risk
            and self.sharpe_ratio == other.sharpe_ratio
        )

    def to_str(self) -> str:
        """Serialize the interpretation as a str to log or write to a file.

        Returns:
            str: An informative str about the interpretation contents.

        Example:

            Let's define an :class:`Interpretation` instance.

            >>> interpretation = Interpretation(
            ...     selected_indexes=pd.Index(['A', 'C', 'D'], dtype='object'),
            ...     investments=np.array([0.75 , 0.125, 0.125]),
            ...     expected_returns=44.5,
            ...     risk=17.153170260916784,
            ...     sharpe_ratio=2.594272622676201,
            ...     )

            Transform the interpretation object as a str to further log.

            >>> output = interpretation.to_str()
            >>> print(output)
            ---------------------------------------------------
                              Interpretation
            ---------------------------------------------------
                      selected funds : investment
                                   A : 0.75
                                   C : 0.125
                                   D : 0.125
                     expected return : 44.5
                                risk : 17.153170260916784
                        sharpe ratio : 2.594272622676201
            ---------------------------------------------------
        """

        output_str = f"{'-':-^51}\n"
        output_str += f"{' Interpretation ':^51}\n"
        output_str += f"{'-':-^51}\n"
        output_str += f"{' selected funds':>24} : {'investment':<24}\n"
        for idx, ivt in it.zip_longest(self.selected_indexes, self.investments):
            output_str += f"{idx:>24} : {ivt:<24}\n"
        output_str += f"{' expected return':>24} : {self.expected_returns:<24}\n"
        output_str += f"{' risk':>24} : {self.risk:<24}\n"
        output_str += f"{' sharpe ratio':>24} : {self.sharpe_ratio:<24}\n"
        output_str += f"{'-':-^51}"
        return output_str


def interpret(assets: Assets, qbits: Qbits) -> Interpretation:
    """Interpret the optimization results.

    Args:
        assets (:class:`~portfolioqtopt.assets.Assets`): The assets.
        qbits (:class:`Qbits`): The qbits resulting of the optimization.

    Returns:
        Interpretation: The interpretation of the optimization process for a given set
            of assets.

    Example:

        Define 5 prices for 4 assets.

        >>> import pandas as pd
        >>> prices = pd.DataFrame(
        ...     [
        ...         [100, 102, 104, 108, 116],
        ...         [10, 10.3, 10.9, 10.27, 10.81],
        ...         [100, 104, 116, 164, 356],
        ...         [100, 101, 102, 103, 104],
        ...     ],
        ...     index=["A", "B", "C", "D"],
        ...     dtype=np.float64
        ...     ).T

        Create an instance of the :class:`~portfolioqtopt.assets.Assets`
        object.

        >>> assets = Assets(df=prices)

        Define some fake qbits made of 0 and 1 for w = 6.

        >>> qbits: Qbits = np.array(
        ...     [
        ...         [0, 1, 1, 0, 0, 0],
        ...         [0, 0, 0, 0, 0, 0],
        ...         [0, 0, 0, 1, 0, 0],
        ...         [0, 0, 0, 1, 0, 0]
        ...     ],
        ...     dtype=np.int8,
        ...     ).flatten()

        Create an :class:`~portfolioqtopt.optimization.interpreter.Interpretation`
        object.

        >>> interpret(assets, qbits)  # doctest: +NORMALIZE_WHITESPACE
        Interpretation(selected_indexes=Index(['A', 'C', 'D'], dtype='object'), investments=array([0.75 , 0.125, 0.125]), expected_returns=44.5, risk=17.153170260916784, sharpe_ratio=2.594272622676201)

    """
    p = len(qbits)
    w = int(p / assets.m)

    investments, selected_indexes = get_investments(qbits, w)

    assets = assets[selected_indexes]  # Reduce the assets to the selected ones.

    deviation: float = ((np.std(assets.prices, axis=0) ** 2) * (investments**2)).sum()

    # Compute the covariance
    indexes = np.triu_indices(assets.m, 1)
    prices_cov = np.cov(assets.prices, rowvar=False)[indexes]
    investments_prod = np.outer(investments, investments)[indexes]
    covariance: float = (investments_prod * prices_cov).sum() * 2

    risk = np.sqrt(deviation + covariance)

    returns = assets.anual_returns * investments
    expected_returns = 100 * returns.sum()
    logger.info(f"{expected_returns=}")

    try:
        sharpe_ratio: float = expected_returns / risk
    except ZeroDivisionError:
        raise

    return Interpretation(
        selected_indexes=assets.df.columns,
        investments=investments,
        expected_returns=expected_returns,
        risk=risk,
        sharpe_ratio=sharpe_ratio,
    )
