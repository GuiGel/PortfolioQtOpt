"""Module that implements the optimization logic.

As we know, due to the state of quantum computers, these computers suffer from problems
of large dimensions, so we have generated a routine within the quantum solver to
automatically select the most interesting funds and make a reduction of the
entire universe. To do this, at first the whole problem is executed a number of times
equal to the number of repetitions introduced as a parameter. Once these first runs
have been carried out, the universe of funds is reduced to only those in which in some
of the runs an investment has been made. After this, the algorithm is run again using
the reduced universe of funds as input. In this way, and with a smaller universe, the
algorithm can arrive at better results with a higher probability and in a more robust
way.
"""
import sys
import typing
from collections import Counter

import numpy as np
import numpy.typing as npt

from portfolioqtopt.dwave_solver import SolverTypes
from portfolioqtopt.interpreter import (get_selected_funds_indexes,
                                        get_sharpe_ratio)
from portfolioqtopt.markovitz_portfolio import Selection


def reduce_dimension(
    selection: Selection,
    runs: int,
    w: int,
    theta1: float,
    theta2: float,
    theta3: float,
    token: str,
    solver: SolverTypes,
) -> Counter[int]:
    """Reduce the entire universe of possibilities.

    At first the whole problem is executed a number of times equal to the number of
    repetitions introduced as a parameter. Once these first runs have been carried out,
    the universe of funds is reduced to only those in which in some of the runs an
    investment has been made.

    Args:
        selection (Selection): A Selection object.
        runs (int): The number of repetitions.
        w (int):  The number of slices is the granularity that we are going to give to
            each fund. That is, the amount of the budget we will be able to invest.
        theta1 (float): First Lagrange multiplier.
        theta2 (float): Second Lagrange multiplier.
        theta3 (float): Third Lagrange multiplier.
        token (str): The D-Wave api token.
        solver (SolverTypes): The chosen solver.

    Returns:
        Counter[int]: The selected funds indexes as well as the number of times they
            have been selected.
    """
    c: Counter[int] = Counter()
    for i in range(runs):
        qbits = selection.solve(theta1, theta2, theta3, token, solver)
        indexes = get_selected_funds_indexes(qbits, w)
        if not i:
            c = Counter(indexes)
        else:
            c.update(Counter(indexes))
    return c


def chose_funds(
    prices: npt.NDArray[np.floating[typing.Any]],
    w: int,
    budget: float,
    theta1: float,
    theta2: float,
    theta3: float,
    token: str,
    solver: SolverTypes,
    indexes: npt.NDArray[np.signedinteger[typing.Any]],
    runs: int,
) -> npt.NDArray[np.signedinteger[typing.Any]]:
    """Look for the best sharpe ration with quantum computing.

    Args:
        prices (npt.NDArray[np.floating[typing.Any]]): The funds prices. Shape (n, m)
        w (int):  The number of slices is the granularity that we are going to give to
            each fund. That is, the amount of the budget we will be able to invest.
        budget (float): The initial budget of the portfolio optimization.
        theta1 (float): First Lagrange multiplier.
        theta2 (float): Second Lagrange multiplier.
        theta3 (float): Third Lagrange multiplier.
        token (str): The D-Wave api token.
        solver (SolverTypes): The chosen solver.
        indexes (npt.NDArray[np.signedinteger[typing.Any]]): The indexes for initial
            dimension reduction. Chose only the funds with the given indexes.
        runs (int): The number of repetitions.

    Returns:
        npt.NDArray[np.signedinteger[typing.Any]]: The chosen indexes.
    """

    # 1. Initialization
    _, m = prices.shape
    chosen_indexes = np.arange(m)

    max_positive_sharpe_ratio = sys.float_info.min

    reduce_prices_dimension: float = True

    for _ in range(runs):
        # 2. Reduce prices dimension
        if reduce_prices_dimension:
            prices = prices[:, indexes]

        # 3. Atomic portfolio optimization
        selection = Selection(prices, w, budget)
        qbits = selection.solve(theta1, theta2, theta3, token, solver)

        # 4. Interpret results
        sharpe_ratio = get_sharpe_ratio(qbits, selection.npp_rev, prices, w)

        # 5. Select max positive sharpe ratio
        if max_positive_sharpe_ratio < sharpe_ratio:

            reduce_prices_dimension = True

            # 6. Record selected fund indexes
            indexes = get_selected_funds_indexes(qbits, w)

            # 7. Reduce fund indexes
            chosen_indexes = chosen_indexes[indexes]

            # 8. Update max positive sharpe ratio
            max_positive_sharpe_ratio = sharpe_ratio
        else:
            reduce_prices_dimension = False

    return chosen_indexes
