from portfolioqtopt.optimization._qubo import get_qubo, QuboData, Q

from __future__ import annotations

import itertools as it
import typing
from collections import Counter
from enum import Enum, unique

import numpy as np
import numpy.typing as npt
from dimod.sampleset import SampleSet
from dwave.system import LeapHybridSampler  # type: ignore
from loguru import logger

from portfolioqtopt.optimization._qubo import get_partitions_granularity


@unique
class SolverTypes(Enum):
    Clique_Embedding = "Clique_Embedding"
    Find_Embedding = "Find_Embedding"
    hybrid_solver = "hybrid_solver"
    SA = "SA"
    exact = "exact"


def solve_qubo(
    q: Q, solver: SolverTypes, api_token: str
) -> SampleSet:

    # api_token = 'DEV-d9751cb50bc095c993f55b3255f728d5b2793c36'
    _URL = "https://na-west-1.cloud.dwavesys.com/sapi/v2/"

    if solver.value == "hybrid_solver":
        sampler = LeapHybridSampler(token=api_token, endpoint=_URL)
        sampleset: SampleSet = sampler.sample_qubo(q)
        return sampleset
    else:
        raise ValueError(f"Bad solver. Solver must be hybrid_solver.")


def get_qbits(q: Q, solver: SolverTypes, api_token: str) -> npt.NDArray[np.int8]:
    sampleset = solve_qubo(q, solver, api_token)
    qbits: npt.NDArray[np.int8] = sampleset.record.sample
    return qbits


Indexes = npt.NDArray[np.signedinteger[typing.Any]]


def get_investment(
    qbits: npt.NDArray[np.int8],
    w: int,
) -> npt.NDArray[np.floating[typing.Any]]:
    """Get the investment per fund.

    Example:

        >>> qbits = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], \
dtype=np.int8)
        >>> investment = get_investment(qbits, 5)
        >>> investment
        array([0.5 , 0.25, 0.25])

        We can verify that all the budget is invest as expected.
        >>> investment.sum()
        1.0

    Args:
        qbits (npt.NDArray[np.int8]): The dwave output array made of 0 and 1.
            Shape (p,).
        w (int): The depth of granularity.

    Returns:
        npt.NDArray[np.floating[typing.Any]]: The total investment for each funds.
            Shape (p/w,).
    """
    qbits = qbits.reshape(-1, w)
    slices = get_partitions_granularity(w).reshape(1, -1)
    investments: npt.NDArray[np.floating[typing.Any]] = (qbits * slices).sum(axis=1)

    total = investments.sum()
    assert (
        total == 1
    ), f"All the budget is not invest! The total investment is {total} in spite of 1.0"

    return investments


def get_selected_funds_indexes(
    qbits: npt.NDArray[np.int8], w: int
) -> npt.NDArray[np.signedinteger[typing.Any]]:
    """Get the positional index of the selected funds in the prices array.

    Example:

        >>> qbits = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], \
dtype=np.int8)
        >>> indexes = get_selected_funds_indexes(qbits, 5)
        >>> indexes
        array([0, 1, 2])

    Args:
        qbits (npt.NDArray[np.int8]): The dwave output array made of 0 and 1.
            Shape (p,).
        w (int): The depth of granularity.

    Returns:
        npt.NDArray[np.floating[typing.Any]]: The total investment for each funds.
            Shape (p/w,).
    """
    investments = get_investment(qbits, w)
    selected_funds = investments.nonzero()[0]  # We know that investment is a 1D array
    return selected_funds


def reduce_dimension(
        q: Q,
        w: int,
        steps: int,
        solver: SolverTypes,
        token_api: str,
    ) -> Counter[int]:
        """Reduce the universe of possibilities.

        At first the whole problem is executed a number of times equal to the number of
        repetitions introduced as a parameter. Once these first steps have been carried
        out, the universe of funds is reduced to only those in which in some of the runs
        an investment has been made.

        Args:
            steps (int): The number of repetitions.

        Returns:
            Counter[int]: The selected funds indexes as well as the number of times they
                have been selected.
        """
        logger.info(f"universe reduction")
        c: typing.Counter[int] = Counter()
        for step in range(steps):
            logger.debug(f"run solver step {step}")
            qbits = get_qbits(q, solver, token_api)
            indexes = get_selected_funds_indexes(qbits, w)
            logger.debug(f"selected indexes {indexes}")
            c.update(Counter(indexes))
        logger.info(f"indexes distribution {c}")
        return c

def found_best_sharpe_ration():
    pass