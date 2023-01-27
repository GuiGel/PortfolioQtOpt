from __future__ import annotations

import typing
from collections import Counter
from enum import Enum, unique

import numpy as np
import pandas as pd
from dimod.sampleset import SampleSet
from dwave.system import LeapHybridSampler  # type: ignore
from loguru import logger

from portfolioqtopt.assets_ import Assets
from portfolioqtopt.optimization.interpreter_ import (
    Indexes,
    Interpretation,
    Qbits,
    get_investments,
    interpret,
)
from portfolioqtopt.optimization.qubo_ import Q, get_qubo


@unique
class SolverTypes(Enum):
    Clique_Embedding = "Clique_Embedding"
    Find_Embedding = "Find_Embedding"
    hybrid_solver = "hybrid_solver"
    SA = "SA"
    exact = "exact"


def solve_qubo(q: Q, solver: SolverTypes, api_token: str) -> SampleSet:

    # api_token = 'DEV-d9751cb50bc095c993f55b3255f728d5b2793c36'
    _URL = "https://na-west-1.cloud.dwavesys.com/sapi/v2/"

    if solver.value == "hybrid_solver":
        sampler = LeapHybridSampler(token=api_token, endpoint=_URL)
        sampleset: SampleSet = sampler.sample_qubo(q)
        return sampleset
    else:
        raise ValueError(f"Bad solver. Solver must be hybrid_solver.")


def get_qbits(q: Q, solver: SolverTypes, api_token: str) -> Qbits:
    sampleset = solve_qubo(q, solver, api_token)
    qbits: Qbits = sampleset.record.sample[0]
    logger.trace(f"qubits {qbits}")
    logger.trace(f"qbits shape {qbits.shape}")
    return qbits


def reduce_dimension(
    assets: Assets,
    q: Q,
    w: int,
    steps: int,
    solver: SolverTypes,
    token_api: str,
) -> Indexes:
    """Reduce the universe of possibilities.

    At first the whole problem is executed a number of times equal to the number of
    repetitions introduced as a parameter. Once these first steps have been carried
    out, the universe of funds is reduced to only those in which in some of the runs
    an investment has been made.

    Args:
        steps (int): The number of repetitions.

    Returns:
        Indexes: The selected funds indexes.
    """
    logger.info(f"universe reduction")
    c: typing.Counter[int] = Counter()
    for step in range(steps):
        logger.debug(f"run solver step {step}")
        qbits = get_qbits(q, solver, token_api)
        logger.info(f"{qbits.shape=}")
        _, indexes = get_investments(qbits, w)
        interpret(assets, qbits)
        logger.debug(f"selected indexes {indexes.tolist()}")
        c.update(Counter(indexes))
    distribution = pd.DataFrame.from_dict(
        c, orient="index", columns=["selected X times"]
    ).T
    logger.debug(f"indexes distribution:\n{distribution}")
    outer_indexes = typing.cast(Indexes, np.sort(np.array([k for k in c])))
    return outer_indexes


def find_best_sharpe_ratio(
    assets: Assets,
    q: Q,
    indexes: Indexes,
    steps: int,
    solver: SolverTypes,
    token_api: str,
):
    interpretation: typing.Optional[Interpretation] = None
    sharpe_ratio = 0.0
    outer_indexes = inner_indexes = indexes

    for step in range(steps):
        logger.debug(f"run solver step {step}")
        qbits = get_qbits(q, solver, token_api)
        interpretation_ = interpret(assets, qbits)
        if interpretation_.sharpe_ratio > sharpe_ratio:
            logger.debug(
                f"Improve sharpe ratio {sharpe_ratio} -> {interpretation_.sharpe_ratio}."
            )
            sharpe_ratio = interpretation_.sharpe_ratio
            interpretation = interpretation_
        else:
            logger.debug(
                f"No improvement sharpe ratio {sharpe_ratio} -> {interpretation_.sharpe_ratio}."
            )

    if interpretation is not None:
        inner_indexes = interpretation.selected_indexes
        outer_indexes = outer_indexes[inner_indexes]  # type: ignore[assignment]
        logger.info(f"final outer indexes {outer_indexes}")

    return typing.cast(Indexes, outer_indexes), interpretation


def optimize(
    assets: Assets,
    b: float,
    w: int,
    theta1: float,
    theta2: float,
    theta3: float,
    solver: SolverTypes,
    token_api: str,
    steps: int,
) -> typing.Tuple[Indexes, typing.Optional[Interpretation]]:

    q = get_qubo(assets, b, w, theta1, theta2, theta3)
    outer_indexes = reduce_dimension(assets, q, w, steps, solver, token_api)

    logger.info(
        f"create new assets with reduce universe of {len(outer_indexes)} "
        f"assets:\n{outer_indexes}"
    )
    assets = assets[outer_indexes]

    logger.info(f"recompute qubo")
    inner_qubo = get_qubo(assets, b, w, theta1, theta2, theta3)

    logger.info(f"iterate to find best sharpe ratio")
    outer_indexes, interpreter = find_best_sharpe_ratio(
        assets, inner_qubo, outer_indexes, steps, solver, token_api
    )
    return outer_indexes, interpreter
