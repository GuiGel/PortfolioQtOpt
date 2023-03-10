from __future__ import annotations

import typing
from collections import Counter
from enum import Enum, unique

import numpy as np
import pandas as pd
from dimod.sampleset import SampleSet
from dwave.cloud import Client  # type: ignore[import]
from dwave.cloud import exceptions as dwave_exceptions
from dwave.system import LeapHybridSampler  # type: ignore
from loguru import logger

from qoptimiza.assets import Assets
from qoptimiza.optimization.interpreter import (
    Indexes,
    Interpretation,
    Qbits,
    get_investments,
    interpret,
)
from qoptimiza.optimization.qubo import Q, get_qubo


def check_dwave_token(token_api: str) -> bool:
    """Check that the given `token_api` allow a connection to dwave-leap.

    Ref: https://dwave-meta-doc.readthedocs.io/en/latest/overview/dwavesys.html

    Args:
        token_api (str): The connection token

    Returns:
        bool: True for a valid token false otherwise

    Example:

        >>> TOKEN_API = 'ABC-123456789123456789123456789'
        >>> client = Client.from_config(token=TOKEN_API)
        >>> client.get_solvers()
        Traceback (most recent call last):
        ...
        dwave.cloud.exceptions.SolverAuthenticationError: Invalid token or access denied

        Now do the same with :func:`~qoptimiza.application.utils.check_dwave_token`.

        >>> check_dwave_token(TOKEN_API)
        False

        >>> check_dwave_token("xxxxxxxxxxxxxxxxxxxxxxxxxxxx")  # doctest: +SKIP
        True

    """
    client = Client.from_config(
        token=token_api,
        endpoint="https://na-west-1.cloud.dwavesys.com/sapi/v2/",
        client="hybrid",
    )
    try:
        client.get_solvers()
    except dwave_exceptions.SolverAuthenticationError as e:
        logger.exception(e)
        return False
    else:
        return True


@unique
class SolverTypes(Enum):
    """Enumeration of the different solver types

    Example:

        >>> SolverTypes.hybrid_solver
        <SolverTypes.hybrid_solver: 'hybrid_solver'>

        If we want to recuperate the string associated with
        :attr:`SolverTypes.hybrid_solver` just write:

        >>> SolverTypes.hybrid_solver.value
        'hybrid_solver'
    """

    Clique_Embedding = "Clique_Embedding"
    Find_Embedding = "Find_Embedding"
    hybrid_solver = "hybrid_solver"
    "Chose the dwave leap hybrid solver"
    SA = "SA"
    exact = "exact"


def solve_qubo(q: Q, solver: SolverTypes, api_token: str) -> SampleSet:

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
    verbose: bool = False,
) -> Assets:
    """Reduce the universe of possibilities.

    At first the whole problem is executed a number of times equal to the number of
    repetitions introduced as a parameter. Once these first steps have been carried
    out, the universe of funds is reduced to only those in which in some of the runs
    an investment has been made.

    .. note::
        We have decided to introduce Assets as a function argument even if it's not
        strictly needed (only the qubits are needed to obtain the prices indexes)
        because it permit to directly works on the prices columns and thus avoid
        some step between collecting indexes in the selecting assets and the original
        ones in order to link them with the original assets names.

    Args:
        steps (int): The number of repetitions.

    Returns:
        Assets: The new assets made of the selected indexes.
    """
    c: typing.Counter[int] = Counter()
    for step in range(steps):
        logger.debug(f"run solver step {step}")
        qbits = get_qbits(q, solver, token_api)
        _, indexes = get_investments(qbits, w)
        if verbose:
            interpret(assets, qbits)  # Just to log some results
        logger.debug(f"selected indexes {indexes.tolist()}")
        c.update(Counter(indexes))
    distribution = pd.DataFrame.from_dict(
        c, orient="index", columns=["selected X times"]
    ).T
    logger.debug(f"indexes distribution:\n{distribution}")
    outer_indexes = typing.cast(Indexes, np.sort(np.array([k for k in c])))
    logger.info(
        f"create new assets with reduce universe of {len(outer_indexes)} "
        f"assets:\n{outer_indexes}"
    )
    return assets[outer_indexes]


def find_best_sharpe_ratio(
    assets: Assets,
    q: Q,
    steps: int,
    solver: SolverTypes,
    token_api: str,
) -> typing.Tuple[Assets, typing.Optional[Interpretation]]:
    interpretation: typing.Optional[Interpretation] = None
    sharpe_ratio = 0.0

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
        assets = assets[inner_indexes]

    return assets, interpretation


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
    verbose: bool = False,
) -> typing.Tuple[Assets, typing.Optional[Interpretation]]:

    logger.info("compute the qubo")
    q = get_qubo(assets, b, w, theta1, theta2, theta3)

    logger.info("step 1: universe reduction")
    assets = reduce_dimension(assets, q, w, steps, solver, token_api, verbose=verbose)

    logger.info(f"recompute qubo with reduce universe")
    inner_q = get_qubo(assets, b, w, theta1, theta2, theta3)

    logger.info(f"step 2: iterate to find best sharpe ratio")
    assets, interpreter = find_best_sharpe_ratio(
        assets, inner_q, steps, solver, token_api
    )
    return assets, interpreter
