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
from __future__ import annotations

import itertools as it
import typing
from collections import Counter
from enum import Enum, unique
from functools import cached_property

import numpy as np
import numpy.typing as npt
from dimod.sampleset import SampleSet
from dwave.system import LeapHybridSampler  # type: ignore
from loguru import logger

from portfolioqtopt.interpreter_utils import (InterpretData, get_covariance,
                                              get_deviation, get_investment,
                                              get_risk,
                                              get_selected_funds_indexes,
                                              get_sharpe_ratio)
from portfolioqtopt.qubo import Qubo, QuboFactory


@unique
class SolverTypes(Enum):
    Clique_Embedding = "Clique_Embedding"
    Find_Embedding = "Find_Embedding"
    hybrid_solver = "hybrid_solver"
    SA = "SA"
    exact = "exact"


def solve_dwave_advantage_cubo(
    qubo: Qubo, solver: SolverTypes, api_token: str
) -> SampleSet:

    # api_token = 'DEV-d9751cb50bc095c993f55b3255f728d5b2793c36'
    _URL = "https://na-west-1.cloud.dwavesys.com/sapi/v2/"

    if solver.value == "hybrid_solver":
        sampler = LeapHybridSampler(token=api_token, endpoint=_URL)
        sampleset: SampleSet = sampler.sample_qubo(qubo.dictionary)
        return sampleset
    else:
        raise ValueError(f"Bad solver. Solver must be hybrid_solver.")


Indexes = npt.NDArray[np.signedinteger[typing.Any]]


class Optimizer:
    def __init__(
        self,
        qubo_factory: QuboFactory,
        token: str,
        solver: SolverTypes,
    ) -> None:
        self._qubo_factory = qubo_factory
        self.token = token
        self.solver = solver

    @property
    def qubo_factory(self) -> QuboFactory:
        logger.info("retrieve qubo_factory")
        return self._qubo_factory

    @qubo_factory.setter
    def qubo_factory(self, qubo_factory: QuboFactory) -> None:
        logger.info("set qubo_factory")
        self._qubo_factory = qubo_factory

    @property
    def qbits(self) -> npt.NDArray[np.int8]:
        """Solve the qubo problem.

        Returns:
            npt.NDArray[np.int8]: _description_
        """
        logger.info(f"Solve qubo with {self.solver.value}")
        sampleset = solve_dwave_advantage_cubo(
            self.qubo_factory.qubo, self.solver, self.token
        )
        qubits: npt.NDArray[np.int8] = sampleset.record.sample
        logger.info(f"{qubits=}")
        return qubits

    def reduce_dimension(
        self,
        steps: int,
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
        logger.info(f"reduce the size of the solution space")
        selected_indexes = lambda: Interpret(self).selected_indexes
        c: Counter[int] = Counter(it.chain(*(selected_indexes() for _ in range(steps))))
        logger.info(f"Selected indexes distribution: {c}")
        return c

    def _opt_step(
        self,
        outer_indexes: Indexes,
        inner_indexes: Indexes,
        sharpe_ratio: float,
    ) -> typing.Tuple[Indexes, Indexes, typing.Optional[Interpret]]:
        logger.info(f"input outer indexes: {outer_indexes}")
        logger.info(f"input inner indexes: {inner_indexes}")
        self.qubo_factory = self.qubo_factory[inner_indexes]
        interpret = Interpret(self)

        if interpret.sharpe_ratio > sharpe_ratio:
            logger.info(f"new sharpe ratio bigger than precedent")
            selected_inner_indexes = interpret.selected_indexes
            logger.info(f"{selected_inner_indexes=}")
            outer_indexes = outer_indexes[selected_inner_indexes]
            sharpe_ratio = interpret.sharpe_ratio
            return outer_indexes, selected_inner_indexes, interpret
        else:
            logger.info(f"new sharpe ratio smaller than precedent")
            return outer_indexes, inner_indexes, None

    def optimize(
        self, indexes: Indexes, steps: int
    ) -> typing.Tuple[typing.Optional[Interpret], Indexes]:
        """Iterate to found the best sharpe ratio."""
        sharpe_ratio = 0.0
        interpreter: typing.Optional[Interpret] = None
        inner_indexes = indexes  # At beginning the inner and outer indexes are the same
        for i in range(steps):
            logger.info(f"step {i}")
            indexes, inner_indexes, _interpreter = self._opt_step(
                indexes, inner_indexes, sharpe_ratio
            )
            if _interpreter is not None:
                sharpe_ratio = _interpreter.sharpe_ratio
                interpreter = _interpreter
                print(f"{interpreter.data}")
        return interpreter, indexes

    def __call__(self, steps: int) -> typing.Tuple[Indexes, typing.Optional[Interpret]]:
        c = self.reduce_dimension(steps)
        indexes: Indexes = np.array([k for k in c])
        interpreter, indexes = self.optimize(indexes, steps)
        return indexes, interpreter


class Interpret:
    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    @cached_property
    def investment(self) -> npt.NDArray[np.floating[typing.Any]]:
        return get_investment(
            self.optimizer.qbits, self.optimizer.qubo_factory.selection.w
        )

    @cached_property
    def selected_indexes(self) -> npt.NDArray[np.signedinteger[typing.Any]]:
        logger.info(f"select indexes")
        return get_selected_funds_indexes(
            self.optimizer.qbits, self.optimizer.qubo_factory.selection.w
        )

    def _deviation(self):
        return get_deviation(
            self.investment, self.optimizer.qubo_factory.selection.prices
        )

    def _covariance(self) -> float:
        return get_covariance(
            self.investment, self.optimizer.qubo_factory.selection.prices
        )

    @cached_property
    def risk(self) -> float:
        return get_risk(self.investment, self.optimizer.qubo_factory.selection.prices)

    @cached_property
    def sharpe_ratio(self) -> float:
        return get_sharpe_ratio(
            self.optimizer.qbits,
            self.optimizer.qubo_factory.selection.npp_rev,
            self.optimizer.qubo_factory.selection.prices,
            self.optimizer.qubo_factory.selection.w,
        )

    @cached_property
    def data(self) -> InterpretData:
        """Get the interpretation results as a DataClass.

        Example:
            >>> prices = np.array(
            ...    [
            ...        [100, 104, 102, 104, 100],
            ...        [10, 10.2, 10.4, 10.5, 10.4],
            ...        [50, 51, 52, 52.5, 52],
            ...        [1.0, 1.02, 1.04, 1.05, 1.04],
            ...    ],
            ...    dtype=np.floating,
            ... ).T
            >>> selection = Selection(prices, 6, 1.0)
            >>> qbits = np.array(
            ...     [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
            ... )
            >>> interpret = Interpret(selection, qbits)
            >>> interpret.data  # doctest: +NORMALIZE_WHITESPACE
            InterpretData(investment=[0.5, 0.25, 0.125, 0.125], \
                selected_indexes=[0, 1, 2, 3], risk=0.9803086248727999, \
                sharpe_ratio=2.0401738281752975)

        Returns:
            InterpretData: An InterpretData dataclass.
        """
        return InterpretData(
            self.investment.tolist(),
            self.selected_indexes.tolist(),
            self.risk,
            self.sharpe_ratio,
        )


__test__ = {"Interpret.data": Interpret.data}


if __name__ == "__main__":
    import numpy as np

from loguru import logger

from portfolioqtopt.markovitz_portfolio import Selection
from portfolioqtopt.optimizer import Optimizer, SolverTypes
from portfolioqtopt.qubo import QuboFactory

if __name__ == "__main__":
    from portfolioqtopt.reader import read_welzia_stocks_file

    file_path = "/home/ggelabert/Projects/PortfolioQtOpt/data/Histórico carteras Welzia Completo.xlsm"
    sheet_name = "BBG (valores)"
    df = read_welzia_stocks_file(file_path, sheet_name)
    logger.info(f"{df.shape=}")
    selection = Selection(df.to_numpy(), 5, 1.0)
    qubo_factory = QuboFactory(selection, 0.9, 0.4, 0.1)
    optimize = Optimizer(
        qubo_factory,
        "DEV-d9751cb50bc095c993f55b3255f728d5b2793c36",
        solver=SolverTypes.hybrid_solver,
    )
    indexes, interpret = optimize(3)
    if interpret is not None:
        logger.info(f"{indexes=}")
        logger.info(f"{interpret.data}")
