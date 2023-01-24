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

from portfolioqtopt.optimization.interpreter_utils import (
    InterpretData,
    get_covariance,
    get_deviation,
    get_investment,
    get_returns,
    get_risk,
    get_selected_funds_indexes,
    get_sharpe_ratio,
)
from portfolioqtopt.optimization.qubo import Qubo, QuboFactory


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
    """Class that implement all the optimization steps."""

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
            npt.NDArray[np.int8]: A numpy array of 0 and 1.
        """
        logger.info(f"Solve qubo with {self.solver.value}")
        sampleset = solve_dwave_advantage_cubo(
            self.qubo_factory.qubo, self.solver, self.token
        )
        qbits: npt.NDArray[np.int8] = sampleset.record.sample
        logger.info(f"{qbits=}")
        return qbits

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
        """An atomic optimization step.

        The method uses inner_index to create a new ``pd.DataFrame`` which will contain
        only the prices of the corresponding funds. From this, a new Qubo is calculated
        and optimized. The Sharpe Ratio obtained as a result of the optimization is
        compared to the old one. If the new ratio is higher then the new indexes
        selected during the optimization will be the ones used in the next iteration.
        The outer_indexes correspond to the indexes of the funds selected in the
        ``pd.DataFrame`` containing the prices of the original funds.

        Args:
            outer_indexes (Indexes): Indexes of the original ``pd.DataFrame``.
            inner_indexes (Indexes): Reduced ``pd.DataFrame`` indexes.
            sharpe_ratio (float): The sharpe ratio compute in the previous step.

        Returns:
            typing.Tuple[Indexes, Indexes, typing.Optional[Interpret]]: A tuple
                compounded of the selected indexes of the original ``pd.DataFrame``,
                the selected indexes of the current reduce ``pd.DataFrame`` and the
                corresponding sharpe ratio.
        """
        logger.debug(f"input outer indexes: {outer_indexes}")
        logger.debug(f"input inner indexes: {inner_indexes}")
        self.qubo_factory = self.qubo_factory[inner_indexes]
        interpret = Interpret(self)

        if interpret.sharpe_ratio > sharpe_ratio:
            logger.debug(f"new sharpe ratio bigger than precedent")
            selected_inner_indexes = interpret.selected_indexes
            logger.debug(f"{selected_inner_indexes=}")
            selected_outer_indexes = outer_indexes[selected_inner_indexes]
            sharpe_ratio = interpret.sharpe_ratio
            return selected_outer_indexes, selected_inner_indexes, interpret
        else:
            logger.debug(f"new sharpe ratio smaller than precedent")
            return outer_indexes, inner_indexes, None

    def optimize(
        self, indexes: Indexes, steps: int
    ) -> typing.Tuple[Indexes, typing.Optional[Interpret]]:
        """Run various optimization step.

        At each optimization step, if the
        :py:attr:`portfolio.optimizer.Interpreter.sharpe_ratio` is greater than the
        precedent compute sharpe ratio, we keep in memory the indexes of the funds that
        have produced it.

        The next step, a new Qubo is created with the selected indexes and the process
        is run again.

        At the end we have the indexes that permit to obtain the corresponding funds
        name.

        Args:
            indexes (Indexes): The original indexes. Expected to be 0, 1, 2, 3 if we
                have 4 funds at the beginning of the optimization process.
            steps (int): The number solver calls.

        Returns:
            typing.Tuple[Indexes, typing.Optional[Interpret]]: The chosen indexes as
                well as the corresponding
                :py:class:`portfolioqtopt.optimizer.Interpreter`.
        """
        sharpe_ratio = 0.0
        interpreter: typing.Optional[Interpret] = None
        outer_indexes = (
            inner_indexes
        ) = indexes  # At beginning the inner and outer indexes are the same
        for i in range(steps):

            logger.debug(f"\n----- step {i} -----")
            logger.debug(f"{outer_indexes=}")
            logger.debug(f"{inner_indexes=}")
            logger.debug(f"{sharpe_ratio=}")
            logger.debug(f"run _opt_step")

            _outer_indexes, _inner_indexes, _interpreter = self._opt_step(
                outer_indexes, inner_indexes, sharpe_ratio
            )

            if _interpreter is not None:
                logger.debug(f"_interpreter is not None")
                sharpe_ratio = _interpreter.sharpe_ratio
                outer_indexes = _outer_indexes
                inner_indexes = _inner_indexes
                interpreter = _interpreter
                logger.debug(f"{sharpe_ratio=}")

        return outer_indexes, interpreter

    def __call__(self, steps: int) -> typing.Tuple[Indexes, typing.Optional[Interpret]]:
        c = self.reduce_dimension(steps)
        indexes: Indexes = np.array([k for k in c])
        indexes, interpreter = self.optimize(indexes, steps)
        return indexes, interpreter


class Interpret:
    """Extract all the information of interest after the optimization process."""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    @cached_property
    def investment(self) -> npt.NDArray[np.floating[typing.Any]]:
        return get_investment(
            self.optimizer.qbits, self.optimizer.qubo_factory.selection.w
        )

    @cached_property
    def expected_returns(self) -> npt.NDArray[np.floating[typing.Any]]:
        return get_returns(
            self.optimizer.qbits, self.optimizer.qubo_factory.selection.npp_rev
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

        Returns:
            InterpretData: An :py:class:`portfolioqtopt.interpreter_utils.InterpretData`
                dataclass.
        """
        return InterpretData(
            self.investment.tolist(),
            self.expected_returns.tolist(),
            self.selected_indexes.tolist(),
            self.risk,
            self.sharpe_ratio,
        )


__test__ = {"Interpret.data": Interpret.data}


if __name__ == "__main__":
    import numpy as np
    from loguru import logger

    from portfolioqtopt.optimization.markovitz_portfolio import Selection
    from portfolioqtopt.optimization.optimizer import Optimizer, SolverTypes
    from portfolioqtopt.optimization.qubo import QuboFactory
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
