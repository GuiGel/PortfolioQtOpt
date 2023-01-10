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
import typing
from collections import Counter

import numpy as np
import numpy.typing as npt

from portfolioqtopt.dwave_solver import SolverTypes, solve_dwave_advantage_cubo
from portfolioqtopt.interpreter import Interpret
from portfolioqtopt.qubo import QuboFactory

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
        return self._qubo_factory

    @qubo_factory.setter
    def qubo_factory(self, qubo_factory: QuboFactory) -> None:
        self._qubo_factory = qubo_factory

    @property
    def qbits(self) -> npt.NDArray[np.int8]:
        sampleset = solve_dwave_advantage_cubo(
            self.qubo_factory.qubo, self.solver, self.token
        )
        qubits: npt.NDArray[np.int8] = sampleset.record.sample
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
        c: Counter[int] = Counter()
        for i in range(steps):
            interpret = Interpret(self)
            if not i:
                c = Counter(interpret.selected_indexes)
            else:
                c.update(Counter(interpret.selected_indexes))
        return c

    def _opt_step(
        self,
        indexes: Indexes,
        sharpe_ratio: float,
    ) -> typing.Tuple[Indexes, typing.Optional[Interpret],]:

        self.qubo_factory = self.qubo_factory[indexes]
        interpret = Interpret(self)

        if interpret.sharpe_ratio > sharpe_ratio:
            selected_indexes = interpret.selected_indexes
            indexes = indexes[selected_indexes]
            sharpe_ratio = interpret.sharpe_ratio
            return indexes, interpret
        else:
            return indexes, None

    def optimize(
        self, indexes: Indexes, steps: int
    ) -> typing.Tuple[typing.Optional[Interpret], Indexes]:
        """Look for the best sharpe ration with quantum computing."""
        sharpe_ratio = 0.0
        interpreter: typing.Optional[Interpret] = None
        for i in range(steps):
            print(f"-------------- {i}")
            indexes, _interpreter = self._opt_step(indexes, sharpe_ratio)
            if _interpreter is not None:
                sharpe_ratio = _interpreter.sharpe_ratio
                interpreter = _interpreter
                print(f"{interpreter.data}")
        return interpreter, indexes

    def __call__(self, steps: int) -> typing.Tuple[Indexes, typing.Optional[Interpret]]:
        c = self.reduce_dimension(steps)
        indexes: Indexes = np.array(c.keys())
        interpreter, indexes = self.optimize(indexes, steps)
        return indexes, interpreter
