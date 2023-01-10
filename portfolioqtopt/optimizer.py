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

from portfolioqtopt.dwave_solver import SolverTypes
from portfolioqtopt.interpreter import Interpret
from portfolioqtopt.markovitz_portfolio import Selection

Indexes = npt.NDArray[np.signedinteger[typing.Any]]


class Optimize:
    def __init__(
        self,
        selection: Selection,
        theta1: float,
        theta2: float,
        theta3: float,
        token: str,
        solver: SolverTypes,
    ) -> None:
        self.selection = selection
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.token = token
        self.solver = solver

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
            qbits = selection.solve(
                self.theta1, self.theta2, self.theta3, self.token, self.solver
            )
            interpret = Interpret(selection, qbits)
            if not i:
                c = Counter(interpret.selected_indexes)
            else:
                c.update(Counter(interpret.selected_indexes))
        return c

    def optimizer_step(
        self,
        indexes: Indexes,
        sharpe_ratio: float,
        runs: int,
    ) -> typing.Tuple[Indexes, typing.Optional[Interpret],]:

        selection = self.selection[indexes]
        qbits = selection.solve(
            self.theta1, self.theta2, self.theta3, self.token, self.solver
        )
        # qbits = qubits_mock[runs]
        interpret = Interpret(selection, qbits)

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
            indexes, _interpreter = self.optimizer_step(indexes, sharpe_ratio, i)
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


if __name__ == "__main__":
    prices = np.array(
        [
            [100, 104, 102, 104, 100],
            [10, 10.2, 10.4, 10.5, 10.4],
            [50, 51, 52, 52.5, 52],
            [1.0, 1.02, 1.04, 1.05, 1.04],
        ],
        dtype=np.float64,
    ).T
    w, b = 6, 1.0
    selection = Selection(prices, w, b)
    qubits_mock = [
        np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]
        ).flatten(),
        np.array(
            [
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ]
        ).flatten(),
        np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        ).flatten(),  # In this case not a better sharpe ratio. Better with a mock of the sharpe returns!
        np.array(
            [
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
            ]
        ).flatten(),
    ]

    runs = 4
    indexes = np.array([0, 1, 2, 3])

    from portfolioqtopt.markovitz_portfolio import Selection

    selection = Selection(prices, w, b)

    print("-------------" * 10)
    opt = Optimize(selection, 0.1, 0.3, 0.4, "", SolverTypes.hybrid_solver)
    interpreter, indexes = opt.optimize(indexes, 4)
    print("-------------" * 10)
    if interpreter is not None:
        print(interpreter.data, indexes)
