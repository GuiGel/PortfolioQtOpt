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

    # 1. Initialization
    _, m = prices.shape
    chosen_indexes = np.arange(m)
    max_positive_sharpe_ratio = sys.float_info.min
    reduce_prices_dimension: float = True

    for i in range(runs):
        # 2. Reduce prices dimension
        if reduce_prices_dimension:
            prices = prices[:, indexes]
            n, m = prices.shape

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
    print(prices[:, -1])
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
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ).flatten(),  # In this case not a better sharpe ratio. Better with a mock of the sharpe returns!
        np.array(
            [
                [0, 1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        ).flatten(),
    ]

    runs = 4
    indexes = np.array([0, 1, 2, 3])

    def chose_funds_mock(
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

        # 1. Initialization
        n, m = prices.shape
        chosen_indexes = np.arange(m)
        max_positive_sharpe_ratio = sys.float_info.min
        reduce_prices_dimension: float = True

        for i in range(runs):
            print(f"----- {i=} -----")
            # 2. Reduce prices dimension
            if reduce_prices_dimension:
                prices = prices[:, indexes]
                n, m = prices.shape

            # 3. Atomic portfolio optimization
            print(f"Solve atomic portfolio optimization")
            selection = Selection(prices, w, budget)
            # qbits = selection.solve(theta1, theta2, theta3, token, solver)
            qbits = qubits_mock[i]

            # 4. Interpret results
            print(f"compute sharpe ratio")
            sharpe_ratio = get_sharpe_ratio(qbits, selection.npp_rev, prices, w)
            print(f"{sharpe_ratio=}")

            # 5. Select max positive sharpe ratio
            if max_positive_sharpe_ratio < sharpe_ratio:
                print(f"----- new max positive sharpe ratio")

                reduce_prices_dimension = True

                # 6. Record selected fund indexes
                indexes = get_selected_funds_indexes(qbits, w)
                print(f"----- selected indexes: {indexes}")

                # 7. Reduce fund indexes
                chosen_indexes = chosen_indexes[indexes]

                # 8. Update max positive sharpe ratio
                max_positive_sharpe_ratio = sharpe_ratio
            else:
                reduce_prices_dimension = False

        return chosen_indexes

    chosen_funds = chose_funds_mock(
        prices, w, b, 0.1, 0.3, 0.4, "", SolverTypes.hybrid_solver, indexes, runs
    )
    print(f"{chosen_funds=}")
