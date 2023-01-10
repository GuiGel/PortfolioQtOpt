import typing
from collections import Counter
from unittest.mock import PropertyMock, patch

import numpy as np
import numpy.typing as npt
import pytest
from loguru import logger

from portfolioqtopt.markovitz_portfolio import Selection
from portfolioqtopt.optimizer import Indexes, Optimizer, SolverTypes
from portfolioqtopt.qubo import QuboFactory


@pytest.fixture
def qubo_factory() -> QuboFactory:
    prices = np.array(
        [
            [100, 104, 102, 104, 100],
            [10, 10.2, 10.4, 10.5, 10.4],
            [50, 51, 52, 52.5, 52],
            [1.0, 1.02, 1.04, 1.05, 1.04],
        ],
        dtype=np.float64,
    ).T
    selection = Selection(prices, 6, 1.0)
    return QuboFactory(selection, 0.1, 0.2, 0.3)


class TestOptimizer:
    @pytest.mark.parametrize(
        "selected_indexes, expected_counter",
        [
            (
                [[0, 2, 4, 6], [0, 4, 6], [0, 4, 7], [0, 2, 5, 6]],
                Counter({0: 4, 4: 3, 6: 3, 2: 2, 7: 1, 5: 1}),
            )
        ],
    )
    def test_reduce_dimension(
        self,
        qubo_factory,
        selected_indexes: typing.Iterable[npt.ArrayLike],
        expected_counter: typing.Counter[int],
    ) -> None:

        # Prepare test inputs
        selected_indexes = list(map(np.array, selected_indexes))
        steps = len(selected_indexes)

        optimizer = Optimizer(qubo_factory, "", SolverTypes.hybrid_solver)
        with patch(
            "portfolioqtopt.optimizer.Interpret.selected_indexes",
            new_callable=PropertyMock,
        ) as mocked_interpreter_selected_indexes:
            mocked_interpreter_selected_indexes.side_effect = selected_indexes
            assert optimizer.reduce_dimension(steps) == expected_counter

    @pytest.mark.parametrize("indexes, sharpe_ratio", [([0, 1, 2, 3], 1.0)])
    def test_opt_step(
        self, qubo_factory, indexes: Indexes, sharpe_ratio: float
    ) -> None:
        # In the first test we have initial indexes of [0, 1, 2, 3] and a sharpe ratio of 1.0
        # Then we say that the new selected indexes are [0, 1, 3] and the new sharpe ratio is bigger than that the initial one.
        # So the result of Optimizer._opt_step must be [0, 1, 3] and the corresponding interpreter.

        optimizer = Optimizer(qubo_factory, "", SolverTypes.hybrid_solver)
        np.set_printoptions(precision=1)
        logger.info(optimizer.qubo_factory.qubo.matrix.shape)

        with patch(
            "portfolioqtopt.optimizer.Interpret.selected_indexes",
            new_callable=PropertyMock,
        ) as mocked_interpreter_selected_indexes, patch(
            "portfolioqtopt.optimizer.Interpret.sharpe_ratio",
            new_callable=PropertyMock,
        ) as mocked_interpreter_sharpe_ratio:

            # gives a value to the properties of the simulated interpreter
            mocked_interpreter_selected_indexes.return_value = np.array([0, 1, 3])
            mocked_interpreter_sharpe_ratio.return_value = 2.0

            # test the function with the mock properties
            indexes, interpret = optimizer._opt_step(np.array(indexes), sharpe_ratio)
            logger.info(f"Return indexes: {indexes}")
            assert indexes.tolist() == [0, 1, 3]
            assert interpret is not None


"""def test_reduce_dimension():
    runs = 4
    w = 6
    qubits_mock = [
        np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ],
        ),
        np.array(
            [
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ],
        ),
        np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    ]
    expected_indexes = Counter({0: 4, 1: 3, 2: 3, 3: 3})
    mock = Mock()
    mock.solve = Mock(side_effect=qubits_mock)
    pre_selected_indexes = reduce_dimension(
        mock, runs, w, 1, 1, 1, "", SolverTypes.hybrid_solver
    )
    assert pre_selected_indexes == expected_indexes
"""
