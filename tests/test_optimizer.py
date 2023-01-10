import typing
from collections import Counter
from unittest.mock import PropertyMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from portfolioqtopt.markovitz_portfolio import Selection
from portfolioqtopt.optimizer import Interpret  # reduce_dimension
from portfolioqtopt.optimizer import Optimizer, SolverTypes
from portfolioqtopt.qubo import QuboFactory


@pytest.fixture
def qubo_factory() -> QuboFactory:
    prices = np.array(
        [
            [
                100,
                104,
                102,
            ],
            [10, 10.2, 10.4],
        ],
        dtype=np.float64,
    ).T
    selection = Selection(prices, 2, 1.0)
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
