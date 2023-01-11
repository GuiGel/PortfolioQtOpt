import typing
from collections import Counter
from unittest.mock import PropertyMock, patch

import numpy as np
import numpy.typing as npt
import pytest
from loguru import logger

from portfolioqtopt.markovitz_portfolio import Selection
from portfolioqtopt.optimizer import Optimizer, SolverTypes
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

        # Test the portfolioqtopt.optimizer.reduce_dimension methods.
        with patch(
            "portfolioqtopt.optimizer.Interpret.selected_indexes",
            new_callable=PropertyMock,
        ) as mocked_interpreter_selected_indexes:
            mocked_interpreter_selected_indexes.side_effect = selected_indexes
            logger.info(f"{np.array([k for k in expected_counter])=}")
            assert optimizer.reduce_dimension(steps) == expected_counter

    @pytest.mark.parametrize(
        (
            "outer_indexes, inner_indexes, sharpe_ratio, fake_inner_indexes, "
            "fake_sharpe_ratio, expected_outer_indexes, expected_inner_indexes, "
            "interpreter_is_none"
        ),
        [
            ([0, 10, 23], [0, 1, 3], 1.0, [0, 2], 2.0, [0, 23], [0, 2], False),
            ([0, 10, 23], [0, 1, 3], 1.0, [0, 2], 0.5, [0, 10, 23], [0, 1, 3], True),
        ],
    )
    def test_opt_step(
        self,
        qubo_factory,
        outer_indexes: typing.List[int],
        inner_indexes: typing.List[int],
        sharpe_ratio: float,
        fake_inner_indexes: typing.List[int],
        fake_sharpe_ratio: float,
        expected_outer_indexes: typing.List[int],
        expected_inner_indexes: typing.List[int],
        interpreter_is_none: bool,
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
            mocked_interpreter_selected_indexes.return_value = np.array(
                fake_inner_indexes
            )
            mocked_interpreter_sharpe_ratio.return_value = fake_sharpe_ratio

            # test the function with the mock properties
            (
                selected_outer_indexes,
                selected_inner_indexes,
                interpret,
            ) = optimizer._opt_step(
                np.array(outer_indexes), np.array(inner_indexes), sharpe_ratio
            )
            assert selected_outer_indexes.tolist() == expected_outer_indexes
            assert selected_inner_indexes.tolist() == expected_inner_indexes
            assert (interpret is None) == interpreter_is_none


from portfolioqtopt.optimizer import Interpret


class TestInterpret:
    def test_investment(self, qubo_factory):
        optimizer = Optimizer(qubo_factory, "", SolverTypes.hybrid_solver)
        with patch(
            "portfolioqtopt.optimizer.Optimizer.qbits",
            new_callable=PropertyMock,
        ) as mocked_optimizer_qbits:
            mocked_qbits = np.array(
                [
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                ],
            ).flatten()

            mocked_optimizer_qbits.return_value = mocked_qbits
            obtained_investment = Interpret(optimizer).investment
            expected_investment = np.array([0.5, 0.25, 0.125, 0.125])
            np.testing.assert_equal(obtained_investment, expected_investment)


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
