import typing
from collections import Counter
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import numpy.typing as npt
import pytest
from loguru import logger

from portfolioqtopt.interpreter_utils import InterpretData
from portfolioqtopt.markovitz_portfolio import Selection
from portfolioqtopt.optimizer import Indexes, Interpret, Optimizer, SolverTypes
from portfolioqtopt.qubo import QuboFactory


@pytest.fixture(scope="class")
def qubo_factory() -> QuboFactory:
    """Pytest fixture that create a :py:class:`work-on-qubo-class.QuboFactory` object
    that can be used as test argument."""
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
        """Test the :py:meth:`work-on-qubo-class.Optimizer.reduce_dimension`.
        For that we simulate the indexes return by the
        :py:meth:`portfolioqtopt.optimizer.Interpret.selected_indexes` method and
        compare the final output :py:`collections.Counter` with the expected one.

        Args:
            qubo_factory (fixture): A simulated QuboFactory object.
            selected_indexes (typing.Iterable[npt.ArrayLike]): The selected index to
                simulate.
            expected_counter (typing.Counter[int]): The expected output.
        """
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

    @pytest.mark.parametrize(
        "indexes, fake_outer_index, fake_inner_index, fake_sharpe_ratio, fake_interpreter, expected_indexes",
        [
            (
                np.array([0, 1, 2, 3, 4, 5, 6]),
                [
                    np.array([1, 2, 3, 4, 6]),
                    np.array([1, 2, 6]),
                ],  # outer indexes selected by _opt_step
                [
                    np.array([1, 2, 3, 4, 6]),
                    np.array([0, 1, 4]),
                ],  # inner indexes selected by _opt_step
                [1.0, 2.0],  # sharpe ratio
                True,
                np.array([1, 2, 6]),
            ),
            (
                np.array([0, 1, 2, 3, 4, 5, 6]),
                [
                    np.array([1, 2, 4]),
                ],  # outer indexes selected by _opt_step
                [
                    np.array([1, 2, 4]),
                ],  # inner indexes selected by _opt_step
                [-1.0],  # sharpe ratio
                False,
                np.array([0, 1, 2, 3, 4, 5, 6]),
            ),
        ],
    )
    def test_optimize(
        self,
        qubo_factory,
        indexes: Indexes,
        fake_outer_index: typing.Iterable[Indexes],
        fake_inner_index: typing.Iterable[Indexes],
        fake_sharpe_ratio: typing.Sequence[float],
        fake_interpreter: bool,
        expected_indexes: Indexes,
    ) -> None:
        """Test that the Optimizer.optimize method is working as expected.

        This test is only partial due to some difficulties..

        The idea is to be sure that the Optimizer.optimize is able to chose the right
        outer indexes depending of if the interpreter is None o no.
        The only important things that we have to check is if the Interpreter
        returned by the call to Optimizer._opt_step is None or not.
        That's why we have 2 tests.

        Args:
            qubo_factory (fixture): _description_
            indexes (Indexes): The initial indexes parameter of Optimize.optimize
                method.
            fake_outer_index (typing.Iterable[Indexes]): The various _outer_indexes
                returned by the Optimizer.optimize method at each step.
            fake_inner_index (typing.Iterable[Indexes]): The various _inner_indexes
                returned by the Optimizer.optimize method at each step.
            fake_sharpe_ratio (typing.Sequence[float]): The sharpe_ratio compute by
                the Optimizer.optimize method at each step and accessible trough
                the sharpe_ratio attribute of the Interpreter object.
            fake_interpreter (bool): If the fake Interpreter instance returns by the
                method Optimizer._opt_step is None o no.
            expected_indexes (Indexes): The expected indexes.
        """
        optimizer = Optimizer(qubo_factory, "", SolverTypes.hybrid_solver)
        steps = len(fake_sharpe_ratio)

        mocked_interpreter: typing.Optional[typing.Any]
        with patch(
            "portfolioqtopt.optimizer.Optimizer._opt_step"
        ) as mocked_optimizer_opt_step, patch(
            "portfolioqtopt.optimizer.Interpret",
        ) as mocked_interpreter:

            if fake_interpreter:
                # Mock the sharpe ratio property of the Interpreter returned by _opt_step.
                mocked_interpreter = MagicMock()
                mocked_sharpe_ratio = PropertyMock(side_effect=fake_sharpe_ratio)
                type(mocked_interpreter).sharpe_ratio = mocked_sharpe_ratio
            else:
                mocked_interpreter = None

            # Mock the output of the _opt_step returned values
            _opt_side_effect = iter(
                [
                    (o, i, mocked_interpreter)
                    for (o, i) in zip(fake_outer_index, fake_inner_index)
                ]
            )
            mocked_optimizer_opt_step.side_effect = _opt_side_effect

            # Run optimize and see if the results are the expected ones!
            found_indexes, found_interpreter = optimizer.optimize(indexes, steps)
            np.testing.assert_equal(found_indexes, expected_indexes)


@pytest.fixture(scope="class")
def mocked_qbits():
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
        yield mocked_optimizer_qbits


@pytest.mark.usefixtures("mocked_qbits")
class TestInterpret:
    def test_investment(self, qubo_factory):
        optimizer = Optimizer(qubo_factory, "", SolverTypes.hybrid_solver)
        interpret = Interpret(optimizer)
        obtained_investment = interpret.investment
        expected_investment = np.array([0.5, 0.25, 0.125, 0.125])
        np.testing.assert_equal(obtained_investment, expected_investment)

    def test_data(self, qubo_factory):
        optimizer = Optimizer(qubo_factory, "", SolverTypes.hybrid_solver)
        interpret = Interpret(optimizer)
        obtained_data = interpret.data

        # Prepare expected results
        expected_data = InterpretData(
            investment=[0.5, 0.25, 0.125, 0.125],
            expected_returns=[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.010000000000000009,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0050000000000000044,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0050000000000000044,
                0.0,
                0.0,
            ],
            selected_indexes=[0, 1, 2, 3],
            risk=0.9803086248727999,
            sharpe_ratio=2.0401738281752975,
        )
        assert obtained_data == expected_data
