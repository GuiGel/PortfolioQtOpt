from collections import Counter
from unittest.mock import Mock

import numpy as np

from portfolioqtopt.markovitz_portfolio import SolverTypes, dimension_reduction


def test_dimension_reduction():
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
    pre_selected_indexes = dimension_reduction(
        mock, runs, w, 1, 1, 1, "", SolverTypes.hybrid_solver
    )
    assert pre_selected_indexes == expected_indexes
