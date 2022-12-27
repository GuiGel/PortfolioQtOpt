"""Test that ``get_expected_return`` is working as expected."""
import numpy as np
import numpy.typing as npt
import pytest

from portfolioqtopt.ExpectedReturn_calculator import get_expected_returns


@pytest.mark.parametrize(
    "normalized_prices, expected",
    [
        (
            np.array(
                [[1.005, 1.004], [1.002, 1.005], [1.005, 1.006]], dtype=np.float64
            ),
            np.array([0.0, 0.001], dtype=np.float64),
        )
    ],
)
def test_get_expected_return(
    normalized_prices: npt.NDArray[np.float64], expected: npt.NDArray[np.float64]
) -> None:
    expected_returns = get_expected_returns(normalized_prices)
    np.testing.assert_allclose(expected_returns, expected, rtol=1e-15, atol=0)
