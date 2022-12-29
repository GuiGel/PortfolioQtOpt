import numpy as np
import numpy.typing as npt

from portfolioqtopt.expand_prices import get_slices_list


def get_investment(
    dwave_array: npt.NDArray[np.int8],
    slices_nb: int,
) -> npt.NDArray[np.floating]:
    """Get the investment per fund.

    Example:

        >>> dwave_array = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8)
        >>> investment = get_investment(dwave_array, 5)
        >>> investment
        array([0.5 , 0.25, 0.25])

        We can verify that all the budget is invest as expected.
        >>> investment.sum()
        1.0

    Args:
        dwave_array (npt.NDArray[np.int8]): The dwave output array made of 0 and 1.
            Shape (p, )
        slices_nb (int): The number of slices values that determines the granularity.

    Returns:
        npt.NDArray[np.floating]: The total investment for each funds.
            Shape (p / slices_nb).
    """
    qbits = dwave_array.reshape(-1, slices_nb)
    slices = get_slices_list(slices_nb).reshape(1, -1)
    investments: npt.NDArray[np.floating] = (qbits * slices).sum(axis=1)

    total = investments.sum()
    assert (
        total == 1
    ), f"All the budget is not invest! The total investment is {total} in spite of 1.0"

    return investments
