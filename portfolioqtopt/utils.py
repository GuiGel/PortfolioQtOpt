import typing

import numpy as np
import numpy.typing as npt


def get_partitions(w: int) -> npt.NDArray[np.floating[typing.Any]]:
    """Compute the possible proportions of the budget that we can allocate to each fund.

    Example:

    >>> get_partitions(5)
    array([1.    , 0.5   , 0.25  , 0.125 , 0.0625])

    Args:
        w (int): The partitions number that determine the granularity that we are
            going to give to each fund. That is, the amount of the budget we
            will be able to invest.

    Returns:
        npt.NDArray[np.floating[typing.Any]]: List of fraction values.
    """
    return np.power(0.5, np.arange(w))
