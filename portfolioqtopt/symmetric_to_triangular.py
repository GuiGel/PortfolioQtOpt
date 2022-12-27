# coding=utf-8
#                                              Symmetric Matrix to Triangular Matrix
########################################################################################################################
# Lo siguiente clase genera matrices triangulares (más fáciles de incrustar para el D-Wave)
# a partir de matrices simétricas QUBO.
########################################################################################################################
# coding=utf-8
import numpy as np
import numpy.typing as npt


class TriangleGenerator:
    def __init__(self, n, q):
        self.n = n
        self.q = q
        self.i = None
        self.upper_matrix = None
        self.lower_matrix = None

    def upper(self):
        for col in range(0, self.n - 1):
            for row in range(col + 1, self.n):
                self.q[row, col] = 0
        for row in range(0, self.n - 1):
            for col in range(row + 1, self.n):
                self.q[row, col] = 2 * self.q[row, col]
        self.upper_matrix = self.q

    def lower(self):
        for row in range(0, self.n - 1):
            for col in range(row + 1, self.n):
                self.q[row, col] = 0
        for col in range(0, self.n - 1):
            for row in range(col + 1, self.n):
                self.q[row, col] = 2 * self.q[row, col]
        self.lower_matrix = self.q


def get_upper_triangular(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Extract an upper triangular matrix.

    Example:
        >>> a = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
        >>> get_upper_triangular(a)
        array([[1, 4, 6],
               [0, 1, 8],
               [0, 0, 1]])


    Args:
        a (npt.NDArray[np.float64]): A numpy array.

    Returns:
        npt.NDArray[np.float64]: A numpy array.
    """
    return np.triu(a, 1) + np.triu(a, 0)


def get_lower_triangular(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Extract a lower triangular matrix.

    Example:
        >>> a = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
        >>> get_lower_triangular(a)
        array([[1, 0, 0],
               [4, 1, 0],
               [6, 8, 1]])


    Args:
        a (npt.NDArray[np.float64]): A numpy array.

    Returns:
        npt.NDArray[np.float64]: A numpy array.
    """
    return np.tril(a, -1) + np.tril(a, 0)
