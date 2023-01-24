import numpy as np

from portfolioqtopt.markovitz_portfolio import Selection
from portfolioqtopt.optimizer import Interpret

if __name__ == "__main__":

    prices = np.array(
        [
            [100, 104, 102, 104, 100],
            [10, 10.2, 10.4, 10.5, 10.4],
            [50, 51, 52, 52.5, 52],
            [1.0, 1.02, 1.04, 1.05, 1.04],
        ],
        dtype=np.floating,
    ).T
    selection = Selection(prices, 6, 1.0)
    qbits = np.array(
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
    )
    interpret = Interpret(selection, qbits)
    print(interpret.data)
