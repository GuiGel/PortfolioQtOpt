import typing
from typing import Dict, Hashable, List, Optional

import numpy as np
import numpy.linalg as la
import pandas as pd
from loguru import logger
from numpy.polynomial import Polynomial as P

from portfolioqtopt.assets import Array, Assets
from portfolioqtopt.simulation.errors import CovNotSymDefPos

# dr: daily returns
# er: expected anual returns
# cov_h: historical daily returns covariance
# chol_h:simulation
# cov_c: correlated covariance
# cov_f: future covariance
# dr: daily returns
# sim_dr


class Simulation:
    """Simulate prices that have the same covariance has the historical prices
    and a given expected anual return.

    Example:

        First we create an :class:`Assets` instance.

        >>> import pandas as pd
        >>> df = pd.DataFrame([[101.45, 102.34, 101.98], [10.34, 11.0, 11.32]], \
index=["A", "B"]).T
        >>> assets = Assets(df=df)

        Set a random seed to ensure reproducibility.

        >>> np.random.seed(12)

        Then we create a :class:`Simulation` instance with some expected returns for
        assets "A" and "B" after 5 days that are collected into de er dictionary:

        >>> predicted_expected_returns = {"A": 0.005, "B": 0.1}
        >>> number_of_days = 5
        >>> simulate = Simulation(assets, predicted_expected_returns, number_of_days)

        Finally we simulate the future prices with the `simulate` callable.

        >>> logger.disable("portfolioqtopt")  # Disable logging messages
        >>> future_prices = simulate(order=5)

        We can observe the simulate prices

        >>> future_assets = future_prices
        >>> future_assets.df
                    A          B
        0  101.980000  11.320000
        1  102.663143  11.721991
        2  102.325945  11.807491
        3  102.807393  12.160255
        4  101.563481  11.945974
        5  102.489900  12.452001

        Finally we can verified that the results are as expected:

        - Check that the daily returns of the simulated prices are the same as 
        those of the historical prices.

        >>> np.testing.assert_almost_equal(future_assets.anual_returns, simulate.er)

        -  Check that covariance of the daily returns of the simulated prices are the 
        same as those of the historical prices.

        >>> np.testing.assert_almost_equal(assets.cov, future_assets.cov)


    Args:
        assets (Assets): An Assets object. Shape (k, n) where k is the number days and
            n the number of assets.
        er (Dict[Hashable, float]): The expected anual returns for each assets.
        ns (int): The number of future daily returns to simulate.

    Attributes:
        cov_h (Array): Historical prices covariance. Matrix of shape
            (m, m) where m is the number of assets.
    """

    def __init__(self, assets: Assets, er: Dict[Hashable, float], ns: int) -> None:
        # TODO check that er are strictly positives, m > 0 and

        self.assets = assets
        self._er: Dict[Hashable, float] = er
        self.ns = ns

        assert (len(er), len(er)) == self.assets.cov.shape
        assert ns > 0

    @property
    def er(self) -> Array:
        """The annual expected returns that must be yields at the end of the simulation.

        This attribute is just for verification purpose un order to be sure that the
        order of the values in the resulting array correspond to the same columns as
        input :class:`Assets` `pd.DataFrame` columns.

        Returns:
            Array: The anual expected returns as an array.
        """
        return np.array([self._er[c] for c in self.assets.df.columns], np.float64)

    @property
    def init_prices(self) -> Array:
        """The vector of initial prices for the simulation.

        The last prices of the historical prices are taken as initial prices.

        Returns:
            Array: A vector of prices.
        """
        return self.assets.prices.T[:, -1:]  # (k, 1)

    def _chol(self, a: Optional[Array] = None) -> Array:
        """Choleski decomposition.

        Return the Cholesky decomposition, L * L.H, of the square matrix a, where L is \
lower-triangular and .H is the conjugate transpose operator (which is the ordinary \
transpose if a is real-valued). a must be Hermitian (symmetric if real-valued) and \
positive-definite (which is the case if a is the covariance matrix of the daily returns\
). No checking is performed to verify whether a is Hermitian or not. In addition, only \
the lower-triangular and diagonal elements of a are used. Only L is actually returned.

        Args:
            a (Optional[Array], optional): A Matrix but here a
                symmetric positive matrix. Defaults to None.

        Raises:
            CovNotSymDefPos: The matrix a is not symmetric definite positive.

        Returns:
            Array: If a is None, return the Choleski decomposition of
                :attr:`Simulation.cov_h`
        """
        if a is None:
            a = self.assets.cov
        try:
            L = la.cholesky(a)
            return L
        except la.LinAlgError as e:
            raise CovNotSymDefPos(a, e)

    def _get_random_unit_cov(self) -> Array:
        """Get random Gaussian vectors with the matrix identity as covariance matrix.

        TODO This is a function that can be put outside of the class.
        """
        x = np.random.normal(0, 1, size=(self.assets.m, self.ns))
        cov = np.cov(x)
        L = self._chol(cov)
        x_ = np.linalg.inv(L).dot(x)
        return typing.cast(Array, x_)

    def correlate(self) -> Array:
        """Create random vectors that have a given covariance matrix.

        This method is used to create random daily returns that have the same covariance
        as the assets returns.

        #. Compute the Choleski decomposition of the covariance of the daily returns \
of the assets prices.
        #. Generate random Gaussian vectors that simulate daily returns with an \
Identity covariance matrix.
        #. Simulate daily returns with the same covariance matrix as historical ones.
        """
        L = self._chol()
        random_daily_returns = self._get_random_unit_cov()
        daily_returns = L.dot(random_daily_returns)  # (k, m)

        # g are daily returns that must all be inferior to 1!
        if np.all(daily_returns < 1):
            logger.warning("Correlated daily returns not all inf to 1!")

        return typing.cast(Array, daily_returns)  # (k, m)

    @staticmethod
    def get_log_taylor_series(cr: Array, er: Array, order: int = 4):  # type: ignore[no-untyped-def]
        """Obtain a polynomial approximation of the expected return.

        Args:
            cr (Array): Matrix of daily returns with the same covariance matrix as the
                historical daily returns. (m, n)
            er (Array): The anual expected returns. Hey must be strictly superior to -1.
                (m,)
            order (int, optional): Order of the polynomial Taylor-Young approximation
                of the :math:`ln` function. Defaults to 4.

        Returns:
            Any: An array of polynomials. (m,)
        """
        # cr: correlated daily returns (k, m)
        # er: anual expected returns (k)
        # limited development of the function ln(1+x) with x = rc + alpha
        # Returns array of polynomials of shape (k,)
        # TODO check that Here 1 + expected returns is always be positive!

        assert np.all(er > -1)
        p = P([0, 1])
        x = np.array(p) + cr  # array of polynomials
        assert order > 1
        lds = x.sum(axis=-1)
        for i in range(2, order):
            lds += (-1) ** (i - 1) / i * (x**i).sum(axis=-1)
        lds -= np.log(1 + er)
        logger.info(f"{type(lds)=}")
        return lds

    @staticmethod
    def get_root(dl: P, min_r: float, max_r: float) -> float:
        # ------------- compute limited development roots
        roots = P.roots(dl)  # type: ignore[no-untyped-call]

        # ------------- select real roots
        # In our case roots is dim 1 so np.where is a tuple of just one element.
        no_imag = np.imag(roots) == 0
        real_idxs = np.argwhere(no_imag).flatten()
        real_roots = np.real(np.take(roots, real_idxs))

        # ------------- select the roots that respect the constrains
        w = (1 + real_roots + min_r > 0) & (real_roots + max_r < 1)
        if not np.any(w):
            logger.warning("Not roots respect the constraints!")
        select_roots = real_roots[w]
        if len(select_roots) > 1:  # This permit (ri + root)^n --> 0
            root_arg = np.argmin(select_roots + max_r)
            root: float = select_roots[root_arg]
        else:
            root = select_roots[0]
        return root

    def get_returns_adjustment(self, cr: Array, order: int = 10) -> Array:
        # cr: correlated daily returns. (m, n)
        # order > 2
        lds = self.get_log_taylor_series(cr, self.er, order=order)

        min_daily_returns = cr.min(axis=-1)
        max_daily_returns = cr.max(axis=-1)

        alpha: List[float] = []
        for dl, r_min, r_max in zip(lds, min_daily_returns, max_daily_returns):
            # r_min can be negative
            logger.info(f"{dl=}, {r_min=}, {r_max=}")
            root = self.get_root(dl, r_min, r_max)
            alpha.append(root)  # Todo --> Look for max...

        return np.expand_dims(alpha, axis=1)  # (m, 1)

    def get_future_prices(self, init_prices: Array, returns: Array) -> Array:
        # init_prices: shape (k, 1)
        # returns: shape (k, m)
        # return: shape (k, m + 1)
        returns_extend = np.concatenate(
            [np.ones((self.assets.m, 1)), (returns + 1).cumprod(axis=1)], axis=1
        )  # (k, m + 1)
        prices = (
            returns_extend * init_prices
        )  # Reconstruct the price from the last prices values. (m, n + 1)
        return typing.cast(Array, prices.T)  # (n + 1, m)

    def check_returns(self, simulated_returns: Array) -> None:
        # Check that the simulated anual returns are near to the expected ones
        sr = Simulation.get_anual_returns_from_daily_returns(simulated_returns)
        check = np.allclose(sr, self.er)
        logger.info(
            f"Are the simulated anual returns equal to the expected ones?  {check}"
        )
        if not check:
            returns_errors = sr - self.er
            assets_name = self.assets.df.columns.to_list()
            name_returns = dict(zip(assets_name, returns_errors))
            logger.debug(f"anual returns error: {name_returns}")

    def check_covariance(self, cov_s: Array) -> None:
        check = np.allclose(cov_s, self.assets.cov)
        logger.debug(
            f"Is the simulated covariance matrix the same as the historical one? {check}"
        )

    @staticmethod
    def get_anual_returns_from_daily_returns(daily_returns: Array) -> Array:
        # daily_returns (k, m)
        # returns: (k,)
        sar = np.exp(np.log(1 + daily_returns).sum(axis=-1)) - 1
        return typing.cast(Array, sar)

    def __call__(self, order: int = 10, precision: Optional[float] = None) -> Assets:
        # simulated_returns (k, m)
        # future_price (k, m + 1)

        # ------------- correlate simulated daily returns with historical ones
        correlated_returns = self.correlate()

        # ------------- adjust daily returns to obtain the anual return
        adjustment = self.get_returns_adjustment(correlated_returns, order=order)
        simulated_returns = correlated_returns + adjustment

        # ------------- check that the covariance is correct
        cov_s = np.cov(simulated_returns)
        self.check_covariance(cov_s)

        # ------------- check that the daily returns are correct
        self.check_returns(simulated_returns)

        # Compute simulated prices
        future_prices = self.get_future_prices(self.init_prices, simulated_returns)

        return Assets(df=pd.DataFrame(future_prices, columns=self.assets.df.columns))


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        [[101.45, 102.34, 101.98], [10.34, 11.0, 11.32]], index=["A", "B"]
    ).T
    assets = Assets(df=df)
    simulation = Simulation(assets, {"A": 0.01, "B": 0.1}, 5)

    q_ = simulation._get_random_unit_cov()
    np.testing.assert_almost_equal(np.cov(q_), np.diag(np.ones((q_.shape[0]))))

    q = simulation.correlate()
    np.testing.assert_almost_equal(np.cov(q), simulation.assets.cov)

    assets_f = simulation()
    print(assets_f.anual_returns, simulation.er)
