from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
from loguru import logger
from numpy.polynomial import Polynomial as P
import pandas as pd


from portfolioqtopt.simulation.errors import CovNotSymDefPos
from portfolioqtopt.assets import Assets, Array

# dr: daily returns
# er: expected anual returns
# cov_h: historical daily returns covariance
# chol_h:
# cov_c: correlated covariance
# cov_f: future covariance
# dr: daily returns
# sim_dr


class Simulation:
    """Simulate prices that have the same covariance has the historical prices
    and a given expected anual return.

    Args:
        assets (Stocks): An Assets object. Shape (k, n) where k is the number days and
            n the number of assets.
        er (npt.ArrayLike): The expected anual returns for each assets.
            List of shape (n, ) where n is the number of assets.
        ns (int): The number of future daily returns to simulate.

    Attributes:
        cov_h (Array): Historical prices covariance. Matrix of shape
            (m, m) where m is the number of assets.
    """

    def __init__(self, assets: Assets, er: Dict[str, float], ns: int) -> None:
        # TODO check that er are strictly positives, m > 0 and

        self.k = assets.m
        self.assets = assets
        self.cov_h = assets.cov  # historical covariance
        self.er: Array = np.array(list(er.values()))
        self.ns = ns

        assert len(self.er), len(self.er) == self.cov_h.shape
        assert ns > 0

    @property
    def init_prices(self) -> Array:
        """Initialization price for the simulation.

        The last prices of the historical prices are taken as initial prices.

        Returns:
            Array: A vector of floats.
        """
        return self.assets.prices.T[:, -1:]  # (k, 1)

    def _chol(
        self, a: Optional[Array] = None
    ) -> Array:
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
            a = self.cov_h
        try:
            L = la.cholesky(a)
            return L
        except la.LinAlgError as e:
            raise CovNotSymDefPos(a, e)

    def _get_random_daily_returns(self) -> Array:
        """Get random Gaussian vectors with the matrix identity as covariance matrix.

        TODO This is a function that can be put outside of the class.
        """
        x = np.random.normal(0, 1, size=(self.k, self.ns))
        cov = np.cov(x)
        L = self._chol(cov)
        return np.linalg.inv(L).dot(x)

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
        random_daily_returns = self._get_random_daily_returns()
        daily_returns = L.dot(random_daily_returns)  # (k, m)

        # g are daily returns that must all be inferior to 1!
        if np.all(daily_returns < 1):
            logger.warning("Correlated daily returns not all inf to 1!")

        return daily_returns  # (k, m)

    @staticmethod
    def get_log_taylor_series(
        cr: Array, er: Array, order: int = 4
    ):
        """Obtain a polynomial approximation of the expected return.

        Args:
            cr (Array): Matrix of daily returns with the same \
covariance matrix as the historical daily returns.
            er (Array): The anual expected returns. Hey must be \
strictly superior to -1.
            order (int, optional): Order of the polynomial Taylor-Young approximation \
of the :math:`ln` function. Defaults to 4.

        Returns:
            _type_: _description_
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
        return lds  # (k,)

    @staticmethod
    def get_root(dl: P, min_r: float, max_r: float) -> float:
        # ------------- compute limited development roots
        roots = P.roots(dl)

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
            root = select_roots[root_arg]
        else:
            root = select_roots[0]
        return root

    def get_returns_adjustment(
        self, cr: Array, order: int = 10
    ) -> Array:
        # cr: correlated daily returns
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

        return np.expand_dims(alpha, axis=1)  # (k, 1)

    def get_future_prices(
        self, init_prices: Array, returns: Array
    ) -> Array:
        # init_prices: shape (k, 1)
        # returns: shape (k, m)
        # return: shape (k, m + 1)
        returns_extend = np.concatenate(
            [np.ones((self.k, 1)), (returns + 1).cumprod(axis=1)], axis=1
        )  # (k, m + 1)
        prices = (
            returns_extend * init_prices
        )  # Reconstruct the price from the last prices values. (k, m + 1)
        return prices.T  # (m + 1, k)

    def check_returns(self, simulated_returns):
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
        check = np.allclose(cov_s, self.cov_h)
        logger.debug(
            f"Is the simulated covariance matrix the same as the historical one? {check}"
        )

    @staticmethod
    def get_anual_returns_from_daily_returns(daily_returns: Array) -> Array:
        # daily_returns (k, m)
        # returns: (k,)
        sar = np.exp(np.log(1 + daily_returns).sum(axis=-1)) - 1
        return sar

    def __call__(
        self, order: int = 10, precision: Optional[float] = None
    ) -> Assets:
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

        # future_assets_df = pd.DataFrame(future_prices.T, columns=self.er)
        # future_assets = Stocks(df=future_assets_df)

        return Assets(df=pd.DataFrame(future_prices, columns=self.assets.df.columns))
