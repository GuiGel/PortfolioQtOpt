import typing
from typing import Dict, Hashable, List, Optional

import numpy as np
import numpy.linalg as la
import pandas as pd
from loguru import logger
from numpy.polynomial import Polynomial as P

from qoptimiza.assets import Array, Assets, Scalar
from qoptimiza.simulation.errors import CovNotSymDefPos


class Simulation:
    """Simulate prices that have daily returns with a covariance matrix identical to 
    that of the daily returns of historical prices and a given expected annual return.

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

        >>> logger.disable("qoptimiza")  # Disable logging messages
        >>> future_assets = simulate(order=5)

        We can observe the simulate prices

        >>> future_assets.df
                    A          B
        0  101.980000  11.320000
        1  102.663143  11.721991
        2  102.325945  11.807491
        3  102.807393  12.160255
        4  101.563481  11.945974
        5  102.489900  12.452001

        Finally we can verified that the results are as expected:

        - Check that the daily returns of the simulated prices are the same as those \
            of the historical prices.  

        >>> np.testing.assert_almost_equal(future_assets.anual_returns, simulate.er)

        -  Check that covariance of the daily returns of the simulated prices are the \
            same as those of the historical prices.  

        >>> np.testing.assert_almost_equal(assets.cov, future_assets.cov)

        If the order is too small the simulation doesn't works as the residual of the 
        Taylor expansion is not small enough. To verify ot we take an order value of 4.

        >>> future_assets = simulate(order=3)
        >>> np.testing.assert_almost_equal(future_assets.anual_returns, simulate.er)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        AssertionError:
        Arrays are not almost equal to 7 decimals

        Mismatched elements: 1 / 2 (50%)
        Max absolute difference: 5.78131155e-05
        Max relative difference: 0.00057813
        x: array([0.0050001, 0.1000578])
        y: array([0.005, 0.1  ])

    Args:
        assets (Assets): An Assets object. Contains the historical prices as a 
            :class:`pd.DataFrame` of shape (n, m) where n is the number days and m the 
            number of assets.
        er (Dict[Hashable, float]): The expected anual returns for each assets. Must be 
            have the same key name and numbers has the `assets.df.columns`.
        ns (int): The number of future daily returns to simulate. Must be strictly 
            positive.

    Attributes:
        assets (Array): An Assets object. Contains the historical prices as a 
            :class:`pd.DataFrame` of shape (n, m) where n is the number days and m the 
            number of assets.
        ns (int): The number of future daily returns to simulate. Must be strictly 
            positive.
    """

    def __init__(
        self,
        assets: Assets,
        er: Dict[typing.Union[Scalar, typing.Tuple[Hashable, ...]], float],
        ns: int,
    ) -> None:
        self.assets = assets
        self._er = er
        self.ns = ns

        assert (len(er), len(er)) == self.assets.cov.shape
        assert ns > 0

    @property
    def er(self) -> Array:
        """The annual expected returns that must be yields at the end of the simulation.

        This attribute is just for verification purpose in order to be sure that the
        order of the values in the resulting array correspond to the same columns as
        input :attr:`Assets.df` columns.

        Returns:
            Array: The anual expected returns as an array. (m,)
        """
        df_er = pd.DataFrame.from_dict(self._er, orient="index").T
        return np.array([df_er[c][0] for c in self.assets.df], np.float64)

    @property
    def init_prices(self) -> Array:
        """The vector of initial prices for the simulation.

        The last prices of the historical prices are taken as initial prices.

        Returns:
            Array: A vector of prices. (m,)
        """
        return self.assets.prices.T[:, -1:]  # (k, 1)

    def _chol(self, C: Optional[Array] = None) -> Array:
        """Choleski decomposition.

        Return the Cholesky decomposition, :math:`L * L^H`, of the square matrix
        :math:`C`, where :math:`L` is lower-triangular and :math:`L^H` is the
        conjugate transpose operator (which is the ordinary transpose if a is
        real-valued). C must be Hermitian (symmetric if real-valued) and
        positive-definite (which is the case if a is the covariance
        matrix of the daily returns). No checking is performed to verify whether a is
        Hermitian or not. In addition, only the lower-triangular and diagonal elements
        of a are used. Only L is actually returned.

        If :math:`C` is not invertible, a :class:`CovNotSymDefPos` exception is raised.

        Args:
            C (Optional[Array], optional): A symmetric positive definite matrix.
                Defaults to None. (m, m)

        Raises:
            CovNotSymDefPos: The matrix C is not symmetric definite positive.

        Returns:
            Array: If a is None, return the Choleski decomposition of
                :attr:`Simulation.asset.cov`. (m, m)
        """
        if C is None:
            C = self.assets.cov
        try:
            L = la.cholesky(C)
        except la.LinAlgError as e:
            raise CovNotSymDefPos(C, e)
        else:
            return L

    def _get_random_unit_cov(self) -> Array:
        """Get random Gaussian vectors with the matrix identity as covariance matrix.

        Returns:
            Array: A numpy array. (m, ns)
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

        Returns:
            Array: A numpy array. (m, ns)
        """
        L = self._chol()
        random_daily_returns = self._get_random_unit_cov()
        daily_returns = L.dot(random_daily_returns)

        # g are daily returns that must all be inferior to 1!
        if np.all(daily_returns >= 1):
            logger.warning("Correlated daily returns not all inf to 1!")

        return typing.cast(Array, daily_returns)

    @staticmethod
    def get_log_taylor_series(cr: Array, er: Array, order: int = 4):  # type: ignore[no-untyped-def]
        """Obtain a polynomial approximation of the expected return.

        For this we use the Taylor expansion of :math:`ln(1+x)` with
        :math:`x = cr + \\alpha`

        Args:
            cr (Array): Matrix of daily returns with the same covariance matrix as the
                historical daily returns. (m, n)
            er (Array): The anual expected returns. Hey must be strictly superior to -1.
                (m,)
            order (int, optional): Order of the polynomial Taylor-Young approximation
                of the :math:`ln` function. Defaults to 4.

        Returns:
            Any: An array of m polynomials. (m,)
        """
        assert np.all(er > -1)
        p = P([0, 1])
        x = np.array(p) + cr  # array of polynomials
        assert order > 1
        lds = x.sum(axis=-1)
        for i in range(2, order):
            lds += (-1) ** (i - 1) / i * (x**i).sum(axis=-1)
        lds -= np.log(1 + er)
        return lds

    @staticmethod
    def get_root(dl: P, min_r: float, max_r: float) -> float:
        """Found a real root of the polynomial :math:`dl` such that \
            :math:`(1 + \\text{real_roots} + min_r > 0)\\space` and :math:`\\space\
            (\\text{real_roots} + max_r < 1)`.

        Args:
            dl (P): A polynomial
            min_r (float): The min value.
            max_r (float): The max value.

        Returns:
            float: The found roof.
        """
        # ------------- compute limited development roots
        roots = P.roots(dl)  # type: ignore[no-untyped-call]

        # ------------- select real roots
        # In our case roots is dim 1 so np.where is a tuple of just one element.
        no_imag = np.imag(roots) == 0
        real_idxs = np.argwhere(no_imag).flatten()
        real_roots = np.real(np.take(roots, real_idxs))
        logger.debug(f"found real roots: {[root for root in real_roots]}")

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
        """For a matrix :math:`Cr` with a desired covariance matrix, found
        a vector :math:`\\alpha` such that the expected anual return of
        :math:`Cr + \\alpha` is the expected one.

        Args:
            cr (Array): Simulated daily returns with the identity matrix as covariance
                matrix. (m, ns)
            order (int, optional): The order of the taylor expansion of the :math:`ln`
                function. The biggest the better the results but expensive to compute.
                Defaults to 10.

        Returns:
            Array: The adjustment vector. (m, 1)
        """
        lds = self.get_log_taylor_series(cr, self.er, order=order)

        min_daily_returns = cr.min(axis=-1)
        max_daily_returns = cr.max(axis=-1)

        alpha: List[float] = []
        for dl, r_min, r_max in zip(lds, min_daily_returns, max_daily_returns):
            logger.trace(f"{dl}")
            logger.debug(f"daily returns in [{r_min:3.2e}, {r_max:3.2g}]")
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

    def check_returns(self, simulated_returns: Array) -> bool:
        # Check that the simulated anual returns are near to the expected ones
        sr = Simulation.get_anual_returns_from_daily_returns(simulated_returns)
        check = np.allclose(sr, self.er)
        if not check:
            logger.warning(
                f"the simulated anual returns are not equal to the expected ones!"
            )
            returns_errors = sr - self.er
            assets_name = self.assets.df.columns.to_list()
            name_returns = dict(zip(assets_name, returns_errors))
            logger.debug(f"anual returns error: {name_returns}")
        return check

    def check_covariance(self, cov_s: Array) -> bool:
        check = np.allclose(cov_s, self.assets.cov)
        if not check:
            logger.warning(
                f"the simulated covariance matrix the not the same as the "
                f"historical one!"
            )
        return check

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
        cov_equal = self.check_covariance(cov_s)

        # ------------- check that the daily returns are correct
        returns_equal = self.check_returns(simulated_returns)

        if cov_equal and returns_equal:
            logger.success("done!")

        # Compute simulated prices
        future_prices = self.get_future_prices(self.init_prices, simulated_returns)

        return Assets(df=pd.DataFrame(future_prices, columns=self.assets.df.columns))


def simulate_assets(
    assets: Assets,
    ns: int,
    er: typing.Optional[
        typing.Dict[typing.Union[Scalar, typing.Tuple[Hashable, ...]], float]
    ] = None,
    order: int = 12,
    seed: Optional[int] = None,
) -> Assets:
    """Function that create the future assets.

    Args:
        assets (Assets): The input assets.
        ns (int): The number of prices to simulate.
        er (typing.Dict[typing.Union[Scalar, typing.Tuple[Hashable, ...]], float]): A
            mapping between each asset name and it's predicted expected returns.
            Defaults to the input expected returns.
        order (int, optional): The order of the polynomial approximation of the
            expected returns. Defaults to 12.
        seed (int, optional): The seed for random number generation.

    Returns:
        Assets: The future assets.
    """
    logger.info("simulate future assets")

    np.random.seed(seed=seed)

    if er is None:
        er = dict(zip(assets.df.columns, assets.anual_returns))
    simulate = Simulation(assets, er, ns)
    return simulate(order=order)


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
