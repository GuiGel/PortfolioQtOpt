"""Module that implement the simulation core functionalities."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
from loguru import logger
from numpy.polynomial import Polynomial as P

from portfolioqtopt.simulation.errors import CovNotSymDefPos
from portfolioqtopt.simulation.stocks import Stocks

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
        stock (Stocks): A Stock object. Shape (k, n) where k is the number days and
            n the number of stocks.
        er (npt.ArrayLike): The expected anual returns for each stocks.
            List of shape (n, ) where n is the number of stocks.
        m (int): The number of future daily returns to simulate.

    Attributes:
        cov_h (npt.NDArray): Historical prices covariance. Matrix of shape
            (n, n) where n is the number of stocks.
    """

    def __init__(self, stock: Stocks, er: Dict[str, float], m: int) -> None:
        # TODO check that er are strictly positives, m > 0 and

        self.k, _ = stock.prices.shape
        self.stock = stock
        self.cov_h = stock.cov  # historical covariance
        self.er: npt.NDArray[np.float64] = np.array(list(er.values()))
        self.m = m

        assert len(self.er), len(self.er) == self.cov_h.shape
        assert m > 0

    @property
    def init_prices(self) -> npt.NDArray[np.float64]:
        """Initialization price for the simulation.

        The last prices of the historical prices are taken as initial prices.

        Returns:
            npt.NDArray[np.float64]: A vector of floats.
        """
        return self.stock.prices[:, -1:]  # (k, 1)

    def _chol(
        self, a: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        """Choleski decomposition.

        Return the Cholesky decomposition, L * L.H, of the square matrix a, where L is \
lower-triangular and .H is the conjugate transpose operator (which is the ordinary \
transpose if a is real-valued). a must be Hermitian (symmetric if real-valued) and \
positive-definite (which is the case if a is the covariance matrix of the daily returns\
). No checking is performed to verify whether a is Hermitian or not. In addition, only \
the lower-triangular and diagonal elements of a are used. Only L is actually returned.

        Args:
            a (Optional[npt.NDArray[np.float64]], optional): A Matrix but here a
                symmetric positive matrix. Defaults to None.

        Raises:
            CovNotSymDefPos: The matrix a is not symmetric definite positive.

        Returns:
            npt.NDArray[np.float64]: If a is None, return the Choleski decomposition of
                :attr:`Simulation.cov_h`
        """
        if a is None:
            a = self.cov_h
        try:
            L = la.cholesky(a)
            return L
        except la.LinAlgError as e:
            raise CovNotSymDefPos(a, e)

    def _get_random_daily_returns(self) -> npt.NDArray[np.float64]:
        """Get random Gaussian vectors with the matrix identity as covariance matrix.

        TODO This is a function that can be put outside of the class.
        """
        x = np.random.normal(0, 1, size=(self.k, self.m))
        cov = np.cov(x)
        L = self._chol(cov)
        return np.linalg.inv(L).dot(x)

    def correlate(self) -> npt.NDArray[np.float64]:
        """Create random vectors that have a given covariance matrix.

        This method is used to create random daily returns that have the same covariance
        as the stock returns.

        #. Compute the Choleski decomposition of the covariance of the daily returns \
of the stock prices.
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
        cr: npt.NDArray[np.float64], er: npt.NDArray[np.float64], order: int = 4
    ):
        """Obtain a polynomial approximation of the expected return.

We have yet a matrix of daily returns :math:`Erd_{s}` that have the same covariance \
matrix :math:`\\Sigma` as the one of our historical prices :math:`P`. But the anual expected \
return vector :math:`Er_{s}` that correspond to these simulated daily returns is not \
the same as the original one :math:`Er`.

By using the fact that, if :math:`c \\in `R^n` a vector then \
:math:`\\mathbb{E}[Er_{s} +  c^T]=\\mathbb{E}[Er_{s}]`, we can easily demonstrate that \
:math:`Cov(Er_{s} + c^T)=Cov(Er)`.

Our goal here is to found that vector :math:`c` such that :math:`Er_s + c^T = Er` so \
let's go!

For a given stock :math:`u`, the expected return :math:`r_{u}` between days :math:`0, n` is:

.. math:: r_{u}=\\frac{r_{u,n}-r_{u,0}}{r_{u,0}}

and we can remarque that:

.. math:: 1 + r_{u}=\\prod_{i=1}^n(1 + r_{u,i})

If for all :math:`i` in :math:`[1,n]` we have :math:`r_{u,i} > -1` then:

.. math:: ln(1 + r_{u})=\\sum_{k=1}^n{ln(1+r_{u,i})} 

.. note::
    The *Taylor* theorem :

    #. :math:`I` a subset of :math:`R`;
    #. :math:`a` in :math:`I`;
    #. :math:`E` a real normed vector space;
    #. :math:`f` a function of :math:`I` in :math:`E` derivable in :math:`a` up to a certain order :math:`n\\geq 1`.  

    Then for any real number :math:`x` belonging to :math:`I` , we have the Taylor-Young formula:

    .. math:: f(a+h)=\\sum_{k=0}^n{\\frac{f^{(k)}(a)}{k!}h^{k}+R_{n}(a+h)}

    where the remaining :math:`R_{n}(x)` is a negligible function with respect to :math:`(x-a)^{n}` in the neighbourhood of :math:`a`.

If we apply to the *Taylor theorem* to the logarithm function in 1 we have for all :math:`x > 0`:   

.. math:: ln(1+x)=\\sum_{k=1}^{n} {(-1)^{k-1}\\frac{x^{k}}{k}}+ R_{n}(1+x)

If :math:`x < 1` then the *Taylor-Young* formula stand that we have:

.. math:: R_{n}(1 + x)=o(x^{n})

In our particular case, we know that the daily returns :math:`r_{u, i}` are strictly \
less than 1 for all :math:`i` and :math:`u`. We can therefore always find a strictly \
positive integer :math:`n` such that :math:`ln(1+r_{u,i})` is approximated with a great \
accuracy by is corresponding polynomial *Taylor-Young* approximation. 

.. math::

    \\lim_{n \\to +\\infty}ln(1 + r_{u,i}) - \\sum_{k=1}^{n}{(-1)^{k-1}\\frac{r_{u,i}^{k}}{k}} = 0


So for a given stock :math:`u` with  :math:`m` daily returns, and :math:`r_{u, i}` a \
daily return at day :math:`i`, we can try to found the constant :math:`c_{u}` such \
that for a simulated daily return :math:`rs_{u,i}` we have:

.. math:: \\sum_{i=1}^m{ln(1 + rs_{u,i} + c_{u})} = ln(1 + r_{u})

To solve this equation we will use the *Taylor-Young* approximation to create the polynomial :math:`P_{u}(X)`:

.. math:: P_{u}(X) = \\sum_{i=1}^{m} \\sum_{k=1}^n(-1)^{k-1}{\\frac{(rs_{u,i} + X)^{k}}{k}} - ln(1 + r_{u})

We can find :math:`c_{u}` as a real root of :math:`P_{u}` such that \
:math:`| \\underset{1 \\leq i \\leq n}{max}(rs_{u,i}) + c_{u} | < 1` this is the \
condition to have:

.. math:: + o((rs_{u,i} + c_{u})^{n})

For that we solve the polynomial P.


        Args:
            cr (npt.NDArray[np.float64]): Matrix of daily returns with the same \
covariance matrix as the historical daily returns.
            er (npt.NDArray[np.float64]): The anual expected returns. Hey must be \
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

    def get_root(self, dl: P, min_r: float, max_r: float) -> float:
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
        self, cr: npt.NDArray[np.float64], order: int = 10
    ) -> npt.NDArray[np.float64]:
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
        self, init_prices: npt.NDArray, returns: npt.NDArray
    ) -> npt.NDArray:
        # init_prices: shape (k, 1)
        # returns: shape (k, m)
        # return: shape (k, m + 1)
        returns_extend = np.concatenate(
            [np.ones((self.k, 1)), (returns + 1).cumprod(axis=1)], axis=1
        )  # (k, m + 1)
        prices = (
            returns_extend * init_prices
        )  # Reconstruct the price from the last prices values. (k, m + 1)
        return prices  # (k, m + 1)

    def check_returns(self, simulated_returns):
        # Check that the simulated anual returns are near to the expected ones
        sr = Simulation.get_anual_returns_from_daily_returns(simulated_returns)
        check = np.allclose(sr, self.er)
        logger.info(
            f"Are the simulated anual returns equal to the expected ones?  {check}"
        )
        if not check:
            returns_errors = sr - self.er
            stocks_name = self.stock.df.columns.to_list()
            name_returns = dict(zip(stocks_name, returns_errors))
            logger.debug(f"anual returns error: {name_returns}")

    def check_covariance(self, cov_s: npt.NDArray) -> None:
        check = np.allclose(cov_s, self.cov_h)
        logger.debug(
            f"Is the simulated covariance matrix the same as the historical one? {check}"
        )

    @staticmethod
    def get_anual_returns_from_daily_returns(daily_returns: npt.NDArray) -> npt.NDArray:
        # daily_returns (k, m)
        # returns: (k,)
        sar = np.exp(np.log(1 + daily_returns).sum(axis=-1)) - 1
        return sar

    def __call__(
        self, order: int = 10, precision: Optional[float] = None
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
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

        # future_stocks_df = pd.DataFrame(future_prices.T, columns=self.er)
        # future_stocks = Stocks(df=future_stocks_df)

        return simulated_returns, cov_s, future_prices
