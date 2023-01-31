"""This module defines the :class:`Assets` object.

The :class:`Assets` object has various purposes.

#. Serve for the runtime validation process of the prices.

    * The prices values are all strictly positives.

    * The prices exists and are always numbers.

    * The `pd.DataFrame` of prices have only one column level.

    * The prices are in floating point precision.

#. Compute values that depend only on prices and are used in both simulation and \
optimization.

"""
from __future__ import annotations

import typing
from datetime import date, timedelta
from functools import cached_property

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera as pa
from pydantic import BaseModel, validator

Array = npt.NDArray[np.float64]
"""Custom typing alias used for all the `numpy.ndarray` in the project."""

Scalar = typing.Union[str, bytes, date, timedelta, int, float, complex]

prices_schema = pa.DataFrameSchema(
    {
        (".*",): pa.Column(
            float,
            checks=[
                pa.Check.greater_than(0, ignore_na=False),
            ],
            regex=True,
        ),
    },
    # index=pa.Index(np.dtype("datetime64[ns]"), name="Date"),
)


class Assets(BaseModel):
    """Create the :class:`Assets` class that contains all the relevant information
    about the given portfolio.

    The :class:`Assets` class only attributes related to prices. i.e que se pueden
    deducir solo de los precios de cada asset.

    Attributes:

        df (pd.DataFrame): The prices of each asset of the portfolio to optimize. (n, m)

    """

    df: pd.DataFrame

    @property
    def prices(self) -> Array:
        """Convert :py:attr:`Assets.df` to a numpy array.

        Returns:
            :class:`Array`: The prices of each asset. (n, m)
        """
        return self.df.to_numpy()

    @cached_property
    def m(self) -> int:
        """The number of assets in the portfolio."""
        return self.df.shape[1]

    @cached_property
    def n(self) -> int:
        """The number of days in the historical data for each asset."""
        return self.df.shape[0]

    @validator("df")
    def schema(cls, v):  # type: ignore
        return prices_schema(v)

    @cached_property
    def cov(self) -> Array:
        """The covariance matrix of the daily returns (:attr:`returns`)  of the assets.

        The covariance indicates the level to which two variables vary together.
        We define the sample covariance element :math:`c_{u, v}`between assets
        :math:`u` and :math:`v` as:

        .. math::

            c_{u,v} = \\sum_{i=1}^{n} \\frac{(dr_{u, i} - \\bar dr_{u, i}) \\times{(dr_{v, i} - \\bar dr_{v, i})}}{n - 2}

        where :math:`dr_{u, i}` is the daily return for asset :math:`u` at day
        :math:`i` as defined in :attr:`returns`.

        .. note::

            The covariance of the daily returns are used in the simulation part of the
            project by :class:`portfolioqtopt.simulation.simulation.Simulation`.

        Returns:
            :class:`Array`: The covariance matrix. (m, m)
        """
        return np.cov(self.returns, rowvar=True)

    @cached_property
    def returns(self) -> Array:
        """The daily returns of the assets.

        For an asset :math:`u` we define the daily returns between day :math:`i` and
        day :math:`i+1` for :math:`i` \\in :math:`[1, n]` as :

        .. math::

            dr_{u,i} = \\frac{(p_{u, i+1} - p_{u, i})}{p_{u,i}}


        .. note::

            The daily returns are used in the simulation part of the project.


        Returns:
            :class:`Array`: The daily returns. (m, n-1)
        """
        return self.df.pct_change()[1:].to_numpy().T  # (m, n-1)

    @cached_property
    def normalized_prices(self) -> Array:
        """The normalized prices :math:`a` as defined in :cite:p:`Grant2021`.

        For an asset :math:`u` with a final price :math:`p_{u,n}` , the normalized
        price :math:`a_{u,i}` at the day :math:`i` is:

        .. math::

            a_{u,i} = \\frac{p_{u,i}}{p_{u,n}}


        .. note::

            The normalized prices are used in the optimization part of the project.

        Returns:
            :class:`Array`: A numpy array of normalized prices. (n, m)
        """
        factor = np.divide(1, self.prices[-1, :], dtype=np.float64, casting="safe")
        normalized_prices = self.prices * factor
        return typing.cast(Array, normalized_prices)

    @cached_property
    def average_daily_returns(self) -> Array:
        """The average daily returns for each asset in the portfolio.

        The average daily return :math:`\\bar dr_{u}` of asset :math:`u` is defined by:

        .. math::

            \\bar dr_{u} = \\sum_{i=1}^{n} \\frac{(p_{u, i} - p_{u, i-1})}{p_{u,i-1}}

        .. note::

            We don't use this attribute yet.

        Returns:
            :class:`Array`: The mean of the daily returns. (m,)
        """
        adr = self.returns.mean(axis=1)
        return typing.cast(Array, adr)  # (m,)

    @cached_property
    def normalized_prices_approx(self) -> Array:
        """An approximation of the daily returns for each asset in the portfolio.

        The approximation of the average daily return :math:`\\bar a_{u}` of asset
        :math:`u` as defined in :cite:p:`Grant2021`. is given by:

        .. math::

            \\bar a_{u} = \\frac{(a_{u, n} - a_{u, 0})}{n - 1}

        where :math:`a_{u}` is the normalized prices as defined in
        :attr:`normalized_prices`.

        We can rewrite the average daily returns this way by introducing the price
        :math:`p_{u}` of the asset :math:`u`.

        .. math::

            \\bar a_{u} = \\frac{(p_{u, n} - p_{u, 0})}{p_{u, n} \\times{(n - 1)}}

        .. note::

            :math:`\\bar a_{u}` is pass to the function
            :func:`portfolioqtopt.optimization.qubo.get_qubo`
            that prepare the qubo for the optimization process.

        Returns:
            :class:`Array`: A numpy array of approximate daily returns.
        """
        diff = self.normalized_prices[-1] - self.normalized_prices[0]
        return typing.cast(Array, diff / (self.n - 1))

    @cached_property
    def anual_returns(self) -> Array:
        """The annual returns for each asset of the portfolio.

        The annual return :math:`r_{u}` of asset :math:`u` is defined by:

        .. math::

            r_{u} = \\frac{(p_{u, n} - p_{u, 0})}{p_{u, 0}}

        .. note::

            This attribute is used in the interpretation part
            :func:`portfolioqtopt.optimization.interpreter.interpret` to compute the
            sharpe ratio of the portfolio.


        Returns:
            :class:`Array`: The annual returns. (m,)
        """
        return typing.cast(Array, (self.prices[-1] - self.prices[0]) / self.prices[0])

    @cached_property
    def columns2idx(
        self,
    ) -> typing.Dict[typing.Union[Scalar, typing.Tuple[typing.Hashable, ...]], int]:
        """Return a `dict`  that map each column name of
        :attr:`~portfolioqtopt.assets.Assets.df` to it's positional order.

        Example:

            >>> df = pd.DataFrame(
            ...     [
            ...         [10, 12, 14, 13],
            ...         [21, 24, 23, 22],
            ...         [101, 104, 102, 103],
            ...     ],
            ...     index=["A", "B", "C"],
            ...     dtype=float,
            ... ).T
            >>> assets = Assets(df=df)
            >>> assets.columns2idx
            {'A': 0, 'B': 1, 'C': 2}
        """
        return dict(zip(self.df, range(self.m)))

    @cached_property
    def idx2columns(
        self,
    ) -> typing.Dict[int, typing.Union[Scalar, typing.Tuple[typing.Hashable, ...]]]:
        """Return a `dict`  that map each column position of
        :attr:`~portfolioqtopt.assets.Assets.df` to it's value.

        Example:

            >>> df = pd.DataFrame(
            ...     [
            ...         [10, 12, 14, 13],
            ...         [21, 24, 23, 22],
            ...         [101, 104, 102, 103],
            ...     ],
            ...     index=["A", "B", "C"],
            ...     dtype=float,
            ... ).T
            >>> assets = Assets(df=df)
            >>> assets.idx2columns
            {0: 'A', 1: 'B', 2: 'C'}
        """
        return dict(zip(range(self.m), self.df))

    def __getitem__(self, key: typing.Any) -> Assets:
        """Implement the getitem magic method for :class:`Assets`.      

        Args:
            key (typing.Any): The columns of :attr:`~Assets.df` to choose.

        Returns:
            Assets: An :class:`Assets` instance with the columns corresponding to the
                given keys.

        Raises:
            ValueError: The key type is not treated yet.

        Example:

            Take a DataFrame composed of the prices on 4 days of 3 assets "A", "B" and 
            "C"

            >>> df = pd.DataFrame([[10, 12, 14, 13], [21, 24, 23, 22], \
[101, 104, 102, 103]], index=["A", "B", "C"], dtype=float).T
            >>> df
                  A     B      C
            0  10.0  21.0  101.0
            1  12.0  24.0  104.0
            2  14.0  23.0  102.0
            3  13.0  22.0  103.0

            Create the corresponding :class:`Assets` object.

            >>> assets = Assets(df=df)
  
            Construct a new :class:`Assets` object with only columns "B" and "C".

            Try with a python slice

            >>> a = assets[1: 3]
            >>> a.m
            2

            >>> a.df
                  B      C
            0  21.0  101.0
            1  24.0  104.0
            2  23.0  102.0
            3  22.0  103.0

            Try with a list of columns indexes

            >>> assets[np.array([1, 2])].df
                  B      C
            0  21.0  101.0
            1  24.0  104.0
            2  23.0  102.0
            3  22.0  103.0

            Try with a list of indexes

            >>> assets[1, 2].df
                  B      C
            0  21.0  101.0
            1  24.0  104.0
            2  23.0  102.0
            3  22.0  103.0

            Try directly with the columns names

            >>> assets["B", "C"].df
                  B      C
            0  21.0  101.0
            1  24.0  104.0
            2  23.0  102.0
            3  22.0  103.0

            >>> assets[("B", "C")].df
                  B      C
            0  21.0  101.0
            1  24.0  104.0
            2  23.0  102.0
            3  22.0  103.0

        .. note::

            We use this method in the optimization process 
            :func:`~portfolioqtopt.optimization.optimization.optimize` portfolio just 
            after the universe reduction in order to create a new :class:`Assets`object
            with the selected indexes.

        """
        if isinstance(key, slice):
            df = self.df.iloc[:, key.start : key.stop : key.step]
        elif isinstance(key, np.ndarray):
            # array = self.prices[:, key]
            df = self.df.iloc[:, key]
        elif isinstance(key, pd.Index):
            df = self.df[key]
        elif isinstance(key, tuple):
            if isinstance(key[0], int):
                df = self.df.iloc[:, list(key)]
            else:
                df = self.df[list(key)]
        elif key is None:
            df = self.df
        else:
            raise ValueError(f"The key type {type(key)} is not allow!")

        return Assets(df=df)

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (
            cached_property,
        )  # https://github.com/pydantic/pydantic/issues/2763
