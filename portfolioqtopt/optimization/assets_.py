"""Preliminary calculations.

In this module, we calculate all preliminary results that depend only on prices,
granularity depth and budget.
"""
from __future__ import annotations

import typing
from functools import cached_property

import numpy as np
import numpy.typing as npt
from dimod.typing import Bias, Variable

T = typing.TypeVar("T", np.float64, np.float_)

Array = npt.NDArray[np.float64]

Q = typing.Mapping[typing.Tuple[Variable, Variable], Bias]


class Assets:
    def __init__(self, prices: Array) -> None:
        self.prices = prices  # (n, m)
        self.n, self.m = prices.shape

    @cached_property
    def normalized_prices(self) -> Array:
        ":math:`\\bar a`"
        factor = np.divide(1, self.prices[-1, :], dtype=np.float64, casting="safe")
        normalized_prices = self.prices * factor
        return typing.cast(Array, normalized_prices)

    @cached_property
    def average_daily_returns(self) -> Array:
        average_daily_returns = (np.diff(self.prices, axis=0) / self.prices[:-1]).mean(
            axis=0
        )
        return typing.cast(Array, average_daily_returns)  # (m,)

    @cached_property
    def normalized_prices_approx(self) -> Array:
        approximate_average = (
            self.normalized_prices[-1, :] - self.normalized_prices[0, :]
        ) / (self.n - 1)
        return approximate_average

    @cached_property
    def anual_returns(self) -> Array:
        return typing.cast(Array, (self.prices[-1] - self.prices[0]) / self.prices[0])

    def __getitem__(self, key: typing.Any) -> Assets:
        return Assets(self.prices[:, key])
