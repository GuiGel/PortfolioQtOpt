from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt

from portfolioqtopt.dwave_solver import SolverTypes, solve_dwave_advantage_cubo
from portfolioqtopt.portfolio_selection import PortfolioSelection
from portfolioqtopt.qubo import get_qubo
from portfolioqtopt.reader import read_welzia_stocks_file


class Portfolio:
    def __init__(
        self,
        api_token: str,  # clave acceso leap
        prices: npt.NDArray[np.float64],
        header: List[Any],
        slices_num: int,
        theta1: float = 0.9,
        theta2: float = 0.4,
        theta3: float = 0.1,
        solver_type: SolverTypes = SolverTypes.hybrid_solver,
    ) -> None:
        self.api_token = api_token
        self.prices = prices
        self.header = header
        self.slices_num = slices_num
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.solver_type = solver_type

        # TODO: Argument validation

        assert slices_num > 1

    @classmethod
    def from_welzia(
        cls,
        api_token: str,
        file_name: str,
        sheet: str,
        slices_num: int,
        theta1: float = 0.9,
        theta2: float = 0.4,
        theta3: float = 0.1,
        solver_type: SolverTypes = SolverTypes.hybrid_solver,
    ) -> Portfolio:

        # Load Welzia from excel file as a pd.DataFrame
        prices_df = read_welzia_stocks_file(file_path=file_name, sheet_name=sheet)

        return Portfolio(
            api_token,
            prices_df.to_numpy(),
            prices_df.columns.to_list(),
            slices_num,
            theta1,
            theta2,
            theta3,
            solver_type,
        )

    def __call__(
        self,
    ) -> Tuple[Any, PortfolioSelection, List[Any], npt.NDArray[np.floating]]:
        portfolio_selection = PortfolioSelection(
            self.theta1, self.theta2, self.theta3, self.prices, self.slices_num
        )

        qubo = get_qubo(portfolio_selection.qi, portfolio_selection.qij)

        # Resolution of the problem
        sampleset = solve_dwave_advantage_cubo(qubo, self.solver_type, self.api_token)

        return sampleset.record.sample, portfolio_selection, self.header, self.prices
