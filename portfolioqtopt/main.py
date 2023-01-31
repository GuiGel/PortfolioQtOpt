import os
import time
from pathlib import Path
from typing import Dict, Hashable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from portfolioqtopt.assets import Assets
from portfolioqtopt.optimization.interpreter import Interpretation
from portfolioqtopt.optimization.optimization_ import SolverTypes, optimize
from portfolioqtopt.reader import read_welzia_stocks_file
from portfolioqtopt.simulation.simulation import Scalar, simulate_assets

# Loading environment variables
load_dotenv()

# logger.add(f"logs/{round(time.time() * 1000)}.log", level="DEBUG")


def store_results(funds: pd.Index, interpretation: Interpretation) -> None:
    final_results = {
        "selected funds": funds,
        "investment": interpretation.investments,
        "expected return": interpretation.expected_returns,
        "risk": interpretation.risk,
        "sharpe_ratio": interpretation.sharpe_ratio,
    }
    curr_time = round(time.time() * 1000)
    logger.info(f"{curr_time=}")
    logger.success(f"{final_results=}")


def main(
    file_path: Union[Path, str],
    sheet_name: str,
    ns: int,
    w: int,
    steps: int,
    expected_returns: Optional[Dict[Union[Scalar, Tuple[Hashable, ...]], float]] = None,
    budget: Optional[float] = None,
    theta1: Optional[float] = None,
    theta2: Optional[float] = None,
    theta3: Optional[float] = None,
    solver: Optional[SolverTypes] = None,
    token_api: Optional[str] = None,
    seed: Optional[int] = None,
):
    """Markovitz Portfolio Optimization with simulated datas and quantum computing.

    Args:
        file_path (Union[Path, str]): Path to xlsx file.
        sheet_name (str): Name of the xlsx sheet to read.
        ns (int): Number of prices to simulate.
        w (int): The granularity depth.
        steps (int): The number of step to run for universe reduction and sharpe ratio
            discovery.
        expected_returns (Optional[Dict[Union[Scalar, Tuple[Hashable, ...]], float]], \
            optional): The predicted expected returns. Defaults to None.
        budget (Optional[float], optional): The budget to allocate. Defaults to None.
        theta1 (Optional[float], optional): The optimization first Lagrange multiplier.
            Defaults to None.
        theta2 (Optional[float], optional): The optimization second Lagrange multiplier.
            Defaults to None.
        theta3 (Optional[float], optional): The optimization third Lagrange multiplier.
            Defaults to None.
        solver (Optional[SolverTypes], optional): The chosen solver. Defaults to None.
        token_api (Optional[str], optional): The token api to access dwave leap solver.
            Defaults to None.
        seed (Optional[int], optional): The random seed to have reproducible simulation
            results. Defaults to None.

    Raises:
        ValueError: The TOKEN_API environment variable has not been defined

    Example:

        >>> main(
        ...     file_path="data/Histórico_carteras_Welzia_2018.xlsm",
        ...     sheet_name="BBG (valores)",
        ...     ns=254,
        ...     w=5,
        ...     steps=5,
        ...     expected_returns=None,
        ...     budget=1.0,
        ...     theta1=0.9,
        ...     theta2=0.4,
        ...     theta3=0.1,
        ... )  # doctest: +SKIP
    """
    if seed is None:
        seed = 42

    np.random.seed(seed)

    if budget is None:
        budget = 1.0

    if theta1 is None:
        theta1 = 0.9

    if theta2 is None:
        theta2 = 0.4

    if theta3 is None:
        theta3 = 0.1

    if solver is None:
        solver = SolverTypes.hybrid_solver

    if token_api is None:
        if (token_api := os.getenv("TOKEN_API")) is None:
            raise ValueError(
                "TOKEN_API env var not set. Please pass your dwave token-api trough "
                "the 'token_api' parameter or define the TOKEN_API env var."
            )

    df = read_welzia_stocks_file(file_path, sheet_name)
    historical_assets = Assets(df=df)

    future_assets = simulate_assets(
        historical_assets, er=expected_returns, ns=ns, order=12
    )

    _, interpretation = optimize(
        future_assets,
        b=budget,
        w=w,
        theta1=theta1,
        theta2=theta2,
        theta3=theta3,
        solver=solver,
        token_api=token_api,
        steps=steps,
    )

    if interpretation is not None:
        selected_funds = future_assets.df.columns
        store_results(selected_funds, interpretation)
        logger.info(f"{selected_funds}")
    else:
        logger.warning(f"No positive sharpe ratio have been found!")
