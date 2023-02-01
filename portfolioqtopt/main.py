import os
from pathlib import Path
from typing import Dict, Hashable, Optional, Tuple, Union

import numpy as np
from dotenv import load_dotenv
from loguru import logger

from portfolioqtopt.assets import Assets, Scalar
from portfolioqtopt.optimization import SolverTypes, optimize
from portfolioqtopt.reader import read_welzia_stocks_file
from portfolioqtopt.simulation import simulate_assets

# Loading environment variables
load_dotenv()


@logger.catch
def main(
    file_path: Union[Path, str],
    sheet_name: str,
    ns: int,
    w: int,
    steps: int,
    expected_returns: Optional[Dict[Union[Scalar, Tuple[Hashable, ...]], float]] = None,
    order: Optional[int] = None,
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
        ValueError: The TOKEN_API environment variable has not been defined.

    Example:

        First if you want to see some logs, don't forget to enable the logs.

        >>> from portfolioqtopt import log
        >>> log.enable("INFO")

        Then we can directly call the main function to run the portfolio optimization.
        
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

    if order is None:
        order = 12

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

    # ----- Historical data
    output_str = f"\n{'-':-^51}\n"
    output_str += f"{' Historical data ':^51}\n"
    output_str += f"{'-':-^51}\n"
    output_str += f"{' file path':>24} : {file_path:<24}\n"
    output_str += f"{' sheet name':>24} : {sheet_name:<24}\n"

    df = read_welzia_stocks_file(file_path, sheet_name)
    historical_assets = Assets(df=df)

    # ----- Simulation
    output_str += f"{'-':-^51}\n"
    output_str += f"{' Simulation ':^51}\n"
    output_str += f"{'-':-^51}\n"
    if expected_returns is not None:
        output_str += f"{' fund':>24} : {'expected return':<24}\n"
        for idx, ivt in expected_returns.items():
            output_str += f"{idx:>24} : {ivt:<24}\n"
    else:
        output_str += f"{' expected returns':>24} : {'None':<24}\n"
    output_str += f"{' ns':>24} : {ns:<24}\n"
    output_str += f"{' order':>24} : {order:<24}\n"

    future_assets = simulate_assets(
        historical_assets,
        er=expected_returns,
        ns=ns,
        order=order,
    )

    # ----- Optimization
    output_str += f"{'-':-^51}\n"
    output_str += f"{' Optimization ':^51}\n"
    output_str += f"{'-':-^51}\n"
    output_str += f"{' budget':>24} : {budget:<24}\n"
    output_str += f"{' w':>24} : {w:<24}\n"
    output_str += f"{' theta1':>24} : {theta1:<24}\n"
    output_str += f"{' theta2':>24} : {theta2:<24}\n"
    output_str += f"{' theta3':>24} : {theta3:<24}\n"
    output_str += f"{' budget':>24} : {budget:<24}\n"
    output_str += f"{' solver':>24} : {solver.value:<24}\n"
    output_str += f"{' steps':>24} : {steps:<24}\n"

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
        output_str += interpretation.to_str()
    else:
        logger.warning(f"No positive sharpe ratio have been found!")

    logger.info(output_str)
