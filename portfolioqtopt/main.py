import time
import os
from pathlib import Path
from typing import Dict, Hashable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from portfolioqtopt.assets import Assets
from portfolioqtopt.optimization.interpreter_ import Interpretation
from portfolioqtopt.optimization.optimization_ import SolverTypes, optimize
from portfolioqtopt.reader import read_welzia_stocks_file
from portfolioqtopt.simulation.simulation import Scalar, simulate_assets

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


# Il me faut penser à comment faire le lien avec les autres parties du programme.
# Au départ un pd.DataFrame qui vient de plusieurs sources.
# L'algo the Tecnalia doit pouvoir prendre en entrée un DataFrame ou un numpy array
# avec une série d'indexes. Ou à moi de transformer la sortie de la simulation en
# dataframe.
# Il faut aussi penser à la simulation...
# Il y a au moins le lien avec stooq ou yahoo finance qui peut être faire.


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
        logger.info(f"token api is None")
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
        token_api=token_api,  # "DEV-a0c729a02f82af930c096956e80b8887ce7b3f6e",  # "DEV-d9751cb50bc095c993f55b3255f728d5b2793c36",
        steps=steps,
    )

    if interpretation is not None:
        selected_funds = future_assets.df.columns
        store_results(selected_funds, interpretation)
        logger.info(f"{selected_funds}")
    else:
        logger.warning(f"No positive sharpe ratio have been found!")

if __name__ == "__main__":
    file_path = "/home/ggelabert/Projects/PortfolioQtOpt/data/Histórico_carteras_Welzia_2018.xlsm"
    sheet_name = "BBG (valores)"
    ns = 254
    main(file_path, sheet_name, ns, 5, 5)