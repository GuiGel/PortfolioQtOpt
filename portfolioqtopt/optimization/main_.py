import time

import pandas as pd
from loguru import logger
import numpy as np

from portfolioqtopt.optimization.assets_ import Assets
from portfolioqtopt.optimization.interpreter_ import Interpretation
from portfolioqtopt.optimization.optimization_ import SolverTypes, optimize

logger.add(f"logs/{round(time.time() * 1000)}.log", level="DEBUG")


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

if __name__ == "__main__":

    from portfolioqtopt.reader import read_welzia_stocks_file

    file_path = "/home/ggelabert/Projects/PortfolioQtOpt/data/Histórico_carteras_Welzia_2018.xlsm"
    sheet_name = "BBG (valores)"

    df = read_welzia_stocks_file(file_path, sheet_name)
    for i, c in enumerate(df.columns):
        logger.info(f"{i}: {c[2]}")
    prices = df.to_numpy(dtype=np.float64)

    assets = Assets(prices)

    print(f"{assets.anual_returns=}")

    indexes, interpretation = optimize(
        assets,
        b=1.0,
        w=5,
        theta1=0.9,
        theta2=0.4,
        theta3=0.1,
        solver=SolverTypes.hybrid_solver,
        token_api="DEV-a0c729a02f82af930c096956e80b8887ce7b3f6e",  # "DEV-d9751cb50bc095c993f55b3255f728d5b2793c36",
        steps=5,
    )

    if interpretation is not None:
        selected_funds = df.columns[indexes]
        store_results(selected_funds, interpretation)
        a = [k[2] for k in selected_funds]
        logger.info(f"{list(a)}")
    else:
        logger.warning(f"No positive sharpe ratio!")
