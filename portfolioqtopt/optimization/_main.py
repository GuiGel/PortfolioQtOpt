import typing

from portfolioqtopt.optimization._interpreter import Interpretation
from portfolioqtopt.optimization._optimization import SolverTypes, optimize
from loguru import logger
import time
import pandas as pd

logger.add(f"logs/{round(time.time() * 1000)}.log", level="DEBUG")

def store_results(
    funds: pd.Index, interpretation: Interpretation
) -> None:
    final_results = {
        "selected funds": funds,
        "investment": interpretation.investment,
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
    prices = df.to_numpy()

    indexes, interpretation = optimize(
        prices,
        b=1.0,
        w=6,
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
    else:
        logger.warning(f"No positive sharpe ratio!")