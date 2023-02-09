import os
from dataclasses import dataclass
from typing import Dict, Hashable, Optional, Tuple, Union

import pandas as pd
import streamlit as st
from bokeh.palettes import Turbo256  # type: ignore[import]
from bokeh.plotting import figure  # type: ignore[import]
from dotenv import load_dotenv
from loguru import logger
from streamlit.runtime.uploaded_file_manager import UploadedFile

from portfolioqtopt.application.utils import visualize_assets
from portfolioqtopt.assets import Assets, Scalar
from portfolioqtopt.optimization import Interpretation, SolverTypes, optimize
from portfolioqtopt.optimization.interpreter import Interpretation
from portfolioqtopt.reader import read_welzia_stocks_file
from portfolioqtopt.simulation import simulate_assets

load_dotenv()

token_api = None
if token_api is None:
    if (token_api := os.getenv("TOKEN_API")) is None:
        raise ValueError(
            "TOKEN_API env var not set. Please pass your dwave token-api trough "
            "the 'token_api' parameter or define the TOKEN_API env var."
        )


memory = st.session_state


@dataclass
class Record:
    order: int = 0
    count: int = 0
    last: int = 0


def form_submit_button_callback(key: str, order: int) -> None:
    logger.info(f"Callback key: {key=}")
    if not key in memory:
        # The submit button has been pressed
        memory[key] = Record(1, order)
    else:
        memory[key].order += 1
        memory[key].last += 1
    logger.debug(f"memory[{key}]={memory[key]}")


def step_0_form() -> Tuple[bool, Optional[UploadedFile], str]:
    with st.sidebar:
        st.markdown(f"## Load historical assets")
        with st.form(key="xlsm"):
            with st.expander("from xlsm file"):

                file = st.file_uploader(
                    "Elige un fichero al formato Welzia", type="xlsm"
                )

                sheet_name = st.text_input(
                    "Elige el nombre de la hoja a leer.",
                    value="BBG (valores)",
                )

            submit = st.form_submit_button(
                "submit values",
                on_click=form_submit_button_callback,
                args=("step_0_form", 1),  # register arguments
            )

    if submit and (file is None or sheet_name is None):
        e = ValueError("You must chose a file and a sheet name before submit!")
        st.exception(e)
        submit = False

    return submit, file, sheet_name


def step_0_compute(file_path: UploadedFile, sheet_name: str) -> Assets:
    df = read_welzia_stocks_file(file_path, sheet_name)
    return Assets(df=df)


def step_1_form(
    assets: Assets,
) -> Tuple[bool, int, Optional[Dict[Union[Scalar, Tuple[Hashable, ...]], float]]]:
    with st.sidebar:
        st.markdown("## Simulate future assets")
        with st.form(key="simulation"):
            with st.expander(
                "Rellenar las distintas opciones si se necesita o seguir con los "
                "valores por defecto."
            ):
                # ----- select number of days
                days_number = st.number_input(
                    "Choose a number of days to simulate",
                    help="Chose a number of days for which a prices must be simulated.",
                    value=256,
                    step=1,
                )

                # ----- select expected return for each asset
                st.text("Choose an expected returns")
                expected_returns: Dict[Union[Scalar, Tuple[Hashable, ...]], float] = {}
                for fund, er in zip(assets.df, assets.anual_returns.tolist()):
                    expected_returns[fund] = st.number_input(
                        str(fund), min_value=-1.0, value=er
                    )

            # ----- submit form
            submit = st.form_submit_button(
                "submit values",
                on_click=form_submit_button_callback,
                args=("step_1_form", 2),
            )
    return submit, int(days_number), expected_returns


@st.cache(suppress_st_warning=True)
def step_1_compute(
    assets: Assets,
    ns: int,
    er: Optional[Dict[Union[Scalar, Tuple[Hashable, ...]], float]] = None,
    order: int = 12,
) -> Assets:
    logger.info("step 1")
    with st.spinner("Generation en curso ..."):
        future = simulate_assets(assets, ns, er, order)
    st.success("Hecho!", icon="✅")
    return future


def step_2_form() -> Tuple[bool, float, int, float, float, float, int]:
    logger.info("optimize portfolio")
    with st.sidebar:
        st.markdown("## Portfolio Optimization")
        with st.form(key="optimization"):
            with st.expander(
                "Rellenar las distintas opciones si se necesita o seguir con los "
                "valores por defecto (lo aconsejado)."
            ):
                budget = float(
                    st.number_input(
                        "budget",
                        value=1.0,
                    )
                )
                w = int(st.number_input("granularity", value=5))
                theta1 = float(st.number_input("theta1", value=0.9))
                theta2 = float(st.number_input("theta2", value=0.4))
                theta3 = float(st.number_input("theta3", value=0.1))
                steps = int(st.number_input("steps", value=5, step=1))

            submit = st.form_submit_button(
                "submit values",
                on_click=form_submit_button_callback,
                args=("step_2_form", 3),
            )

        return submit, budget, w, theta1, theta2, theta3, steps


def step_2_compute(
    assets: Assets,
    b: float,
    w: int,
    theta1: float,
    theta2: float,
    theta3: float,
    solver: SolverTypes,
    token_api: str,
    steps: int,
    verbose: bool = False,
) -> Optional[Interpretation]:
    import time

    logger.info(f"step 2")
    with st.spinner("Ongoing optimization... Take your time it can last!"):
        time.sleep(2)
        _, interpretation = optimize(
            assets,
            b,
            w,
            theta1,
            theta2,
            theta3,
            solver,
            "token_api",
            steps,
            verbose=verbose,
        )
        # import numpy as np
        # 
        # interpretation = Interpretation(
        #     selected_indexes=pd.Index(["A", "C", "D"], dtype="object"),
        #     investments=np.array([0.75, 0.125, 0.125]),
        #     expected_returns=44.5,
        #     risk=17.153170260916784,
        #     sharpe_ratio=2.594272622676201,
        # )
    st.success("Hecho!", icon="✅")
    return interpretation


@st.cache
def step_2_convert_df(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


def step_2_download_button(df: pd.DataFrame, text: str, file_name: str) -> None:
    csv = step_2_convert_df(df)
    st.download_button(
        text,
        csv,
        file_name=file_name,
        mime=None,
        key=None,
        help=None,
        on_click=None,
        args=None,
        kwargs=None,
        disabled=False,
    )


def step_2_display(interpretation: Optional[Interpretation] = None):
    if interpretation is not None:
        # ----- DataFrame to collect selected funds and theirs related investment

        st.text("Metrics obtained for the selected portfolio.")

        df2 = pd.DataFrame(
            [
                interpretation.expected_returns,
                interpretation.risk,
                interpretation.sharpe_ratio,
            ],
            index=["expected_returns", "risk", "sharpe ratio"],
            columns=["Portfolio metrics"],
        )
        st.table(df2)
        step_2_download_button(
            df2, "download table", file_name="qoptimiza_portfolio.csv"
        )

        st.text("Investment for each asset in the portfolio.")

        df1 = pd.DataFrame(
            interpretation.investments,
            index=interpretation.selected_indexes,
            columns=["Investment"],
        )
        df1.index.name = "asset"
        st.table(df1)

        step_2_download_button(
            df1, "download table", file_name="qoptimiza_investments.csv"
        )

    else:
        st.markdown("Optimization fails. Run it again!")


def app():

    logger.info(f"{'-':->50}")
    logger.info(f"{' Enter App ':^50}")
    logger.info(f"{'-':->50}")

    st.title("Portfolio Quantum Optimization")

    with st.sidebar:
        st.title("Parameter selection")
        st.markdown(
            "In this sidebar you are invited to parametrize the optimization process "
            "in 3 ordered steps."
        )
        st.markdown("---")

    sub0, file_path, sheet_name = step_0_form()

    if sub0 or memory.get("step_1_form"):

        assert file_path is not None

        st.markdown("## 1. Historical assets")

        history = step_0_compute(file_path, sheet_name)

        with st.expander("Raw data"):
            st.dataframe(data=history.df)

        with st.expander("Plot data"):
            visualize_assets(history)

        sub1, ns, er = step_1_form(history)

        if sub1 or memory.get("step_2_form"):

            if not sub0:

                assert er is not None

                st.markdown("## 2. Assets simulation")

                future = step_1_compute(history, ns, er, order=12)

                with st.expander("Raw future assets"):
                    st.dataframe(data=future.df)

                with st.expander("Plot the future assets"):
                    visualize_assets(future)

                sub2, budget, w, theta1, theta2, theta3, steps = step_2_form()

                if sub2:

                    st.markdown("## 3. Find the best portfolio")

                    interpretation = step_2_compute(
                        future,
                        budget,
                        w,
                        theta1,
                        theta2,
                        theta3,
                        SolverTypes.hybrid_solver,
                        token_api,
                        steps,
                        verbose=False,
                    )

                    step_2_display(interpretation)

    logger.info(f"{'-':->50}")
    logger.info(f"{' Done ':^50}")
    logger.info(f"{'-':->50}")
    logger.info(f"\n\n\n\n")


if __name__ == "__main__":
    app()
