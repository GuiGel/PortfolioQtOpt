import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, Optional, Tuple, Union

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from loguru import logger
from streamlit.runtime.uploaded_file_manager import UploadedFile

from qoptimiza.application.utils import visualize_assets
from qoptimiza.application.memory import _token_api
from qoptimiza.assets import Assets, Scalar
from qoptimiza.optimization import Interpretation, SolverTypes, optimize
from qoptimiza.optimization.interpreter import Interpretation
from qoptimiza.reader import read_welzia_stocks_file
from qoptimiza.simulation import simulate_assets

load_dotenv()


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
) -> Tuple[bool, int, Optional[Dict[Union[Scalar, Tuple[Hashable, ...]], float]], int]:
    with st.sidebar:
        st.markdown("## Simulate future assets")
        with st.form(key="simulation"):
            with st.expander(
                "Rellenar las distintas opciones si se necesita o seguir con los "
                "valores por defecto."
            ):
                # ----- chose a seed for random number generation
                seed = st.number_input(
                    "choose a seed for random number generation",
                    help=(
                        "Internally the simulation begin by generating random numbers "
                        "that are exactly the same between each runs if they have the "
                        "same seed. "
                    ),
                    value=42,
                )
                logger.debug(f"{seed=}")

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
    return submit, int(days_number), expected_returns, int(seed)


@st.cache(suppress_st_warning=True)
def step_1_compute(
    assets: Assets,
    ns: int,
    er: Optional[Dict[Union[Scalar, Tuple[Hashable, ...]], float]] = None,
    order: int = 12,
    seed: Optional[int] = None,
) -> Assets:
    logger.info("step 1")
    with st.spinner("Generation en curso ..."):
        future = simulate_assets(assets, ns, er, order, seed)
    st.success("Hecho!", icon="✅")
    logger.debug(f"{future.df.iloc[2, 2]=}")
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
    import copy

    logger.info(f"step 2")
    with st.spinner("Ongoing optimization... Take your time it can last!"):
        _, interpretation = optimize(
            copy.deepcopy(assets),
            b,
            w,
            theta1,
            theta2,
            theta3,
            solver,
            token_api,
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


def get_token_api() -> str:
    def callback():
        if st.session_state["token_api"] != "":
            logger.debug("token api not equal to ''")
            logger.debug("empty token api container")
            _token_api.value = memory.token_api
            token_api_container.empty()
            logger.info(f"token_api = {memory.token_api}")

    st.markdown(
        f"Set the token api to connect to dwave solver in order to solve the "
        f"optimization problem"
    )

    if st.button("token api", key="token_api_button"):
        logger.debug("token api button press")
        token_api_container = st.empty()
        token_api_container.text_input(
            "Enter your dwave leap token",
            label_visibility="visible",
            key="token_api",
            on_change=callback,
        )

    logger.debug(f"returned token api = {memory.get('token_api')}")

    return memory.get("token_api", "")


def app():
    logger.info(f"{'-':->50}")
    logger.info(f"{' Enter App ':^50}")
    logger.info(f"{'-':->50}")

    logger.debug(st.session_state)
    logger.debug(_token_api)

    logo_path = str(Path(__file__).parent / "images/logo.png")
    icon_path = str(Path(__file__).parent / "images/icon.png")

    st.set_page_config(
        page_title="QOptimiza",
        page_icon=icon_path,
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.image(logo_path)
    st.markdown("**Markowitz Portfolio Quantum Optimization**")
    st.markdown(":blue[by: Serikat & Tecnalia]")
    st.markdown("---")

    with st.sidebar:
        st.title("Parameter selection for portfolio optimization")
        st.markdown(
            "In this sidebar you can parametrize the optimization process "
            "in 3 ordered steps."
        )
        st.markdown("---")

    with st.sidebar:
        get_token_api()

    if _token_api.value is not None:

        sub0, file_path, sheet_name = step_0_form()

        if sub0 or memory.get("step_1_form"):

            assert file_path is not None

            st.markdown("## 1. Historical assets")

            history = step_0_compute(file_path, sheet_name)

            with st.expander("Raw data"):
                st.dataframe(data=history.df)

            with st.expander("Plot data"):
                visualize_assets(history)

            sub1, ns, er, seed = step_1_form(history)

            if sub1 or memory.get("step_2_form"):

                if not sub0:

                    assert er is not None

                    st.markdown("## 2. Assets simulation")

                    future = step_1_compute(history, ns, er, order=12)

                    with st.expander("Raw future assets"):
                        st.dataframe(data=future.df)

                    with st.expander("Plot the future assets"):
                        visualize_assets(future)

                    (
                        sub2,
                        budget,
                        w,
                        theta1,
                        theta2,
                        theta3,
                        steps,
                    ) = step_2_form()

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
                            _token_api.value,
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
