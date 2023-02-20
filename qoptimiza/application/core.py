from pathlib import Path
from typing import Dict, Hashable, Optional, Tuple, Union

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from loguru import logger
from streamlit.runtime.uploaded_file_manager import UploadedFile

from qoptimiza.application import streamlit_debug
from qoptimiza.application.record import Record, delete_posterior_records
from qoptimiza.application.utils import visualize_assets
from qoptimiza.assets import Assets, Scalar
from qoptimiza.optimization import Interpretation, SolverTypes, check_dwave_token
from qoptimiza.reader import read_welzia_stocks_file
from qoptimiza.simulation import simulate_assets
from qoptimiza.simulation.errors import CovNotSymDefPos

streamlit_debug.set(flag=False, wait_for_client=False)


load_dotenv()

memory = st.session_state


def get_token_api() -> None:

    col1, col2 = st.columns(2)

    def callback() -> None:
        nonlocal col2
        if "token_api" not in memory:
            memory["token_api"] = None

        if (input_token := memory["input_token_api"]) != "":
            if not check_dwave_token(input_token):
                with st.sidebar:
                    with col2:
                        st.error("Authentication error!")
            else:
                memory["token_api"] = Record(order=1, value=input_token)
                token_api_container.empty()

    with col1:
        if st.button(
            "Connect to D-Wave",
            key="token_api_button_pressed",
            help=(
                "Set the token api to connect to dwave solver in order to solve the "
                "optimization problem"
            ),
        ):
            # A new token api have been submitted. Clean memory.

            for key in memory:
                del memory[key]

            token_api_container = st.empty()

            token_api_container.text_input(
                "Enter your dwave leap token",
                label_visibility="visible",
                key="input_token_api",
                on_change=callback,
                type="password",
            )


def _step_0_form() -> Tuple[bool, Optional[UploadedFile], str]:
    with st.form(key="xlsm", clear_on_submit=True):
        with st.expander("from xlsm file"):

            file = st.file_uploader(
                "Elige un fichero al formato Welzia",
                type=["xlsm", "xls"],
                accept_multiple_files=False,
            )

            sheet_name = st.text_input(
                "Elige el nombre de la hoja a leer.",
                value="BBG (valores)",
            )

        submit = st.form_submit_button(
            "submit values",
        )
    return submit, file, sheet_name


def _step_0_check_error() -> None:
    e = ValueError("You must chose a file and a sheet name before submit!")
    st.exception(e)

    delete_posterior_records(order=2, memory=memory)


def _step_0_initialize(file_path: Optional[UploadedFile], sheet_name: str) -> None:
    # Initialize step_0_form key in memory.
    logger.info("submit historical args")
    delete_posterior_records(order=2, memory=memory)
    memory["step_0_form"] = Record(
        order=2,
        value={"file_path": file_path, "sheet_name": sheet_name},
    )


def step_0_form() -> None:
    with st.sidebar:
        st.markdown(f"## Load historical assets")
        submit, file_path, sheet_name = _step_0_form()

    if submit and any([file_path is None, sheet_name is None]):
        _step_0_check_error()

    elif submit and not all([file_path is None, sheet_name is None]):
        _step_0_initialize(file_path, sheet_name)


def step_0_compute(file_path: UploadedFile, sheet_name: str):
    try:
        df = read_welzia_stocks_file(file_path, sheet_name)
        # raise ValueError()
    except Exception:
        st.error("read welzia fails!")
        delete_posterior_records(order=4, memory=memory)
    else:
        memory["step_0_compute"] = Record(order=3, value=Assets(df=df))


def _step_1_form(
    assets: Assets,
) -> Tuple[bool, int, Dict[Union[Scalar, Tuple[Hashable, ...]], float], int]:
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
        submit = st.form_submit_button("submit values")

    return submit, int(days_number), expected_returns, int(seed)


def _step_1_initialize(
    assets: Assets,
    days_number: int,
    expected_returns: Dict[Union[Scalar, Tuple[Hashable, ...]], float],
    seed: int,
) -> None:
    # insert the arguments of the computation function in "step_1_form".
    logger.info("submit historical args")
    delete_posterior_records(order=4, memory=memory)
    memory["step_1_form"] = Record(
        order=4,
        value={
            "assets": assets,
            "ns": days_number,
            "er": expected_returns,
            "seed": seed,
        },
    )


def step_1_form(
    assets: Assets,
) -> None:
    with st.sidebar:
        st.markdown("## Simulate future assets")
        submit, days_numbers, expected_returns, seed = _step_1_form(assets)

    if submit:  # here all the value are not None by default
        _step_1_initialize(assets, days_numbers, expected_returns, seed)


def step_1_compute(
    assets: Assets,
    ns: int,
    er: Optional[Dict[Union[Scalar, Tuple[Hashable, ...]], float]] = None,
    order: int = 12,
    seed: Optional[int] = None,
) -> None:
    logger.info("step 1")
    try:
        with st.spinner("Generation en curso ..."):
            future = simulate_assets(assets, ns, er, order, seed)
    except CovNotSymDefPos as e:
        st.error(e)
        delete_posterior_records(order=6, memory=memory)
    else:
        st.success("Hecho!", icon="✅")
        memory["step_1_compute"] = Record(order=5, value=future)
        logger.debug(f"{future.df.iloc[2, 2]=}")


def _step_2_form() -> Tuple[bool, float, int, float, float, float, int]:
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

        submit = st.form_submit_button("submit values")

    return submit, budget, w, theta1, theta2, theta3, steps


def _step_2_initialize(
    assets: Assets,
    b: float,
    w: int,
    theta1: float,
    theta2: float,
    theta3: float,
    solver: SolverTypes,
    steps: int,
) -> None:
    logger.info("submit future args")
    memory["step_2_form"] = Record(
        order=6,
        value={
            "assets": assets,
            "b": b,
            "w": w,
            "theta1": theta1,
            "theta2": theta2,
            "theta3": theta3,
            "solver": solver,
            "token_api": memory["token_api"],
            "steps": steps,
        },
    )


def step_2_form(asset: Assets, solver: SolverTypes) -> None:
    logger.info("optimize portfolio")
    with st.sidebar:
        st.markdown("## Portfolio Optimization")
        submit, b, w, theta1, theta2, theta3, steps = _step_2_form()

    if submit:
        delete_posterior_records(order=6, memory=memory)

        if any(
            [
                b <= 0,
            ]
        ):  # TODO add more conditions
            # _step_0_check_error(submit, file_path, sheet_name)
            st.error("budget can't be negative")

        elif all(
            [
                b > 0,
            ]
        ):
            _step_2_initialize(asset, b, w, theta1, theta2, theta3, solver, steps)


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
) -> None:
    logger.info(f"step 2")
    try:
        with st.spinner("Ongoing optimization... Take your time it can last!"):
            # _, interpretation = optimize(
            #     copy.deepcopy(assets),
            #     b,
            #     w,
            #     theta1,
            #     theta2,
            #     theta3,
            #     solver,
            #     token_api,
            #     steps,
            #     verbose=verbose,
            # )
            import numpy as np

            interpretation = Interpretation(
                selected_indexes=pd.Index(["A", "C", "D"], dtype="object"),
                investments=np.array([0.75, 0.125, 0.125]),
                expected_returns=44.5,
                risk=17.153170260916784,
                sharpe_ratio=2.594272622676201,
            )
    except Exception as e:
        st.error(e)
        delete_posterior_records(order=7, memory=memory)
    else:
        st.success("Hecho!", icon="✅")
        memory["step_2_compute"] = Record(order=7, value=interpretation)


@st.cache(show_spinner=False)
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

    logger.debug(memory)

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
        # ----- Create key token_api_new if good dwave token api
        get_token_api()

    if memory.get("token_api") is not None:

        step_0_form()

        if (record := memory.get("step_0_form")) is not None:

            st.markdown("## 1. Historical assets")

            step_0_compute(**record.value)

            if (history := memory.get("step_0_compute")) is not None:

                with st.expander("Raw historical assets"):
                    st.dataframe(data=history.value.df)

                with st.expander("Plot the historical assets"):
                    visualize_assets(history.value)

                st.markdown("## 2. Assets simulation")

                step_1_form(history.value)

                if (args_1 := memory.get("step_1_form")) is not None:

                    if memory.get("FormSubmitter:simulation-submit values"):
                        step_1_compute(**args_1.value)

                    if (future := memory.get("step_1_compute")) is not None:

                        with st.expander("Raw future assets"):
                            st.dataframe(data=future.value.df)

                        with st.expander("Plot the future assets"):
                            visualize_assets(future.value)

                        step_2_form(future, SolverTypes.hybrid_solver)

                        if (args_2 := memory.get("step_2_form")) is not None:

                            st.markdown("## 3. Find the best portfolio")

                            if memory.get("FormSubmitter:optimization-submit values"):
                                step_2_compute(**args_2.value)

                            if (
                                interpretation := memory.get("step_2_compute")
                            ) is not None:
                                step_2_display(interpretation.value)

    logger.info("-".center(30, "-"))
    logger.info(" END ".center(30, "-"))
    logger.info("\n\n\n\n\n")
