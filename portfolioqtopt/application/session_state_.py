import os
import typing as ty

import streamlit as st
from bokeh.palettes import Turbo256  # type: ignore[import]
from bokeh.plotting import figure  # type: ignore[import]
from dotenv import load_dotenv
from loguru import logger
from streamlit.runtime.uploaded_file_manager import UploadedFile

from portfolioqtopt.application.utils import visualize_assets
from portfolioqtopt.assets import Assets, Scalar
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


def form_submit_button_callback(key: str) -> None:
    logger.info(f"Callback key: {key=}")
    if not key in st.session_state:
        # The submit has been pressed
        st.session_state[key] = 1
    else:
        st.session_state[key] += 1


def step_0_form(
    form_key: str, file_uploader_key: str, sheet_name_key: str, button_key: str
) -> ty.Optional[ty.Tuple[UploadedFile, str]]:
    # collect value of argument with a form
    logger.info("load data")

    with st.form(key=form_key):

        st.file_uploader(
            "Elige un fichero al formato Welzia", type="xlsm", key=file_uploader_key
        )

        st.text_input(
            "Elige el nombre de la hoja a leer.",
            value="BBG (valores)",
            key=sheet_name_key,
        )

        submit = st.form_submit_button(
            "submit values",
            on_click=form_submit_button_callback,
            args=(button_key,),
        )

        if submit and (
            st.session_state[file_uploader_key] is None
            or st.session_state[sheet_name_key] is None
        ):
            e = ValueError("You must chose a file and a sheet name before submit!")
            st.session_state[button_key] -= 1
            st.exception(e)

    # Create the arguments for calling Assets
    if (
        st.session_state.get(file_uploader_key) is not None
        and st.session_state.get(sheet_name_key) is not None
    ):
        st.write(f"inside get args: {st.session_state[file_uploader_key]}")
        return st.session_state[file_uploader_key], st.session_state[sheet_name_key]
    else:
        return None


def step_0_compute(file: UploadedFile, sheet_name: str, history_key: str) -> Assets:
    df = read_welzia_stocks_file(file, sheet_name)
    st.session_state[history_key] = Assets(df=df)
    return st.session_state[history_key]


def step_1_form(
    assets: Assets, days_key: str, button_key: str, expected_return_key: str
) -> ty.Optional[
    ty.Tuple[
        Assets,
        int,
        ty.Optional[ty.Dict[ty.Union[Scalar, ty.Tuple[ty.Hashable, ...]], float]],
    ]
]:
    logger.info("get simulation args")
    with st.form(key="simulation"):
        with st.expander(
            "Rellene las diferentes opciones si es necesario o continúe con los "
            "valores por defecto."
        ):
            # ----- select number of days
            num = st.number_input(
                "numero de días a simular", value=256, step=1, key=days_key
            )
            logger.debug(f"{num=}")
            logger.debug(f"{st.session_state.get(days_key)=}")

            # ----- select expected return for each asset
            st.text("Elige el rendimiento de cada activo:")

            expected_returns: ty.Dict[
                ty.Union[Scalar, ty.Tuple[ty.Hashable, ...]], float
            ] = {}
            for fund, er in zip(assets.df, assets.anual_returns.tolist()):
                expected_returns[str(fund)] = st.slider(str(fund), -1.0, 3.0, er)
            st.session_state[expected_return_key] = expected_returns

        # ----- submit form
        submit = st.form_submit_button(
            "submit values",
            on_click=form_submit_button_callback,
            args=(button_key,),  # register arguments
        )

    # Create the arguments for calling Simulation
    if (
        submit
        and st.session_state.get(days_key) is not None
        and st.session_state.get(expected_return_key) is not None
    ):
        return assets, st.session_state[days_key], st.session_state[expected_return_key]
    else:
        return None


def step_1_compute(
    assets: Assets,
    ns: int,
    er: ty.Optional[
        ty.Dict[ty.Union[Scalar, ty.Tuple[ty.Hashable, ...]], float]
    ] = None,
    order: int = 12,
) -> Assets:
    logger.info("step 1")
    with st.spinner("Generation en curso ..."):
        future = simulate_assets(assets, ns, er, order)
    st.success("Hecho!", icon="✅")
    return future


if __name__ == "__main__":

    args1 = step_0_form("xlsm1", f"xlsm1_file", f"xlsm1_sheet_name", "xlsm1_pressed")

    if args1 is not None:

        history = step_0_compute(*args1, "history")

        with st.expander("Raw historical assets"):
            st.dataframe(data=history.df)

        with st.expander("Plot the historical assets"):
            visualize_assets(history)

        args2 = step_1_form(
            history, "simulation_days", "simulation_pressed", "expected_return"
        )

        if args2 is not None:

            future = step_1_compute(*args2)

            with st.expander("Raw future assets"):
                st.dataframe(data=future.df)

            with st.expander("Plot the future assets"):
                visualize_assets(future)

    st.write(args1)
    st.write(st.session_state)
