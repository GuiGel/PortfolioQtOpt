import os
import typing as ty
import streamlit as st
from bokeh.palettes import Turbo256  # type: ignore[import]
from bokeh.plotting import figure  # type: ignore[import]
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path
from streamlit.runtime.uploaded_file_manager import UploadedFile
from portfolioqtopt.assets import Assets
from portfolioqtopt.reader import read_welzia_stocks_file

load_dotenv()

token_api = None
if token_api is None:
    if (token_api := os.getenv("TOKEN_API")) is None:
        raise ValueError(
            "TOKEN_API env var not set. Please pass your dwave token-api trough "
            "the 'token_api' parameter or define the TOKEN_API env var."
        )


def form_submit_button_callback(key: str) -> None:
    # create key name
    if not key in st.session_state:
        # The submit has been pressed
        st.session_state[key] = 1
    else:
        st.session_state[key] += 1


def step_0_form(form_key: str, file_uploader_key: str, sheet_name_key: str, button_key: str) -> None:
    logger.info("load data")

    with st.form(key=form_key):

        st.file_uploader(
            "Elige un fichero al formato Welzia", type="xlsm", key=file_uploader_key
        )

        st.text_input(
            "Elige el nombre de la hoja a leer.",
            value="BBG (valores)",
            key=f"{form_key}_sheet_name",
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
            # we remove the press count
            st.session_state[button_key] -= 1
            st.exception(e)

        logger.info(f"{submit=}")


def step_0_get_args(file_uploader_key: str, sheet_name_key: str) -> ty.Optional[ty.Tuple[UploadedFile, str]]:    # argument to process step 1
    if (
        st.session_state.get(file_uploader_key) is not None
        and st.session_state.get(sheet_name_key) is not None
    ):
        st.write(f"inside get args: {st.session_state[file_uploader_key]}")
        return st.session_state[file_uploader_key], st.session_state[sheet_name_key]
    else:
        return None


def step_0_compute(*args: str) -> Assets:
    df = read_welzia_stocks_file(*args)
    return Assets(df=df)


def visualize_assets(assets: Assets) -> None:
    logger.info("visualize df")
    # ----- bokeh visualization
    p = figure(
        title="simple line example",
        x_axis_label="Fechas",
        y_axis_label="Activos",
        x_axis_type="datetime",
        y_axis_type="log",
        background_fill_color="#fafafa",
    )

    for i, column in enumerate(assets.df):
        color = Turbo256[int(256 * i / assets.df.shape[1])]
        p.line(
            assets.df.index,
            assets.df[column],
            legend_label=column,
            line_width=1,
            color=color,
        )
    st.bokeh_chart(p, use_container_width=True)


if __name__ == "__main__":

    st.write(st.session_state)

    args2 = None

    step_0_form("xlsm1", f"xlsm1_file", f"xlsm1_sheet_name", "xlsm1_pressed")
    args1 = step_0_get_args(f"xlsm1_file", f"xlsm1_sheet_name")

    if st.session_state.get("xlsm1_pressed"):

        step_0_form("xlsm2", f"xlsm2_file", f"xlsm2_sheet_name", "xlsm2_pressed")
        args2 = step_0_get_args(f"xlsm2_file", f"xlsm2_sheet_name")

    if args1 is not None:
        st.write(args1)
    if args2 is not None:
        st.write(args2)


