import os
from typing import List, Optional, cast

import pandas as pd
import streamlit as st
from bokeh.palettes import Turbo256  # type: ignore[import]
from bokeh.plotting import figure  # type: ignore[import]
from dotenv import load_dotenv
from loguru import logger

from portfolioqtopt import log
from portfolioqtopt.application.memory import register
from portfolioqtopt.assets import Assets
from portfolioqtopt.optimization import Interpretation, SolverTypes, optimize
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


def step_0_form() -> bool:
    with st.form(key="xlsm"):
        logger.info("load data")

        st.file_uploader("Elige un fichero al formato Welzia", type="xlsm", key="xlsm")
        # uploaded_file = "/home/ggelabert/Projects/PortfolioQtOpt/data/Histórico_carteras_Welzia_2018.xlsm"

        st.text_input(
            "Elige el nombre de la hoja a leer.",
            value="BBG (valores)",
            key="sheet_name",
        )

        logger.info(f"F1 = {st.session_state.keys()}")

        submit = st.form_submit_button(
            "submit values",
            on_click=register,
            args=("xlsm", 0, "xlsm", "sheet_name"),  # register arguments
        )
        logger.info(f"{submit=}")
        return submit


def step_0_compute(*args: str) -> Assets:
    df = read_welzia_stocks_file(*args)
    return Assets(df=df)


def step_0_visualize(assets: Assets) -> None:
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


def step_1_form(assets: Assets) -> bool:
    logger.info("get simulation args")
    with st.form(key="simulation"):
        with st.expander(
            "Rellenar las distintas opciones si se necesita o seguir con los "
            "valores por defecto."
        ):
            # ----- select number of days
            st.number_input("numero de días a simular", value=256, step=1, key="ns")

            # ----- select expected return for each asset
            st.text("Elije el retorno para cada activo:")
            funds: List[str] = assets.df.columns.tolist()
            for fund, er in zip(assets.df, assets.anual_returns.tolist()):
                st.slider(str(fund), -1.0, 3.0, er, key=cast(str, fund))

        # ----- submit form
        submit = st.form_submit_button(
            "submit values",
            on_click=register,
            args=("step_1", 1, "ns", *funds),  # register arguments
        )
        logger.info(f"{submit=}")
        return submit


def step_1_compute(*args) -> Assets:
    logger.info("step 1")
    with st.spinner("Generation en curso ..."):
        future = simulate_assets(*args)
    st.success("Hecho!", icon="✅")
    return future


def step_2_form() -> bool:
    logger.info("optimize portfolio")
    with st.form(key="optimization"):
        with st.expander(
            "Rellenar las distintas opciones si se necesita o seguir con los "
            "valores por defecto (lo aconsejado)."
        ):
            st.number_input("budget", value=1.0, key="b")
            st.number_input("granularity", value=5, key="w")
            st.number_input("theta1", value=0.9, key="theta1")
            st.number_input("theta2", value=0.4, key="theta2")
            st.number_input("theta3", value=0.1, key="theta3")
            st.number_input("steps", value=5, step=1, key="steps")

        submit = st.form_submit_button(
            "Run optimization",
            on_click=register,
            args=(
                "step_2",
                2,
                "b",
                "w",
                "theta1",
                "theta2",
                "theta3",
                "steps",
            ),  # register arguments
        )
        return submit


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
    with st.spinner("Optimización en curso..."):
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
        """import numpy as np

        from portfolioqtopt.optimization.interpreter import Interpretation

        interpretation = Interpretation(
            selected_indexes=pd.Index(["A", "C", "D"], dtype="object"),
            investments=np.array([0.75, 0.125, 0.125]),
            expected_returns=44.5,
            risk=17.153170260916784,
            sharpe_ratio=2.594272622676201,
        )"""
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

    st.markdown("# QOptimiza  ")
    st.markdown("---")
    st.markdown("Quantum optimization of Markovitz Porfolio")
    st.markdown("---")
    st.markdown("## 1. Select the historical assets")

    # ----- collect arguments for creating assets
    submit_0 = step_0_form()
    logger.info(f"{submit_0}")

    # ----- argument for step 1 has been collected
    if hasattr(register, "xlsm"):

        # ----- compute step 1
        if submit_0:
            register.xlsm.val = step_0_compute(*register.xlsm.args)

        st.markdown("**Look at the historical assets loaded.**")

        # ----- display results of step 0
        with st.expander("Explore the historical assets"):
            st.dataframe(data=register.xlsm.val.df)

        with st.expander("Plot the historical assets"):
            step_0_visualize(register.xlsm.val)

        # ----- collect arguments for step 1

        st.markdown("## 2. Simulate future asset prices")

        submit_1 = step_1_form(register.xlsm.val)
        logger.info(f"{submit_1=}")

        # ----- argument for step 1 has been collected
        if hasattr(register, "step_1"):

            # ----- compute step 1
            if submit_1:
                # ----- collect args for step_1_compute
                # TODO quite bad this code!

                assets = register.xlsm.val
                ns = register.step_1.args[0]
                er = dict(zip(assets.df, register.step_1.args[1:]))

                register.step_1.val = step_1_compute(assets, ns, er)

            st.markdown("**Look at the simulated assets.**")

            # ----- display results of step 1
            with st.expander("Explore the future assets"):
                st.dataframe(register.step_1.val.df)

            with st.expander("Plot the future assets"):
                step_0_visualize(register.step_1.val)

            st.markdown("## 3. Found the best porfolio")

            # ----- collect arguments for step 2
            submit_2 = step_2_form()
            logger.info(f"{submit_2=}")

            #  ----- the button in step 2 has been pressed
            if submit_2:

                # ----- compute step 2
                (
                    b,
                    w,
                    theta1,
                    theta2,
                    theta3,
                    steps,
                ) = register.step_2.args  # For better readability

                args = (
                    register.step_1.val,
                    *(b, w, theta1, theta2, theta3),
                    SolverTypes.hybrid_solver,
                    token_api,  # ""
                    steps,
                )
                register.step_2.val = step_2_compute(*args)

            # ----- display results step 2
            if hasattr(register, "step_2"):
                with st.expander("Optimization results"):
                    step_2_display(register.step_2.val)

    logger.info(f"memo = {register}")

    logger.info(f"{'-':->50}")
    logger.info(f"{' Done ':^50}")
    logger.info(f"{'-':->50}")


class App:
    @staticmethod
    def f1():
        logger.info("enter f1")
        with st.form(key="f1"):
            st.text_input("val1", 10, key="f1_v1")
            st.text_input("val2", 30, key="f1_v2")

            logger.info(f"F1 = {st.session_state.keys()}")

            submit = st.form_submit_button(
                "submit values",
                on_click=register,
                args=("f1", 0, "f1_v1", "f1_v2"),  # register arguments
            )
            logger.info(f"{submit=}")
            return submit

    @staticmethod
    @st.cache
    def calc_1(a, b) -> float:
        logger.debug(f"inside calc_1 {locals()}")
        return a + b

    @staticmethod
    def f2():
        logger.info("enter f2")
        with st.form(key="f2"):
            st.text_input("val1", 10, key="f2_v1")
            st.text_input("val2", 30, key="f2_v2")

            submit = st.form_submit_button(
                "submit values",
                on_click=register,
                args=("f2", 1, "f2_v1", "f2_v2"),
            )
            logger.info(f"submit 2: {submit}")
            return submit

    @staticmethod
    @st.cache
    def calc_2(a, b, c) -> float:
        logger.debug("inside calc_2")
        return a + b + c

    @staticmethod
    def __call__():
        logger.info(f"{'-':->50}")
        logger.info(f"{' Enter App ':^50}")
        logger.info(f"{'-':->50}")

        # ----- collect arguments for creating assets
        submit_0 = step_0_form()
        logger.info(f"{submit_0}")

        # ----- argument for step 1 has been collected
        if hasattr(register, "xlsm"):

            # ----- compute step 1
            if submit_0:
                register.xlsm.val = step_0_compute(*register.xlsm.args)

            # ----- display results of step 0
            # st.text(f"XLSM TEST STAY {register.xlsm.val}")
            step_0_visualize(register.xlsm.val)

            # ----- collect arguments for step 1
            submit_1 = App.f1()
            logger.info(f"{submit_1=}")

            # ----- argument for step 1 has been collected
            if hasattr(register, "f1"):

                # ----- compute step 1
                if submit_1:
                    register.f1.val = App.calc_1(*register.f1.args)

                # ----- display results of step 1
                st.text(f"F1 TEST STAY {register.f1.val}")

                # ----- collect arguments for step 2
                submit_2 = App.f2()
                logger.info(f"{submit_2=}")

                #  ----- the button in step 2 has been pressed
                if submit_2:

                    # ----- compute step 2
                    args = register.f2.args + (register.f1.val,)
                    register.f2.val = App.calc_2(*args)

                # ----- display results step 2
                if hasattr(register, "f2"):
                    st.text(f"F2 TEST STAY {register.f2.val}")

        logger.info(f"memo = {register}")

        logger.info(f"{'-':->50}")
        logger.info(f"{' Done ':^50}")
        logger.info(f"{'-':->50}")


if __name__ == "__main__":
    log.enable(log.LevelName.DEBUG)
    app()
