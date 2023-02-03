import streamlit as st
from portfolioqtopt.reader import read_welzia_stocks_file
from portfolioqtopt.assets import Assets
from portfolioqtopt.simulation import simulate_assets
from portfolioqtopt.optimization import optimize, SolverTypes
from portfolioqtopt import log
from pathlib import Path
from bokeh.plotting import figure
from bokeh.palettes import Turbo256
from bokeh.models import Legend
from typing import Hashable, Dict, Tuple, Union, Optional
from portfolioqtopt.assets import Scalar
import pandas as pd
import time

from loguru import logger

log.enable(log.LevelName.DEBUG)


class App:

    @staticmethod
    def load_xlsm_form() -> None:
        with st.form(key='xlsm'):
            logger.info("load data")
            uploaded_file = None
            sheet_name: Optional[str] = None

            uploaded_file = st.file_uploader(
                "Elige un fichero al formato Welzia",
                type="xlsm",
            )
            # uploaded_file = "/home/ggelabert/Projects/PortfolioQtOpt/data/Histórico_carteras_Welzia_2018.xlsm"

            if uploaded_file is not None:
                sheet_name = st.text_input(
                    "Elige el nombre de la hoja a leer.",
                    value="BBG (valores)",
                )

            if sheet_name is not None and uploaded_file is not None:


    def load_data(self) -> Optional[Assets]:
        logger.info("load data")
        uploaded_file = None
        sheet_name: Optional[str] = None

        uploaded_file = st.file_uploader(
            "Elige un fichero al formato Welzia",
            type="xlsm",
        )
        # uploaded_file = "/home/ggelabert/Projects/PortfolioQtOpt/data/Histórico_carteras_Welzia_2018.xlsm"

        if uploaded_file is not None:
            sheet_name = st.text_input(
                "Elige el nombre de la hoja a leer.",
                value="BBG (valores)",
            )

        if sheet_name is not None and uploaded_file is not None:
            df = read_welzia_stocks_file(uploaded_file, sheet_name)
            f"Tabla `{Path(uploaded_file.name).stem}` hoja `{sheet_name}`."
            f"Se ha cargo el valor de {df.shape[1]} fondos durante {df.shape[0]} días."
            return Assets(df=df)
        else:
            return None

    def visualize_df(self, assets: Assets) -> None:
        logger.info("visualize df")
        # ----- bokeh visualization
        p = figure(
            title='simple line example',
            x_axis_label='Fechas',
            y_axis_label='Activos',
            x_axis_type='datetime',
            y_axis_type='log',
            background_fill_color="#fafafa",
        )

        for i, column in enumerate(assets.df):
            color = Turbo256[int(256 * i / assets.df.shape[1])]
            p.line(assets.df.index, assets.df[column], legend_label=column, line_width=1, color=color)
        st.bokeh_chart(p, use_container_width=True)

    def get_simulation_args(self, assets: Assets) -> Optional[Tuple[Assets, int, Dict[Union[Scalar, Tuple[Hashable, ...]], float]]]:
        logger.info("get simulation args")
        with st.form(key='simulation'):
            with st.expander(
                "Rellenar las distintas opciones si se necesita o seguir con los "
                "valores por defecto."
            ):
                ns = st.number_input("numero de días a simular", value = 256, step=1)

                st.text("Elije el retorno para cada activo:")

                sliders: Dict[Union[Scalar, Tuple[Hashable, ...]], float] = {}

                for fund, er in zip(assets.df, assets.anual_returns.tolist()):
                    sliders[fund] = st.slider(str(fund), -1.0, 3.0, er)  # type: ignore[index]
            submitted = st.form_submit_button("Run simulation")
            logger.debug(f"{submitted=}")
        if submitted:
            return assets, int(ns), sliders
        else:
            return None

    @staticmethod
    def optimize_portfolio(assets: Assets, b: float,
        w: int,
        theta1: float,
        theta2: float,
        theta3: float,
        solver: SolverTypes,
        token_api: str,
        steps: int,
        verbose: bool = False,
    ):
        logger.info(f"run optimization from app")
        with st.spinner("Optimización en curso..."):
            time.sleep(1)
        st.text("Hola")
        with st.spinner("Optimización en curso..."):
            time.sleep(1)
        st.text("Error!")
        with st.spinner("Optimización en curso..."):
            # _, interpretation = optimize(assets, b, w, theta1, theta2, theta3, solver, token_api, steps, verbose=verbose)
            from portfolioqtopt.optimization.interpreter import Interpretation
            import numpy as np
            interpretation  = Interpretation(
                selected_indexes=pd.Index(['A', 'C', 'D'], dtype='object'),
                investments=np.array([0.75 , 0.125, 0.125]),
                expected_returns=44.5,
                risk=17.153170260916784,
                sharpe_ratio=2.594272622676201,
            )

            st.text("I am the optimization process!")
        if interpretation is not  None:
            st.success("Hecho!", icon="✅")
            st.markdown(f"{interpretation.to_str()}")
        else:
            st.markdown("Optimization fails. Run it again!")

    @staticmethod
    def optimize_portfolio_callback(optimization_args):
        # Store optimization args inside call back
        st.session_state.optimization_args = optimization_args
        logger.info(f"optimization callback: {optimization_args}")

    def optimize_portfolio_form(self, assets: Assets) -> None:  # -> Optional[Tuple[float, int, float, float, float, SolverTypes, str, int]]:
        logger.info("optimize portfolio")
        with st.form(key='optimization'):
            with st.expander(
                "Rellenar las distintas opciones si se necesita o seguir con los "
                "valores por defecto (lo aconsejado)."
            ):
                budget = st.number_input("budget", value=1.0)
                w = st.number_input("granularity", value=5)
                theta1 = st.number_input("theta1", value=0.9)
                theta2 = st.number_input("theta2", value=0.4)
                theta3 = st.number_input("theta3", value=0.1)
                steps = st.number_input("steps", value=5, step=1)

            args = (
                float(budget),
                int(w),
                float(theta1),
                float(theta2),
                float(theta3),
                SolverTypes.hybrid_solver,
                "",
                int(steps),
            )
 
            submitted = st.form_submit_button("Run optimization", on_click=App.optimize_portfolio_callback(args))  #, on_click=lambda: App._optimize_portfolio(assets, *args))
            """logger.debug(f"{submitted=}")
            if submitted:
                logger.debug("submit form!")
                time.sleep(2)
                return args
            else:
                logger.debug("form not submit!")
                return None"""

    st.cache(hash_funcs={"portfolioqtopt.assets.Assets": id})
    def get_future(self, assets: Assets) -> Optional[Assets]:
        logger.info("get future")
        # future: Optional[Assets] = None
        logger.debug(f"asset is None? {assets is None}")

        st.markdown("## 2. Simulation de los datos futuros")
        st.markdown(
            "Vamos a simular los precios futuros de los assets que corresponden a "
            "los valores a rellenar.",
        )
        simulation_args = self.get_simulation_args(assets)
        if simulation_args is not None:
            with st.spinner("Generation en curso ..."):
                future = simulate_assets(*simulation_args)
            st.success("Hecho!", icon="✅")
            with st.expander("Gráfico de los datos."):
                self.visualize_df(future)

            return future
        else:
            return None

    def __call__(self) -> None:

        st.title("Optimización de un portfolio con ordenador cuántico")

        st.markdown("## 1. Carga de los datos históricos")

        history = self.load_data()
    
        if history is not None:
            with st.expander("Visualizar los datos cargados."):
                st.dataframe(data=history.df)
            with st.expander("Gráfico los datos."):
                self.visualize_df(history)

            future = self.get_future(history)

            logger.info(f"future is None? {future is None}")

            if future is not None:
                st.markdown("## 3. Optimizar el portfolio")
                st.markdown("Buscamos maximizar el retornos esperado minimizando el riesgo")

                self.optimize_portfolio_form(future)
                optimization_args = st.session_state.get("optimization_args", False)

                if optimization_args:
                    self.optimize_portfolio(future, *optimization_args)
                    st.session_state.optimization_args = False
            
        logger.debug(f"{st.session_state=}")
                

def app():
    logger.debug(f"{st.session_state.to_dict()=}")
    return App()()

if __name__ == "__main__":
    app()