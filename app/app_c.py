import streamlit as st
from portfolioqtopt.reader import read_welzia_stocks_file
from portfolioqtopt.assets import Assets
from portfolioqtopt.simulation import simulate_assets
from portfolioqtopt.optimization import optimize
from pathlib import Path
from bokeh.plotting import figure
from bokeh.palettes import Turbo256
from bokeh.models import Legend
from typing import Hashable, Dict, Tuple, Union, Optional
from portfolioqtopt.assets import Scalar
import pandas as pd


class App:

    def load_data(self) -> Optional[Assets]:
        uploaded_file = None
        sheet_name: Optional[str] = None

        uploaded_file = st.file_uploader("Elige un fichero al formato Welzia", type="xlsm")
        # uploaded_file = "/home/ggelabert/Projects/PortfolioQtOpt/data/Histórico_carteras_Welzia_2018.xlsm"

        if uploaded_file is not None:
            sheet_name = st.text_input("Elige el nombre de la hoja a leer.", value="BBG (valores)")

        if sheet_name is not None and uploaded_file is not None:
            df = read_welzia_stocks_file(uploaded_file, sheet_name)
            # f"Tabla `{Path(uploaded_file.name).stem}` hoja `{sheet_name}`")
            f"Se ha cargo el valor de {df.shape[1]} fondos durante {df.shape[0]} días."
            return Assets(df=df)
        else:
            return None

    def visualize_df(self, assets: Assets) -> None:
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
        with st.form(key='columns_in_form'):
            with st.expander(
                "Rellenar las distintas opciones si se necesita o seguir con los "
                "valores por defecto."
            ):
                ns = st.number_input("numero de días a simular", min_value = 256, step=1)

                st.text("Elije el retorno para cada activo:")

                sliders: Dict[Union[Scalar, Tuple[Hashable, ...]], float] = {}

                for fund, er in zip(assets.df, assets.anual_returns.tolist()):
                    sliders[fund] = st.slider(str(fund), -1.0, 3.0, er)  # type: ignore[index]
            submitted = st.form_submit_button("Enviar")
        if submitted:
            return assets, ns.as_integer_ratio()[0], sliders
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

        
        future: Optional[Assets] = None
        if history is not None:
            st.markdown("## 2. Simulation de los datos futuros")
            # st.markdown("Rellenar las distintas opciones si se necesita!")
            simulation_args = self.get_simulation_args(history)
            if simulation_args is not None:
                st.markdown("Generar valores futuros")
                future = simulate_assets(*simulation_args)
                st.success("Hecho!", icon="✅")
                with st.expander("Gráfico de los datos."):
                    self.visualize_df(future)

        if future is not None:
            st.markdown("## 3. Optimizar el portfolio")

def app():
    return App()()

if __name__ == "__main__":
    app()