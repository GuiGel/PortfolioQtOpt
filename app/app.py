import streamlit as st
from portfolioqtopt.reader import read_welzia_stocks_file
from portfolioqtopt.assets import Assets
from portfolioqtopt.simulation import simulate_assets
from portfolioqtopt.optimization import optimize
from pathlib import Path
from bokeh.plotting import figure
from bokeh.palettes import Turbo256
from bokeh.models import Legend
from typing import Hashable, Dict, Tuple, Union


# uploaded_file = st.file_uploader("Elige un fichero al formato Welzia", type="xlsm")
uploaded_file = "/home/ggelabert/Projects/PortfolioQtOpt/data/Histórico_carteras_Welzia_2018.xlsm"


if uploaded_file is not None:

    # sheet_name = st.text_input("Elige el nombre de la hoja a leer.")
    sheet_name = "BBG (valores)"

    if len(sheet_name) != 0:
        df = read_welzia_stocks_file(uploaded_file, sheet_name)
        # st.text(f"Tabla `{Path(uploaded_file.name).stem}` hoja `{sheet_name}`")
        st.dataframe(data=df)

        # ----- bokeh visualization
        st.text("We can visualize the original datas")

        p = figure(
            title='simple line example',
            x_axis_label='Fechas',
            y_axis_label='Activos',
            x_axis_type='datetime',
            y_axis_type='log',
        )

        for i, column in enumerate(df):
            color = Turbo256[int(256 * i / df.shape[1])]
            p.line(df.index, df[column], legend_label=column, line_width=1, color=color)
        st.bokeh_chart(p, use_container_width=True)

        # ----- simulation part

        st.title("Simulate the future data")

        historical_assets = Assets(df=df)

        with st.form(key='columns_in_form'):
            st.text("Elije el numero días simulados:")
            ns = st.number_input('Insert a number', min_value = 256, step=1)

            from portfolioqtopt.assets import Scalar

            st.text("Elije el retorno para cada activo:")

            sliders: Dict[Union[Scalar, Tuple[Hashable, ...]], float] = {}

            for fund, er in zip(historical_assets.df, historical_assets.anual_returns.tolist()):
                sliders[fund] = st.slider(str(fund), -1.0, 3.0, er)  # type: ignore[index]
            submitted = st.form_submit_button("Enviar")

        if submitted:
            future_assets = simulate_assets(historical_assets, ns.as_integer_ratio()[0], sliders)

            # ----- bokeh visualization
            st.text("We can visualize the future datas")

            p = figure(
                title='Future assets',
                x_axis_label='numero del día',
                y_axis_label='Activos',
                x_axis_type='datetime',
                y_axis_type='log',
            )

            for i, column in enumerate(df):
                color = Turbo256[int(256 * i / future_assets.df.shape[1])]
                p.line(future_assets.df.index, future_assets.df[column], legend_label=column, line_width=2, color=color)
            st.bokeh_chart(p, use_container_width=True)

            # ----- optimization part
            st.title("Buscar el mejor portfolio")

            """_, interpretation = optimize(
                future_assets,
                b=budget,
                w=w,
                theta1=theta1,
                theta2=theta2,
                theta3=theta3,
                solver=solver,
                token_api=token_api,
                steps=steps,
            )"""
