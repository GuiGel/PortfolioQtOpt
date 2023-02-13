import streamlit as st
from bokeh.palettes import Turbo256  # type: ignore[import]
from bokeh.plotting import figure  # type: ignore[import]
from loguru import logger

from qoptimiza.assets import Assets


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
