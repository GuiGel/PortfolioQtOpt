import streamlit as st
from bokeh.models import Legend
from bokeh.palettes import Turbo256
from bokeh.plotting import figure
from loguru import logger

from qoptimiza.assets import Assets


def visualize_assets(assets: Assets) -> None:
    """Create a `bokeh` graph of the assets.

    Args:
        assets (Assets): The assets to plot.
    """
    logger.info("visualize df")

    # Sort column in decreasing order based on their last price. This is done to have
    # an ordered color plot. By this way the first asset that appears on the top of
    # the legend correspond to the highest value on the left of the plot.


    last_row = assets.df.tail(1).iloc[-1, :]
    sorted_columns = last_row.argsort().to_numpy()[::-1]

    # ----- bokeh visualization
    p = figure(
        # title="simple line example",
        x_axis_label="Fechas",
        y_axis_label="Activos",
        x_axis_type="datetime",
        y_axis_type="log",
        background_fill_color="#fafafa",
        height=assets.m * 26,
        width=1000,
    )

    legend_it = []
    for i, column in enumerate(assets[sorted_columns].df):
        color = Turbo256[int(256 * i / assets.df.shape[1])]
        c = p.line(
            assets.df.index,
            assets.df[column],
            line_width=1,
            color=color,
        )
        # append a number to the column name
        column = f"{i+1} {column}"

        legend_it.append((column, [c]))

    legend = Legend(items=legend_it)
    legend.title = "Assets"
    legend.click_policy = "hide"
    legend.border_line_width = 1
    legend.border_line_color = "grey"
    legend.background_fill_color = "#fafafa"
    legend.label_text_font_size = "8px"

    p.add_layout(legend, "right")

    st.bokeh_chart(p, use_container_width=False)
