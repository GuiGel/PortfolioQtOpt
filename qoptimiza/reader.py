"""This module contains the data readers."""
from pathlib import Path
from typing import Union

import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile


def read_welzia_stocks_file(
    file_path: Union[Path, str, UploadedFile],
    sheet_name: str,
) -> pd.DataFrame:
    """Read welzia .xlsx input files as a pandas ``DataFrame``.

    Args:
        file_path (Union[Path, str]): File path.
        sheet_name (str): Name of the Excel sheet where the data are located.

    Raises:
        FileExistsError: The file path doesn't exists.

    Returns:
        pd.DataFrame: The stock prices read as a pandas ``DataFrame``.
    """
    if isinstance(file_path, (str, Path)):
        if not Path(file_path).exists():
            raise FileExistsError()

    df = (
        pd.read_excel(
            io=file_path,
            sheet_name=sheet_name,
            skiprows=5,
            header=[0, 1, 2, 3],
            index_col=1,
            parse_dates=True,
            keep_default_na=True,
            thousands=",",
        )
        .dropna(axis=1)
        .droplevel([0, 1, 3], axis="columns")
    )
    return df


if __name__ == "__main__":

    file_path = "/home/ggelabert/Projects/PortfolioQtOpt/data/Histórico_carteras_Welzia_2018.xlsm"
    sheet_name = "BBG (valores)"

    df = read_welzia_stocks_file(file_path, sheet_name)
    print(df)
    print(df.columns)
    print(df.columns.names)
    print(df.columns.get_level_values(0))
