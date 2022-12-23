"""This module contains the tests for the data readers."""
import numpy as np
import pandas as pd

from portfolioqtopt.reader import read_welzia_stocks_file


def test_read_welzia_stocks_file(welzia):
    """Test that the welzia file are read as expected."""
    df = read_welzia_stocks_file(*welzia)

    # test that the numpy array inside df is the expected one
    array = df.to_numpy()
    expected_array = np.array(
        [
            [221.86, 149.25, 14.23],
            [221.73, 148.18, 14.09],
            [221.86, 147.96, 14.21],
            [222.12, 149.31, 14.33],
        ]
    )

    np.testing.assert_equal(array, expected_array)
    assert array.shape == (4, 3)

    # test that the pd.MultiIndex of df is the expected one
    column_arrays = [
        ["LU0151325312", "LU0329760937", "LU0963540371"],
        [
            "Candriam Bonds - Credit Opport",
            "DWS Invest Global Infrastructu",
            "Fidelity Funds - America Fund",
        ],
        [
            "DEXHISI LX Equity",
            "DWSGIFC LX Equity",
            "FIAYEHG LX Equity",
        ],
        ["PX_LAST", "PX_LAST", "PX_LAST"],
    ]
    column_names = [None, "Name", "TICKER_AND_EXCH_CODE", "Date"]
    columns = pd.MultiIndex.from_arrays(column_arrays, names=column_names)

    assert isinstance(df.columns, pd.MultiIndex)  # To avoid mypy complains
    assert pd.MultiIndex.equals(df.columns, columns)
