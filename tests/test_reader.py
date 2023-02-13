"""This module contains the tests for the data readers."""
import numpy as np
import pandas as pd

from qoptimiza.reader import read_welzia_stocks_file


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

    # test that the name of the columns are the expected ones
    column_arrays = ["DEXHISI LX Equity", "DWSGIFC LX Equity", "FIAYEHG LX Equity"]
    column_names = ["TICKER_AND_EXCH_CODE"]
    columns = pd.Index(column_arrays)

    assert isinstance(df.columns, pd.Index)  # To avoid mypy complains
    assert pd.Index.equals(df.columns, columns)
    assert column_names == df.columns.names
