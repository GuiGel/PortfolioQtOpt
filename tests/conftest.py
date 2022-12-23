import pytest

from . import ROOT_DIR


@pytest.fixture()
def welzia():
    file_path = ROOT_DIR / "test_reader" / "data_welzia.xlsm"
    sheet_name = "BBG (valores)"
    return (file_path, sheet_name)


def test_welzia(welzia):
    (file_path, sheet_name) = welzia
    assert file_path == ROOT_DIR / "test_reader" / "data_welzia.xlsm"
    assert sheet_name == "BBG (valores)"
