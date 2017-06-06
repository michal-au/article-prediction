import os
import pandas
import pytest

from ...lib.utils import read_settings

__all__ = (
    'df_train',
    'df_heldout',
    'df_test',
    'df_train_tun',
    'df_heldout_tun',
    'df_test_tun',
    'df_simple'
)


@pytest.fixture
def df_train():
    settings = read_settings()
    path = settings.get("paths", "dataFeatures")
    return pandas.read_pickle(os.path.join(path, "train.pkl"))


@pytest.fixture
def df_heldout():
    settings = read_settings()
    path = settings.get("paths", "dataFeatures")
    return pandas.read_pickle(os.path.join(path, "heldout.pkl"))


@pytest.fixture
def df_test():
    settings = read_settings()
    path = settings.get("paths", "dataFeatures")
    return pandas.read_pickle(os.path.join(path, "test.pkl"))


@pytest.fixture
def df_train_tun():
    settings = read_settings()
    path = settings.get("paths", "dataFeatures")
    return pandas.read_pickle(os.path.join(path, "tun_train.pkl"))


@pytest.fixture
def df_heldout_tun():
    settings = read_settings()
    path = settings.get("paths", "dataFeatures")
    return pandas.read_pickle(os.path.join(path, "tun_heldout.pkl"))


@pytest.fixture
def df_test_tun():
    settings = read_settings()
    path = settings.get("paths", "dataFeatures")
    return pandas.read_pickle(os.path.join(path, "tun_test.pkl"))


@pytest.fixture
def df_simple(df_train):
    return df_train.sample(n=100, random_state=42)
