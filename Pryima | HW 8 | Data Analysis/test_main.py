# Тести для HW 8 — експлоративний аналіз WDBC
"""
Перевірка: завантаження даних, форма, відсутність пропусків,
діагнози B/M, викиди IQR, кореляція radius–area, різниця B vs M.
Запуск: pytest test_main.py -v
"""

import sys
from pathlib import Path

import pytest

# додати корінь проекту в path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import main as m


@pytest.fixture(scope="module")
def df():
    return m.load_data()


def test_load_data_returns_dataframe(df):
    import pandas as pd
    assert isinstance(df, pd.DataFrame)


def test_shape_and_columns(df):
    assert df.shape[0] == 569
    assert df.shape[1] == 32
    assert "diagnosis" in df.columns
    assert "radius_mean" in df.columns
    assert "area_mean" in df.columns


def test_no_missing_values(df):
    assert df.isna().sum().sum() == 0


def test_diagnosis_only_B_and_M(df):
    assert set(df["diagnosis"].unique()) == {"B", "M"}
    assert df["diagnosis"].value_counts().get("B", 0) == 357
    assert df["diagnosis"].value_counts().get("M", 0) == 212


def test_outliers_iqr_counts(df):
    out_radius = m.get_outliers_iqr(df["radius_mean"]).sum()
    out_area = m.get_outliers_iqr(df["area_mean"]).sum()
    assert out_radius >= 0 and out_radius <= 50
    assert out_area >= 0 and out_area <= 50


def test_correlation_radius_area_high(df):
    r = df["radius_mean"].corr(df["area_mean"])
    assert r > 0.95


def test_B_vs_M_radius_mean_differ(df):
    mean_b = df.loc[df["diagnosis"] == "B", "radius_mean"].mean()
    mean_m = df.loc[df["diagnosis"] == "M", "radius_mean"].mean()
    assert mean_m > mean_b


def test_key_features_present(df):
    for col in m.KEY_FEATURES:
        assert col in df.columns
        assert df[col].dtype in ("float64", "int64")


def test_describe_has_expected_stats(df):
    desc = df[m.KEY_FEATURES].describe()
    assert "mean" in desc.index
    assert "std" in desc.index
    assert "50%" in desc.index


def test_corr_matrix_shape(df):
    sub = df[m.CORR_FEATURES]
    corr = sub.corr()
    assert corr.shape == (len(m.CORR_FEATURES), len(m.CORR_FEATURES))
    assert corr.loc["radius_mean", "radius_mean"] == pytest.approx(1.0)
