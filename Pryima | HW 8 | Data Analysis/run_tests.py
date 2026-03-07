#!/usr/bin/env python3
"""
Запуск тестів HW 8 без pytest: python3 run_tests.py
(Якщо встановлено pytest: python3 -m pytest test_main.py -v)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import main as m


def run():
    df = m.load_data()
    errors = []

    if df.shape != (569, 32):
        errors.append(f"shape: got {df.shape}, expected (569, 32)")
    if df.isna().sum().sum() != 0:
        errors.append("expected no missing values")
    if set(df["diagnosis"].unique()) != {"B", "M"}:
        errors.append(f"diagnosis: got {df['diagnosis'].unique().tolist()}")
    if df["diagnosis"].value_counts().get("B", 0) != 357:
        errors.append("expected 357 B")
    if df["diagnosis"].value_counts().get("M", 0) != 212:
        errors.append("expected 212 M")
    if df["radius_mean"].corr(df["area_mean"]) <= 0.95:
        errors.append("expected correlation radius_mean–area_mean > 0.95")
    mean_b = df.loc[df["diagnosis"] == "B", "radius_mean"].mean()
    mean_m = df.loc[df["diagnosis"] == "M", "radius_mean"].mean()
    if mean_m <= mean_b:
        errors.append("expected M radius_mean > B radius_mean")
    out_r = m.get_outliers_iqr(df["radius_mean"]).sum()
    out_a = m.get_outliers_iqr(df["area_mean"]).sum()
    if not (0 <= out_r <= 50 and 0 <= out_a <= 50):
        errors.append(f"outlier counts out of expected range: radius={out_r}, area={out_a}")

    if errors:
        for e in errors:
            print("FAIL:", e)
        sys.exit(1)
    print("OK: всі перевірки пройдено.")


if __name__ == "__main__":
    run()
