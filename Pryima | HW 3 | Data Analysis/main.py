from pathlib import Path
import io

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler, StandardScaler


BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "datasets" / "diagnostic" / "wdbc.data"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Назви колонок скопійовані з 1-го ДЗ
COLUMN_NAMES = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


def load_dataset(path: Path = DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    # невелике "забруднення": додамо штучну колонку з датами у різних форматах
    df = add_raw_dates(df)
    return df


def add_raw_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base_date = pd.Timestamp("2024-01-01")
    patterns = [
        "%Y-%m-%d",     # 2024-01-05
        "%d/%m/%Y",     # 05/01/2024
        "%m.%d.%Y",     # 01.05.2024
        "%Y/%m/%d",     # 2024/01/05
    ]

    dates_raw: list[str] = []
    for i in range(len(df)):
        d = base_date + pd.Timedelta(days=int(i))
        fmt = patterns[i % len(patterns)]
        dates_raw.append(d.strftime(fmt))

    df["measurement_date_raw"] = dates_raw
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if df[col].dtype.kind in "biufc":  # numeric
            median = df[col].median()
            df[col] = df[col].fillna(median)
        else:
            df[col] = df[col].fillna("unknown")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "measurement_date_raw" in df.columns:
        df["measurement_date"] = pd.to_datetime(
            df["measurement_date_raw"],
            errors="coerce",
            dayfirst=False,
        )
        df["measurement_date_num"] = df["measurement_date"].view("int64")
    return df


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].astype(str).str.strip().str.upper()
        df["diagnosis"] = df["diagnosis"].astype("category")
        df["diagnosis_malignant"] = df["diagnosis"].map({"M": 1, "B": 0}).astype(
            "Int64"
        )

    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "diagnosis" in df.columns:
        df["diagnosis"] = (
            df["diagnosis"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"MALIGNANT": "M", "BENIGN": "B"})
        )

    return df


def select_feature_columns_for_numeric_processing(df: pd.DataFrame) -> list[str]:
    exclude = {
        "id",
        "diagnosis",
        "measurement_date_raw",
        "measurement_date",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric_cols if c not in exclude]


def fix_numeric_outliers_and_errors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in select_feature_columns_for_numeric_processing(df):
        low = df[col].quantile(0.01)
        high = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=low, upper=high)

    return df


def scale_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = select_feature_columns_for_numeric_processing(df)
    if len(numeric_cols) == 0:
        return df

    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    standard_scaled = standard_scaler.fit_transform(df[numeric_cols])
    minmax_scaled = minmax_scaler.fit_transform(df[numeric_cols])

    for i, col in enumerate(numeric_cols):
        df[f"{col}_zscore"] = standard_scaled[:, i]
        df[f"{col}_minmax"] = minmax_scaled[:, i]

    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Видалено дублікатів: {before - after}")
    return df


def df_info_to_string(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer, verbose=True)
    return buffer.getvalue()


def format_missing_summary(df: pd.DataFrame, max_rows: int = 20) -> str:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return "No missing values."
    if len(missing) > max_rows:
        missing = missing.head(max_rows)
        return f"{missing.to_string()}\n... (truncated)"
    return missing.to_string()


def get_techniques_list() -> list[str]:
    return [
        "Очистка назв колонок",
        "Робота з пропущеними значеннями",
        "Зчитування та парсинг дат",
        "Зміна типів даних",
        "Робота з текстовими даними",
        "Обробка числових даних (winsorization)",
        "Нормалізація та масштабування",
        "Виявлення та видалення дублікатів",
    ]


def save_cleaning_report(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path = Path(output_path)

    raw_info = df_info_to_string(raw_df)
    clean_info = df_info_to_string(clean_df)

    text_lines = [
        "HW3: Data Cleaning Report",
        "",
        "Techniques used (8/8):",
        *[f"- {t}" for t in get_techniques_list()],
        "",
        f"Raw shape: {raw_df.shape}",
        f"Clean shape: {clean_df.shape}",
        "",
        "=== Missing values (raw) ===",
        format_missing_summary(raw_df),
        "",
        "=== Missing values (clean) ===",
        format_missing_summary(clean_df),
        "",
        "=== RAW df.info() ===",
        raw_info,
        "",
        "=== CLEAN df.info() ===",
        clean_info,
    ]
    text = "\n".join(text_lines)

    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            text,
            ha="left",
            va="top",
            fontsize=9,
            fontfamily="monospace",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    df_raw = load_dataset()
    print("=== Перші рядки сирих даних (wdbc) ===")
    print(df_raw.head())

    df = clean_column_names(df_raw)
    df = handle_missing_values(df)
    df = parse_dates(df)
    df = convert_dtypes(df)
    df = clean_text_columns(df)
    df = fix_numeric_outliers_and_errors(df)
    df = scale_numeric_features(df)
    df = drop_duplicates(df)

    print("\n=== Очищені та підготовлені дані (перші рядки) ===")
    print(df.head())

    output_path = BASE_DIR / "cleaned_wdbc.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nОчищений датасет збережено до: {output_path}")

    report_path = PLOTS_DIR / "cleaning_report.pdf"
    save_cleaning_report(df_raw, df, report_path)
    print(f"Звіт по очистці збережено до: {report_path}")


if __name__ == "__main__":
    main()

