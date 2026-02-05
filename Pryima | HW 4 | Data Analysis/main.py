from pathlib import Path
import io
import sys

import pandas as pd


BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "datasets" / "diagnostic" / "wdbc.data"

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


OUTPUT_BUFFER = io.StringIO()
_ORIGINAL_STDOUT = sys.stdout


class TeeOutput:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, obj):
        for s in self.streams:
            s.write(obj)
            s.flush()
        OUTPUT_BUFFER.write(obj)

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = TeeOutput(_ORIGINAL_STDOUT)


def load_main_dataset(path: Path = DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    return df


def create_lookup_table(df: pd.DataFrame) -> pd.DataFrame:
    unique_ids = (
        df[["id", "diagnosis"]]
        .drop_duplicates()
        .sample(frac=0.7, random_state=42)
        .reset_index(drop=True)
    )

    diagnosis_name_map = {
        "M": "Malignant tumor",
        "B": "Benign tumor",
    }
    lookup = unique_ids.assign(
        diagnosis_full=lambda d: d["diagnosis"].map(diagnosis_name_map)
    )
    return lookup


def demonstrate_merges(main_df: pd.DataFrame, lookup: pd.DataFrame) -> None:
    inner_join = pd.merge(
        main_df,
        lookup,
        on="id",
        how="inner",
        suffixes=("_main", "_lookup"),
    )
    left_join = pd.merge(
        main_df,
        lookup,
        on="id",
        how="left",
        suffixes=("_main", "_lookup"),
    )

    print("=== INNER JOIN (only rows with a match in lookup) ===")
    print(f"Inner join shape: {inner_join.shape}")
    print(
        inner_join[
            ["id", "diagnosis_main", "diagnosis_lookup", "diagnosis_full"]
        ].head()
    )

    print(
        "\n=== LEFT JOIN (all rows from main table, matches from lookup when available) ==="
    )
    print(f"Left join shape: {left_join.shape}")
    print(
        left_join[
            ["id", "diagnosis_main", "diagnosis_lookup", "diagnosis_full"]
        ].head(10)
    )

    return left_join


def demonstrate_concat(main_df: pd.DataFrame) -> pd.DataFrame:
    sample = main_df.sample(3, random_state=1).reset_index(drop=True)

    new_rows = sample.copy()
    new_rows["id"] = new_rows["id"] + 10_000_000
    new_rows["diagnosis"] = new_rows["diagnosis"].replace({"M": "B", "B": "M"})

    concatenated = pd.concat([main_df, new_rows], ignore_index=True)

    print("\n=== CONCAT (appending new rows) ===")
    print(f"Original shape: {main_df.shape}, after concat: {concatenated.shape}")
    print("Example of newly added rows:")
    print(new_rows[["id", "diagnosis", "radius_mean", "area_mean"]])

    return concatenated


def demonstrate_melt(df: pd.DataFrame) -> pd.DataFrame:
    mean_cols = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
    ]

    melted = df.melt(
        id_vars=["id", "diagnosis"],
        value_vars=mean_cols,
        var_name="metric",
        value_name="value",
    )

    print("\n=== MELT (long format for mean metrics) ===")
    print(melted.head(10))
    return melted


def demonstrate_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    mean_cols = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
    ]

    pivot = df.pivot_table(
        index="diagnosis",
        values=mean_cols,
        aggfunc="mean",
    )

    print("\n=== PIVOT TABLE (mean values by diagnosis) ===")
    print(pivot)
    return pivot


def demonstrate_assign(df_with_lookup: pd.DataFrame) -> pd.DataFrame:
    result = df_with_lookup.assign(
        radius_area_ratio=lambda d: d["radius_mean"] / d["area_mean"],
        is_large_radius=lambda d: d["radius_mean"] > d["radius_mean"].median(),
        high_risk=lambda d: (d["radius_mean"] > d["radius_mean"].median())
        & (d["diagnosis_main"] == "M"),
    )

    print("\n=== ASSIGN (new calculated features) ===")
    print(
        result[
            [
                "id",
                "diagnosis_main",
                "radius_mean",
                "area_mean",
                "radius_area_ratio",
                "is_large_radius",
                "high_risk",
            ]
        ].head(10)
    )
    return result


def generate_insights(df_merged: pd.DataFrame, pivot_df: pd.DataFrame) -> None:
    print("\n" + "=" * 40)
    print("DATA INSIGHTS REPORT")
    print("=" * 40)

    total = len(df_merged)
    matched = df_merged["diagnosis_full"].notna().sum()
    print(f"1. JOIN ANALYSIS: Left join kept {total} rows.")
    print(
        f"   Successfully matched {matched} records ({matched / total:.1%}) with descriptive labels."
    )

    if "M" in pivot_df.index and "B" in pivot_df.index:
        m_rad = pivot_df.loc["M", "radius_mean"]
        b_rad = pivot_df.loc["B", "radius_mean"]
        print(
            f"2. CATEGORY ANALYSIS: Malignant mean radius ({m_rad:.2f}) "
            f"is larger than Benign mean radius ({b_rad:.2f}) "
            f"by {((m_rad - b_rad) / b_rad):.1%}."
        )

    print(
        "3. FEATURE ENGINEERING: Derived metrics like radius_area_ratio and high_risk "
        "help to highlight potentially dangerous tumor profiles."
    )
    print("=" * 40)


def main() -> None:
    main_df = load_main_dataset()
    print("=== HEAD OF MAIN DATASET (wdbc) ===")
    print(main_df.head())

    lookup_df = create_lookup_table(main_df)
    print("\n=== LOOKUP TABLE ===")
    print(lookup_df.head())

    left_join = demonstrate_merges(main_df, lookup_df)

    df_with_new_rows = demonstrate_concat(main_df)

    melted = demonstrate_melt(main_df)

    pivot = demonstrate_pivot_table(main_df)

    enriched = demonstrate_assign(left_join)

    generate_insights(enriched, pivot)

    print("\n=== SUMMARY ===")
    print(f"Main dataset shape: {main_df.shape}")
    print(f"Shape after concat (with new rows): {df_with_new_rows.shape}")
    print(f"Long format (melt) shape: {melted.shape}")


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout = _ORIGINAL_STDOUT
        results_path = BASE_DIR / "results.txt"
        with results_path.open("w", encoding="utf-8") as f:
            f.write(OUTPUT_BUFFER.getvalue())

