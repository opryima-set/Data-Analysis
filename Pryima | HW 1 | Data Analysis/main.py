from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


DATA_FILE = Path(__file__).parent / "datasets" / "diagnostic" / "wdbc.data"

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
    """Читає датасет, додає назви колонок."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path, header=None, names=COLUMN_NAMES)


def summarize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Обчислює базову статистику для числових ознак."""
    numeric_df = df.select_dtypes(include=[np.number])
    summary = pd.DataFrame(
        {
            "mode": numeric_df.mode().iloc[0],
            "median": numeric_df.median(),
            "mean": numeric_df.mean(),
            "variance": numeric_df.var(),
            "q25": numeric_df.quantile(0.25),
            "q50": numeric_df.quantile(0.50),
            "q75": numeric_df.quantile(0.75),
        }
    )
    return summary


def summarize_categorical(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Обчислює ті самі статистики для категорійних ознак через їх коди.
    Коди будуються у відсортованому порядку категорій, щоб інтерпретація була прозорою.
    """
    cat_df = df.select_dtypes(exclude=[np.number])
    result: Dict[str, Dict[str, Any]] = {}

    for col in cat_df.columns:
        series = cat_df[col]
        ordered = series.astype(
            pd.CategoricalDtype(categories=sorted(series.unique()))
        )
        codes = ordered.cat.codes
        mode_vals = series.mode()

        result[col] = {
            "category_to_code": {
                category: code for code, category in enumerate(ordered.cat.categories)
            },
            "mode": mode_vals.iloc[0] if not mode_vals.empty else None,
            "mean_code": float(codes.mean()),
            "median_code": float(codes.median()),
            "variance_code": float(codes.var()),
            "q25_code": float(codes.quantile(0.25)),
            "q50_code": float(codes.quantile(0.50)),
            "q75_code": float(codes.quantile(0.75)),
            "value_counts": series.value_counts().to_dict(),
        }

    return result


def main() -> None:
    df = load_dataset()

    numeric_summary = summarize_numeric(df.drop(columns=["diagnosis"]))
    categorical_summary = summarize_categorical(df[["diagnosis"]])

    pd.set_option("display.max_rows", None)
    print("=== Числові ознаки ===")
    print(numeric_summary)

    print("\n=== Категорійні ознаки (статистики за кодами категорій) ===")
    for col, stats in categorical_summary.items():
        print(f"\nКолонка: {col}")
        print(f"Мапа категорія -> код: {stats['category_to_code']}")
        print(f"Мода: {stats['mode']}")
        print(f"Розподіл значень: {stats['value_counts']}")
        print(
            "Середнє/медіана/дисперсія кодів: "
            f"{stats['mean_code']:.3f} / {stats['median_code']} / {stats['variance_code']:.3f}"
        )
        print(
            "Квантілі кодів 25% / 50% / 75%: "
            f"{stats['q25_code']} / {stats['q50_code']} / {stats['q75_code']}"
        )

    output_pdf = Path(__file__).parent / "results.pdf"
    save_to_pdf(numeric_summary, categorical_summary, output_pdf)
    print(f"\nРезультати також збережено у файл: {output_pdf}")


def save_to_pdf(
    numeric_summary: pd.DataFrame,
    categorical_summary: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Створює простий звіт у PDF з числовими та категоріальними статистиками.
    """
    output_path = Path(output_path)

    num_text = numeric_summary.to_string()

    cat_lines = ["Категорійні ознаки (коди у порядку сортування категорій):"]
    for col, stats in categorical_summary.items():
        cat_lines.append(f"\nКолонка: {col}")
        cat_lines.append(f"Мапа категорія -> код: {stats['category_to_code']}")
        cat_lines.append(f"Мода: {stats['mode']}")
        cat_lines.append(f"Розподіл значень: {stats['value_counts']}")
        cat_lines.append(
            "Середнє / медіана / дисперсія кодів: "
            f"{stats['mean_code']:.3f} / {stats['median_code']} / {stats['variance_code']:.3f}"
        )
        cat_lines.append(
            "Квантілі кодів 25% / 50% / 75%: "
            f"{stats['q25_code']} / {stats['q50_code']} / {stats['q75_code']}"
        )
    cat_text = "\n".join(cat_lines)

    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            "Числові ознаки",
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            0.02,
            0.94,
            num_text,
            ha="left",
            va="top",
            fontsize=9,
            fontfamily="monospace",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            cat_text,
            ha="left",
            va="top",
            fontsize=9,
            fontfamily="monospace",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
