from pathlib import Path
import io
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "datasets" / "diagnostic" / "wdbc.data"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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
        n = 0
        for s in self.streams:
            n = s.write(obj)
            s.flush()
        return n

    def flush(self):
        for s in self.streams:
            s.flush()


# write to console + to buffer
sys.stdout = TeeOutput(_ORIGINAL_STDOUT, OUTPUT_BUFFER)


def load_dataset(path: Path = DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path, header=None, names=COLUMN_NAMES)


def _counts_mb(df: pd.DataFrame) -> pd.Series:
    counts = df["diagnosis"].value_counts()
    # stable order if present
    order = [c for c in ["M", "B"] if c in counts.index]
    if order:
        counts = counts.reindex(order)
    return counts


def plot_pie_chart(df: pd.DataFrame) -> None:
    counts = _counts_mb(df)
    labels = [f"{k} ({v})" for k, v in counts.items()]
    colors = ["#e74c3c", "#3498db"]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors[: len(counts)],
        startangle=90,
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")

    ax.set_title("Diagnosis distribution (M vs B)")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "01_pie_diagnosis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/01_pie_diagnosis.png")


def plot_donut_chart(df: pd.DataFrame) -> None:
    counts = _counts_mb(df)
    labels = list(counts.index)
    colors = ["#e74c3c", "#3498db"]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors[: len(counts)],
        startangle=90,
        pctdistance=0.75,
    )
    centre_circle = plt.Circle((0, 0), 0.5, fc="white")
    ax.add_artist(centre_circle)

    for t in autotexts:
        t.set_fontweight("bold")

    ax.set_title("Diagnosis distribution (Donut)")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "02_donut_diagnosis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/02_donut_diagnosis.png")


def plot_bar_chart(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(
        index="diagnosis",
        values=["radius_mean", "texture_mean", "area_mean"],
        aggfunc="mean",
    )
    pivot = pivot.reindex([c for c in ["M", "B"] if c in pivot.index])

    x = np.arange(len(pivot.columns))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (idx, row) in enumerate(pivot.iterrows()):
        offset = (i - (len(pivot.index) - 1) / 2) * width
        ax.bar(x + offset, row.values, width, label=idx)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_ylabel("Mean value")
    ax.set_title("Mean indicators: Malignant vs Benign")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "03_bar_means_by_diagnosis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/03_bar_means_by_diagnosis.png")


def plot_line_chart(df: pd.DataFrame) -> None:
    # More meaningful than sorting by id: sort by the feature value
    s = df["radius_mean"].sort_values().reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s.index, s.values, label="radius_mean (sorted)", alpha=0.7)

    window = 30
    rolling_mean = s.rolling(window).mean()
    ax.plot(
        rolling_mean.index,
        rolling_mean.values,
        label=f"rolling mean ({window})",
        linewidth=2,
    )

    ax.set_xlabel("Sorted observation index")
    ax.set_ylabel("radius_mean")
    ax.set_title("radius_mean profile (sorted) + rolling mean")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "04_line_trends.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/04_line_trends.png")


def plot_histogram(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    r_min = df["radius_mean"].min()
    r_max = df["radius_mean"].max()
    bins = np.linspace(r_min, r_max, 26)

    for diag, color in [("M", "#e74c3c"), ("B", "#3498db")]:
        subset = df.loc[df["diagnosis"] == diag, "radius_mean"]
        ax.hist(subset, bins=bins, alpha=0.6, label=diag, color=color, edgecolor="black")

    ax.set_xlabel("radius_mean")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of radius_mean by diagnosis")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "05_histogram_radius_by_diagnosis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/05_histogram_radius_by_diagnosis.png")


def plot_box_plot(df: pd.DataFrame) -> None:
    data_b = df.loc[df["diagnosis"] == "B", "area_mean"].values
    data_m = df.loc[df["diagnosis"] == "M", "area_mean"].values

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        [data_b, data_m],
        labels=["Benign (B)", "Malignant (M)"],
        patch_artist=True,
        showmeans=True,
        meanline=False,
    )

    bp["boxes"][0].set_facecolor("#3498db")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#e74c3c")
    bp["boxes"][1].set_alpha(0.7)

    ax.set_ylabel("area_mean")
    ax.set_title("Box plot: area_mean by diagnosis")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "06_box_area_by_diagnosis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/06_box_area_by_diagnosis.png")


def plot_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for diag, color, label in [("M", "#e74c3c", "Malignant"), ("B", "#3498db", "Benign")]:
        subset = df[df["diagnosis"] == diag]
        ax.scatter(
            subset["radius_mean"],
            subset["area_mean"],
            alpha=0.6,
            c=color,
            label=label,
        )

    ax.set_xlabel("radius_mean")
    ax.set_ylabel("area_mean")
    ax.set_title("radius_mean vs area_mean (colored by diagnosis)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "07_scatter_radius_vs_area.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/07_scatter_radius_vs_area.png")


def main() -> None:
    df = load_dataset()
    counts = _counts_mb(df)
    
    total = len(df)
    b_perc = (counts.get('B', 0) / total) * 100
    m_perc = (counts.get('M', 0) / total) * 100

    print("HW5: Data visualization and analysis")
    print(f"Dataset shape: {df.shape}")
    print("Diagnosis value_counts:", counts.to_dict())
    print("radius_mean range: [{:.2f}, {:.2f}]".format(df["radius_mean"].min(), df["radius_mean"].max()))
    print("-" * 30)

    plot_pie_chart(df)
    plot_donut_chart(df)
    plot_bar_chart(df)
    plot_line_chart(df)
    plot_histogram(df)
    plot_box_plot(df)
    plot_scatter(df)

    print("\n--- ВИСНОВКИ ПО ДАНИМ ---")
    print(f"1. ДІАГНОЗ: У наборі даних переважають доброякісні пухлини (B) — {b_perc:.1f}% "
          f"над злоякісними (M) — {m_perc:.1f}%.")
    
    print("2. ПОКАЗНИКИ: Злоякісні пухлини (M) мають значно вищі середні показники основних ознак. "
          "Наприклад, середня площа (area_mean) для M суттєво перевищує показники для B.")
    
    print("3. КОРЕЛЯЦІЯ: Scatter plot чітко демонструє позитивну залежність між radius_mean та area_mean. "
          "Злоякісні випадки формують окремий кластер у зоні високих значень.")
    
    print("4. ВАРІАТИВНІСТЬ: Гістограма показує, що розподіл радіуса злоякісних пухлин зміщений вправо "
          "та має більший розкид (дисперсію) порівняно з доброякісними.")
    
    print("-" * 30)
    print("All plots saved to plots/")


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout = _ORIGINAL_STDOUT
        results_path = BASE_DIR / "results.txt"
        with results_path.open("w", encoding="utf-8") as f:
            f.write(OUTPUT_BUFFER.getvalue())
        print(f"Console output saved to {results_path}")
