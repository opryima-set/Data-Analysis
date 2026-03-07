from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "datasets" / "diagnostic" / "wdbc.data"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 11

COLUMN_NAMES = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
    "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst",
    "symmetry_worst", "fractal_dimension_worst",
]


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    return pd.read_csv(DATA_FILE, header=None, names=COLUMN_NAMES)


def save_plot(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout(pad=1.2)
    fig.savefig(PLOTS_DIR / filename, dpi=140, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)


def diagnosis_means(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("diagnosis")[col].mean().reindex(["B", "M"])


# --- 1. Truncated axis ---
def example_01_truncated_axis(df: pd.DataFrame) -> None:
    by_diag = diagnosis_means(df, "radius_mean")
    x = by_diag.index.tolist()
    y = by_diag.values

    # Manipulative: y-axis truncated to exaggerate difference
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x, y, color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("Mean radius")
    ax.set_xlabel("Diagnosis")
    ax.set_ylim(11.8, 18.5)
    ax.set_title("Mean radius by diagnosis\n(manipulative: truncated axis)", fontsize=10)
    save_plot(fig, "manip_01_truncated_axis.png")

    # Honest: full scale from zero
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x, y, color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("Mean radius")
    ax.set_xlabel("Diagnosis")
    ax.set_ylim(0, max(y) * 1.1)
    ax.set_title("Mean radius by diagnosis\n(honest: full axis)", fontsize=10)
    save_plot(fig, "honest_01_truncated_axis.png")


# --- 2. Cherry-picking + ordering ---
def example_02_cherry_picking(df: pd.DataFrame) -> None:
    # Manipulative:
    # беремо тільки частину спостережень з великим radius_mean
    # і додатково сортуємо їх, щоб зобразити штучний "тренд"
    cherry = df.nlargest(30, "radius_mean").sort_values("radius_mean")
    x_manip = np.arange(len(cherry))
    y_manip = cherry["radius_mean"].values

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_manip, y_manip, "o-", color="#3498db", linewidth=1.8)
    ax.set_xlabel("Sample index (selected subset)")
    ax.set_ylabel("Radius (mean)")
    ax.set_title("Radius across samples\n(manipulative: cherry-picking + ordering)", fontsize=10)
    save_plot(fig, "manip_02_cherry_picking.png")

    # Honest: all data in original order
    x_honest = np.arange(len(df))
    y_honest = df["radius_mean"].values

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_honest, y_honest, ".", color="#3498db", alpha=0.6, markersize=3)
    ax.set_xlabel("Sample index (all data)")
    ax.set_ylabel("Radius (mean)")
    ax.set_title("Radius across samples\n(honest: full data, original order)", fontsize=10)
    save_plot(fig, "honest_02_cherry_picking.png")


# --- 3. Log scale disguised as linear ---
def example_03_log_disguised(df: pd.DataFrame) -> None:
    by_diag = diagnosis_means(df, "area_mean")
    x = by_diag.index.tolist()
    y = by_diag.values

    # Manipulative: log scale, but not clearly disclosed
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x, y, color=["#9b59b6", "#e67e22"])
    ax.set_yscale("log")
    ax.set_ylabel("Mean area")
    ax.set_xlabel("Diagnosis")
    ax.set_title("Mean area by diagnosis\n(manipulative: log scale not disclosed)", fontsize=10)
    save_plot(fig, "manip_03_log_disguised.png")

    # Honest: linear scale — візуально відрізняється від маніпулятивного, показує справжні пропорції
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x, y, color=["#9b59b6", "#e67e22"])
    ax.set_ylabel("Mean area")
    ax.set_xlabel("Diagnosis")
    ax.set_title("Mean area by diagnosis\n(honest: linear scale)", fontsize=10)
    save_plot(fig, "honest_03_log_disguised.png")


# --- 4. Dual-axis correlation illusion ---
def example_04_dual_axis_illusion(df: pd.DataFrame) -> None:
    # Більше точок: бінуємо дані за radius_mean і рахуємо середні texture_mean та area_mean
    temp = df[["radius_mean", "texture_mean", "area_mean"]].copy()
    temp = temp.sort_values("radius_mean").reset_index(drop=True)

    n_bins = 20
    temp["bin"] = pd.qcut(temp.index, q=n_bins, labels=False)

    agg = temp.groupby("bin").agg(
        radius_mean=("radius_mean", "mean"),
        texture_mean=("texture_mean", "mean"),
        area_mean=("area_mean", "mean"),
    )

    x = np.arange(len(agg))
    left_y = agg["texture_mean"].values
    right_y = agg["area_mean"].values

    # Manipulative: dual axis with tuned limits to make two series look synchronized
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(x, left_y, "o-", linewidth=2, label="Texture mean", color="#2980b9")
    ax1.set_xlabel("Ordered bins by radius_mean")
    ax1.set_ylabel("Texture mean", color="#2980b9")
    ax1.tick_params(axis="y", labelcolor="#2980b9")

    ax2 = ax1.twinx()
    ax2.plot(x, right_y, "s-", linewidth=2, label="Area mean", color="#c0392b")
    ax2.set_ylabel("Area mean", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    # Підігнані межі для схожої форми
    lmin, lmax = left_y.min(), left_y.max()
    rmin, rmax = right_y.min(), right_y.max()
    left_pad = (lmax - lmin) * 0.10
    right_pad = (rmax - rmin) * 0.10
    ax1.set_ylim(lmin - left_pad, lmax + left_pad)
    ax2.set_ylim(rmin - right_pad, rmax + right_pad)

    ax1.set_title("Two metrics across ordered bins\n(manipulative: dual-axis illusion)", fontsize=10)
    save_plot(fig, "manip_04_dual_axis_illusion.png")

    # Honest: separate panels with independent scales
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax_top.plot(x, left_y, "o-", linewidth=2, color="#2980b9")
    ax_top.set_ylabel("Texture mean")
    ax_top.set_title("Texture mean across ordered bins", fontsize=10)
    ax_top.grid(alpha=0.25)

    ax_bottom.plot(x, right_y, "s-", linewidth=2, color="#c0392b")
    ax_bottom.set_ylabel("Area mean")
    ax_bottom.set_xlabel("Ordered bins by radius_mean")
    ax_bottom.set_title("Area mean across ordered bins", fontsize=10)
    ax_bottom.grid(alpha=0.25)

    fig.suptitle("Honest comparison: separate panels avoid false visual correlation", fontsize=11)
    save_plot(fig, "honest_04_dual_axis_illusion.png")


# --- 5. Aspect ratio manipulation ---
def example_05_aspect_ratio(df: pd.DataFrame) -> None:
    x = df["radius_mean"].values
    y = df["perimeter_mean"].values

    # Manipulative: very wide chart makes slope/correlation feel flatter or more stretched visually
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(x, y, alpha=0.65, s=18, color="#16a085", edgecolors="none")
    ax.set_xlabel("Radius mean")
    ax.set_ylabel("Perimeter mean")
    ax.set_title("Radius vs perimeter\n(manipulative: extreme aspect ratio)", fontsize=10)
    save_plot(fig, "manip_05_aspect_ratio.png")

    # Honest: balanced aspect ratio
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, alpha=0.65, s=18, color="#16a085", edgecolors="none")
    ax.set_xlabel("Radius mean")
    ax.set_ylabel("Perimeter mean")
    ax.set_title("Radius vs perimeter\n(honest: balanced aspect ratio)", fontsize=10)
    save_plot(fig, "honest_05_aspect_ratio.png")


# --- 6. Misleading aggregation ---
def example_06_misleading_aggregation(df: pd.DataFrame) -> None:
    by_diag = diagnosis_means(df, "radius_mean")
    n_b = (df["diagnosis"] == "B").sum()
    n_m = (df["diagnosis"] == "M").sum()
    true_overall = df["radius_mean"].mean()
    wrong_overall = (by_diag["B"] + by_diag["M"]) / 2

    cats = ["Benign", "Malignant", "Overall"]

    # Manipulative: incorrect overall average from group means
    fig, ax = plt.subplots(figsize=(6, 4))
    vals_manip = [by_diag["B"], by_diag["M"], wrong_overall]
    bars = ax.bar(cats, vals_manip, color=["#2ecc71", "#e74c3c", "#95a5a6"])
    ax.set_ylabel("Mean radius")
    ax.set_title("Overall = average of group means\n(manipulative: wrong aggregation)", fontsize=10)
    for bar, val in zip(bars, vals_manip):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15, f"{val:.2f}", ha="center", fontsize=9)
    save_plot(fig, "manip_06_misleading_aggregation.png")

    # Honest: overall = true grand mean, plus sample sizes
    fig, ax = plt.subplots(figsize=(6, 4))
    vals_honest = [by_diag["B"], by_diag["M"], true_overall]
    bars = ax.bar(cats, vals_honest, color=["#2ecc71", "#e74c3c", "#95a5a6"])
    ax.set_ylabel("Mean radius")
    ax.set_title("Overall = grand mean\n(honest: correct aggregation)", fontsize=10)
    ax.annotate(f"n(B)={n_b}, n(M)={n_m}", xy=(0.02, 0.98), xycoords="axes fraction", fontsize=9, va="top")
    for bar, val in zip(bars, vals_honest):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15, f"{val:.2f}", ha="center", fontsize=9)
    save_plot(fig, "honest_06_misleading_aggregation.png")


# --- 7. Absolute numbers without normalization ---
def example_07_absolute_without_normalization(df: pd.DataFrame) -> None:
    counts = df["diagnosis"].value_counts().reindex(["B", "M"])
    b_count = int(counts["B"])
    m_count = int(counts["M"])

    # Manipulative: counts only
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Benign", "Malignant"], [b_count, m_count], color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("Number of cases")
    ax.set_title("Cases by diagnosis\n(manipulative: absolute counts only)", fontsize=10)
    save_plot(fig, "manip_07_absolute_no_normalization.png")

    # Honest: normalized shares with counts shown as annotation
    fig, ax = plt.subplots(figsize=(5, 4))
    total = b_count + m_count
    props = [b_count / total * 100, m_count / total * 100]
    bars = ax.bar(["Benign", "Malignant"], props, color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("Share of dataset (%)")
    ax.set_title("Cases by diagnosis\n(honest: normalized shares + counts)", fontsize=10)
    for bar, c, p in zip(bars, [b_count, m_count], props):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 1, f"{p:.1f}%\n(n={c})", ha="center", fontsize=9)
    save_plot(fig, "honest_07_absolute_no_normalization.png")


# --- 8. Hiding variance ---
def example_08_hiding_variance(df: pd.DataFrame) -> None:
    by_diag = df.groupby("diagnosis")["radius_mean"].agg(["mean", "std"]).reindex(["B", "M"])
    x = by_diag.index.tolist()
    mean_vals = by_diag["mean"].values

    # Manipulative: only means, hides distribution overlap and spread
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x, mean_vals, color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("Mean radius")
    ax.set_xlabel("Diagnosis")
    ax.set_title("Mean radius by diagnosis\n(manipulative: variance hidden)", fontsize=10)
    save_plot(fig, "manip_08_hiding_variance.png")

    # Honest: boxplot reveals spread and overlap
    benign = df.loc[df["diagnosis"] == "B", "radius_mean"]
    malignant = df.loc[df["diagnosis"] == "M", "radius_mean"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([benign, malignant], tick_labels=["B", "M"], patch_artist=True,
               boxprops=dict(facecolor="#d5f5e3"),
               medianprops=dict(color="black"),
               whiskerprops=dict(color="#555555"),
               capprops=dict(color="#555555"))
    ax.set_ylabel("Radius mean")
    ax.set_xlabel("Diagnosis")
    ax.set_title("Radius by diagnosis\n(honest: distribution and variance shown)", fontsize=10)
    save_plot(fig, "honest_08_hiding_variance.png")


def main() -> None:
    df = load_data()

    example_01_truncated_axis(df)
    example_02_cherry_picking(df)
    example_03_log_disguised(df)
    example_04_dual_axis_illusion(df)
    example_05_aspect_ratio(df)
    example_06_misleading_aggregation(df)
    example_07_absolute_without_normalization(df)
    example_08_hiding_variance(df)

    print(f"All examples saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()