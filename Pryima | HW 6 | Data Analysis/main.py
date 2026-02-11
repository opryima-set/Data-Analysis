from pathlib import Path
import io
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

try:
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "datasets" / "diagnostic" / "wdbc.data"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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

OUTPUT_BUFFER = io.StringIO()
_ORIGINAL_STDOUT = sys.stdout


class TeeOutput:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, obj):
        for s in self.streams:
            s.write(obj)
        return len(obj)

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = TeeOutput(_ORIGINAL_STDOUT, OUTPUT_BUFFER)


def load_dataset(path: Path = DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path, header=None, names=COLUMN_NAMES)


def _stable_counts(df: pd.DataFrame) -> dict:
    vc = df["diagnosis"].value_counts()
    out = {k: int(vc[k]) for k in ["B", "M"] if k in vc.index}
    for k in vc.index:
        if k not in out:
            out[k] = int(vc[k])
    return out


def plot_area_simple(df: pd.DataFrame) -> None:
    y = df["radius_mean"].sort_values().reset_index(drop=True).values
    x = np.arange(len(y))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x, 0, y, alpha=0.5)
    ax.plot(x, y, linewidth=1)
    ax.set_xlabel("Sorted observation index")
    ax.set_ylabel("radius_mean")
    ax.set_title("Area chart: radius_mean (sorted)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "area_01_simple.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/area_01_simple.png")


def plot_area_multiple(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    cols = ["radius_mean", "texture_mean", "area_mean"]
    x = np.arange(len(df))

    for col in cols:
        y = df[col].sort_values().reset_index(drop=True).values
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
        ax.fill_between(x, 0, y_norm, alpha=0.35, label=col)
        ax.plot(x, y_norm, linewidth=1, alpha=0.9)

    ax.set_xlabel("Sorted observation index (per feature)")
    ax.set_ylabel("Normalized value (0–1)")
    ax.set_title("Multiple area charts: key metrics (sorted & normalized)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "area_02_multiple.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/area_02_multiple.png")


def plot_area_stacked(df: pd.DataFrame) -> None:
    cols = ["radius_mean", "texture_mean", "area_mean"]
    df_s = df.sort_values("radius_mean").reset_index(drop=True)

    data = df_s[cols].values
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-9)
    stack = np.cumsum(data_norm, axis=1)
    x = np.arange(len(df_s))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x, 0, stack[:, 0], alpha=0.7, label=cols[0])
    ax.fill_between(x, stack[:, 0], stack[:, 1], alpha=0.7, label=cols[1])
    ax.fill_between(x, stack[:, 1], stack[:, 2], alpha=0.7, label=cols[2])

    ax.set_xlabel("Observation index (sorted by radius_mean)")
    ax.set_ylabel("Stacked normalized value")
    ax.set_title("Stacked area: cumulative contribution (sorted by radius_mean)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "area_03_stacked.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/area_03_stacked.png")


def plot_area_confidence(df: pd.DataFrame) -> None:
    y = df["radius_mean"].sort_values().reset_index(drop=True).astype(float)
    x = np.arange(len(y))
    window = 50

    roll_mean = y.rolling(window, center=True).mean()
    roll_std = y.rolling(window, center=True).std()
    n_eff = window
    se = roll_std / np.sqrt(n_eff)

    roll_mean = roll_mean.bfill().ffill()
    se = se.bfill().ffill()

    lo = (roll_mean - 1.96 * se).values
    hi = (roll_mean + 1.96 * se).values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x, lo, hi, alpha=0.25, label="95% CI (rolling mean)")
    ax.plot(x, roll_mean.values, linewidth=2, label=f"Rolling mean (window={window})")

    ax.set_xlabel("Sorted observation index")
    ax.set_ylabel("radius_mean")
    ax.set_title("Area with 95% CI for rolling mean (sorted)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "area_04_confidence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/area_04_confidence.png")


COLOR_B = "#3498db"
COLOR_M = "#e74c3c"


def plot_density_histogram(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    r = df["radius_mean"].astype(float)
    bins = np.linspace(r.min(), r.max(), 26)

    for diag, label, color in [("B", "Benign", COLOR_B), ("M", "Malignant", COLOR_M)]:
        subset = df.loc[df["diagnosis"] == diag, "radius_mean"].astype(float)
        ax.hist(subset, bins=bins, alpha=0.6, label=label, color=color, edgecolor="black")

    ax.set_xlabel("radius_mean")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram: radius_mean by diagnosis")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "density_01_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/density_01_histogram.png")


def plot_density_kde(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x_plot = np.linspace(df["radius_mean"].min(), df["radius_mean"].max(), 300)

    for diag, label, color in [("B", "Benign", COLOR_B), ("M", "Malignant", COLOR_M)]:
        data = df.loc[df["diagnosis"] == diag, "radius_mean"].astype(float).values
        kde = stats.gaussian_kde(data)
        y = kde(x_plot)
        ax.plot(x_plot, y, linewidth=2, label=label, color=color)
        ax.fill_between(x_plot, 0, y, alpha=0.25, color=color)

    ax.set_xlabel("radius_mean")
    ax.set_ylabel("Density")
    ax.set_title("KDE: radius_mean by diagnosis")
    ax.legend()
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "density_02_kde.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/density_02_kde.png")


def plot_density_2d(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    r_min, r_max = df["radius_mean"].min(), df["radius_mean"].max()
    a_min, a_max = df["area_mean"].min(), df["area_mean"].max()
    rr = np.linspace(r_min, r_max, 90)
    aa = np.linspace(a_min, a_max, 90)
    R, A = np.meshgrid(rr, aa)
    pos = np.vstack([R.ravel(), A.ravel()])

    legend_handles = []
    for diag, label, color in [("B", "Benign", COLOR_B), ("M", "Malignant", COLOR_M)]:
        sub = df[df["diagnosis"] == diag][["radius_mean", "area_mean"]].astype(float).values.T
        if sub.shape[1] < 10:
            continue
        kde = stats.gaussian_kde(sub)
        Z = kde(pos).reshape(R.shape)
        ax.contour(R, A, Z, levels=6, linewidths=1.5, alpha=0.9, colors=[color])
        legend_handles.append(Line2D([0], [0], color=color, lw=2, label=label))

    ax.scatter(df["radius_mean"], df["area_mean"], s=8, alpha=0.15, color="gray")
    ax.set_xlabel("radius_mean")
    ax.set_ylabel("area_mean")
    ax.set_title("2D density (KDE contours): radius_mean vs area_mean")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "density_03_2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/density_03_2d.png")


def plot_density_ridge(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x_plot = np.linspace(df["radius_mean"].min(), df["radius_mean"].max(), 300)

    offsets = {"B": 0.0, "M": 0.55}
    for diag, label, color in [("B", "Benign", COLOR_B), ("M", "Malignant", COLOR_M)]:
        data = df.loc[df["diagnosis"] == diag, "radius_mean"].astype(float).values
        kde = stats.gaussian_kde(data)
        y = kde(x_plot)
        y = y / (y.max() + 1e-9) * 0.4
        off = offsets[diag]
        ax.fill_between(x_plot, off, off + y, alpha=0.65, label=label, color=color)
        ax.plot(x_plot, off + y, linewidth=1.5, color=color)

    ax.set_xlabel("radius_mean")
    ax.set_ylabel("Density (ridge)")
    ax.set_title("Ridge plot: radius_mean by diagnosis")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "density_04_ridge.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/density_04_ridge.png")


def plot_ts_simple(df: pd.DataFrame) -> None:
    y = df["radius_mean"].sort_values().reset_index(drop=True).astype(float).values
    x = np.arange(len(y))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, linewidth=1, label="radius_mean (sorted)")
    ax.set_xlabel("Sorted observation index")
    ax.set_ylabel("radius_mean")
    ax.set_title("Ordered profile: radius_mean (sorted)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "ts_01_simple.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/ts_01_simple.png")


def plot_ts_multiple(df: pd.DataFrame) -> None:
    df_s = df.sort_values("radius_mean").reset_index(drop=True)
    x = np.arange(len(df_s))

    cols = ["radius_mean", "texture_mean", "area_mean"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for col in cols:
        y = df_s[col].astype(float).values
        y_n = (y - y.min()) / (y.max() - y.min() + 1e-9)
        ax.plot(x, y_n, linewidth=1, label=col, alpha=0.9)

    ax.set_xlabel("Observation index (sorted by radius_mean)")
    ax.set_ylabel("Normalized value")
    ax.set_title("Multiple ordered profiles (sorted by radius_mean)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "ts_02_multiple.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/ts_02_multiple.png")


def plot_ts_trend_smoothing(df: pd.DataFrame) -> None:
    y = df["radius_mean"].sort_values().reset_index(drop=True).astype(float)
    x = np.arange(len(y))
    window = 40
    smooth = y.rolling(window, center=True).mean().bfill().ffill()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y.values, linewidth=0.8, alpha=0.35, label="Raw (sorted)")
    ax.plot(x, smooth.values, linewidth=2, label=f"Rolling mean (window={window})")
    ax.set_xlabel("Sorted observation index")
    ax.set_ylabel("radius_mean")
    ax.set_title("Ordered profile with smoothing (rolling mean)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "ts_03_trend_smoothing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/ts_03_trend_smoothing.png")


def plot_hierarchical_tree(df: pd.DataFrame) -> None:
    feats = ["radius_mean", "texture_mean", "area_mean", "smoothness_mean"]
    X = df[feats].astype(float).values
    n_sample = min(80, len(df))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), n_sample, replace=False)
    X_sub = X[idx]

    dist = pdist(X_sub, metric="euclidean")
    link = hierarchy.linkage(dist, method="ward")

    fig, ax = plt.subplots(figsize=(10, 5))
    hierarchy.dendrogram(link, ax=ax, leaf_rotation=90, color_threshold=0.7 * max(link[:, 2]))
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Distance (Ward)")
    ax.set_title("Hierarchical tree (dendrogram): clustering by mean features")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "tree_01_hierarchical.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/tree_01_hierarchical.png")


def plot_radial_tree() -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.axis("off")

    nodes = [
        (0.5, 0.5, "Data", 12),
        (0.35, 0.72, "Benign (B)", 10),
        (0.65, 0.72, "Malignant (M)", 10),
        (0.25, 0.35, "radius_mean", 10),
        (0.75, 0.35, "area_mean", 10),
    ]

    for px, py, label, ms in nodes:
        ax.plot(px, py, "o", markersize=ms, markeredgecolor="black")
        ax.annotate(label, (px, py), fontsize=9, ha="center", va="center")

    for (x1, y1), (x2, y2) in [((0.5, 0.5), (0.35, 0.72)),
                              ((0.5, 0.5), (0.65, 0.72)),
                              ((0.35, 0.72), (0.25, 0.35)),
                              ((0.65, 0.72), (0.75, 0.35))]:
        ax.plot([x1, x2], [y1, y2], "k-", lw=1.2, alpha=0.8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Conceptual radial tree: Data → Diagnosis → key features")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "tree_02_radial.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/tree_02_radial.png")


def plot_decision_tree(df: pd.DataFrame) -> None:
    if not HAS_SKLEARN:
        print("Skipped: plots/tree_03_decision.png (sklearn not installed)")
        return

    feats = ["radius_mean", "texture_mean", "area_mean", "smoothness_mean", "compactness_mean"]
    X = df[feats].astype(float)
    y = (df["diagnosis"] == "M").astype(int)

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(
        clf,
        feature_names=feats,
        class_names=["Benign", "Malignant"],
        filled=True,
        ax=ax,
        fontsize=8,
    )
    ax.set_title("Decision tree: diagnosis from mean features (max_depth=3)")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "tree_03_decision.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/tree_03_decision.png")


def plot_network_graph() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    nodes = {
        "Diagnosis": (0.5, 0.85),
        "Benign": (0.25, 0.55),
        "Malignant": (0.75, 0.55),
        "radius_mean": (0.15, 0.2),
        "area_mean": (0.5, 0.15),
        "texture_mean": (0.85, 0.2),
    }
    edges = [
        ("Diagnosis", "Benign"), ("Diagnosis", "Malignant"),
        ("Benign", "radius_mean"), ("Benign", "area_mean"), ("Benign", "texture_mean"),
        ("Malignant", "radius_mean"), ("Malignant", "area_mean"), ("Malignant", "texture_mean"),
    ]

    for a, b in edges:
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-", lw=1.4, alpha=0.7))

    for name, (x, y) in nodes.items():
        ax.plot(x, y, "o", markersize=14, markeredgecolor="black")
        ax.text(x, y - 0.06, name, ha="center", va="top", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Conceptual network graph: Diagnosis and key features")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "tree_04_network.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/tree_04_network.png")


def print_conclusions(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("ANALYTICAL CONCLUSIONS (HW6)")
    print("=" * 60)

    b_mean = df.loc[df["diagnosis"] == "B", "radius_mean"].astype(float).mean()
    m_mean = df.loc[df["diagnosis"] == "M", "radius_mean"].astype(float).mean()
    b_med = df.loc[df["diagnosis"] == "B", "radius_mean"].astype(float).median()
    m_med = df.loc[df["diagnosis"] == "M", "radius_mean"].astype(float).median()

    print("\n--- Area / Ordered profiles ---")
    print("Using sorted ordering avoids implying a real time axis. The profile shows spread and tail behavior.")
    print("Rolling mean + CI highlights stable central regions and variability at the extremes.")

    print("\n--- Density charts ---")
    print(f"Malignant tends to have larger radius_mean: mean(M)={m_mean:.2f} vs mean(B)={b_mean:.2f}; "
          f"median(M)={m_med:.2f} vs median(B)={b_med:.2f}.")
    print("2D density in (radius_mean, area_mean) indicates partial separation of groups (clusters).")
    print("Ridge plot reinforces the distribution shift toward larger values for Malignant.")

    print("\n--- Tree / Network ---")
    print("Dendrogram clusters observations by similarity in selected mean features.")
    if HAS_SKLEARN:
        print("Decision tree demonstrates that a small set of mean features can classify B vs M reasonably well.")
    else:
        print("Decision tree was skipped (sklearn not installed).")
    print("Radial/network graphs are conceptual diagrams to summarize relationships (not data-driven layouts).")

    print("=" * 60)


def main() -> None:
    df = load_dataset()

    print("HW6: Advanced visualizations")
    print(f"Dataset shape: {df.shape}")
    print(f"Diagnosis: {_stable_counts(df)}")

    print("\n--- Area charts ---")
    plot_area_simple(df)
    plot_area_multiple(df)
    plot_area_stacked(df)
    plot_area_confidence(df)

    print("\n--- Density charts ---")
    plot_density_histogram(df)
    plot_density_kde(df)
    plot_density_2d(df)
    plot_density_ridge(df)

    print("\n--- Ordered profiles (replacing pseudo-time series) ---")
    plot_ts_simple(df)
    plot_ts_multiple(df)
    plot_ts_trend_smoothing(df)

    print("\n--- Tree / Network ---")
    plot_hierarchical_tree(df)
    plot_radial_tree()
    plot_decision_tree(df)
    plot_network_graph()

    print_conclusions(df)
    print("\nAll plots saved to plots/")


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout = _ORIGINAL_STDOUT
        out_path = BASE_DIR / "results.txt"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(OUTPUT_BUFFER.getvalue())
        print(f"Console output saved to {out_path}")
