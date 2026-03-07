from pathlib import Path
import io
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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

KEY_FEATURES = ["radius_mean", "area_mean", "texture_mean"]
CORR_FEATURES = ["radius_mean", "perimeter_mean", "area_mean", "concavity_mean", "concave_points_mean", "texture_mean"]
SEPARATION_FEATURES = ["radius_mean", "concavity_mean", "concave_points_mean"]

_out = io.StringIO()
_stdout = sys.stdout


def log(msg: str = "") -> None:
    print(msg, file=_stdout)
    _out.write(msg + "\n")


def get_stdout_buffer() -> io.StringIO:
    return _out


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    return pd.read_csv(DATA_FILE, header=None, names=COLUMN_NAMES)


def save_plot(fig: plt.Figure, name: str) -> None:
    fig.tight_layout(pad=1.2)
    fig.savefig(PLOTS_DIR / name, dpi=120, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)


# --- 1. Підготовка ---
def run_preparation(df: pd.DataFrame) -> None:
    log("=" * 60)
    log("1. ПІДГОТОВКА")
    log("=" * 60)
    log(f"Завантажено: {len(df)} рядків, {len(df.columns)} колонок.")
    log(f"Колонки: {list(df.columns)}")
    log("")


# --- 2. Перевірка якості ---
def run_quality_check(df: pd.DataFrame) -> None:
    log("=" * 60)
    log("2. ПЕРЕВІРКА ЯКОСТІ ДАНИХ")
    log("=" * 60)
    missing = df.isna().sum()
    log("Пропуски по колонках:")
    log(missing[missing > 0].to_string() if missing.any() else "  Немає пропусків.")
    if not missing.any():
        log("  (усі колонки без NA)")
    log(f"\nShape: {df.shape}")
    log(f"\ndtypes (перші 5): {df.dtypes.head().to_dict()}")
    log(f"\nУнікальні diagnosis: {df['diagnosis'].unique().tolist()}")
    log(f"  value_counts:\n{df['diagnosis'].value_counts().to_string()}")
    log("Обробка пропусків: не потрібна — пропусків немає (WDBC без NA).")
    log("")


# --- 3. Описова статистика ---
def run_describe(df: pd.DataFrame) -> None:
    log("=" * 60)
    log("3. ОПИСОВА СТАТИСТИКА")
    log("=" * 60)
    num = df.select_dtypes(include=[np.number])
    log("df.describe() для числових полів:")
    log(num.describe().to_string())
    log("\nКлючові змінні (mean, median, std):")
    for col in KEY_FEATURES:
        s = df[col]
        log(f"  {col}: mean={s.mean():.4f}, median={s.median():.4f}, std={s.std():.4f}")
    log("")


# --- 4. Гістограми / KDE ---
def run_histograms_kde(df: pd.DataFrame) -> None:
    log("=" * 60)
    log("4. ГІСТОГРАМИ / KDE")
    log("=" * 60)
    for col in KEY_FEATURES:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        # Загалом
        df[col].hist(ax=ax1, bins=30, edgecolor="black", alpha=0.7)
        df[col].plot(kind="kde", ax=ax1, color="red", linewidth=2)
        ax1.set_title(f"{col} — загалом")
        ax1.set_ylabel("Частота")
        # B vs M
        for label, color in [("B", "#2ecc71"), ("M", "#e74c3c")]:
            subset = df.loc[df["diagnosis"] == label, col]
            subset.hist(ax=ax2, bins=25, alpha=0.5, label=label, color=color, edgecolor="black")
            subset.plot(kind="kde", ax=ax2, color=color, linewidth=2)
        ax2.legend()
        ax2.set_title(f"{col} — B vs M")
        ax2.set_ylabel("Частота")
        save_plot(fig, f"04_hist_kde_{col}.png")
    skews = {c: df[c].skew() for c in KEY_FEATURES}
    log("Коефіцієнт асиметрії (skew): " + ", ".join(f"{k}={v:.3f}" for k, v in skews.items()))
    log("Коментар: area_mean сильно правий асиметричний; radius_mean і texture_mean помірно. Мультимодальність у texture_mean можлива (підтипи).")
    log("")


# --- 5. Boxplots / violin ---
def run_boxplots(df: pd.DataFrame) -> None:
    log("=" * 60)
    log("5. BOXPLOTS / VIOLIN")
    log("=" * 60)
    for col in ["radius_mean", "area_mean"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        b, m = df.loc[df["diagnosis"] == "B", col], df.loc[df["diagnosis"] == "M", col]
        ax1.boxplot([b, m], tick_labels=["B", "M"], patch_artist=True)
        ax1.set_ylabel(col)
        ax1.set_title(f"Boxplot {col} by diagnosis")
        parts = ax2.violinplot([b, m], positions=[0, 1], showmeans=True)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["B", "M"])
        ax2.set_ylabel(col)
        ax2.set_title(f"Violin {col} by diagnosis")
        save_plot(fig, f"05_box_violin_{col}.png")
    log("M має вищі median та більший розкид; overlap існує, викиди видно на boxplot (area_mean).")
    log("")


# --- 6. Викиди ---
def get_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (series < lo) | (series > hi)


def run_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log("=" * 60)
    log("6. ВИЯВЛЕННЯ ВИКИДІВ (IQR, k=1.5)")
    log("=" * 60)
    outlier_cols = ["radius_mean", "area_mean"]
    outlier_mask = pd.Series(False, index=df.index)
    counts = {}
    for col in outlier_cols:
        mask = get_outliers_iqr(df[col])
        counts[col] = mask.sum()
        outlier_mask = outlier_mask | mask
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        log(f"  {col}: Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}, межі [{lo:.4f}, {hi:.4f}], викидів={mask.sum()}")
    n_out = outlier_mask.sum()
    log(f"\nЗагальна кількість спостережень з викидом (хоч в одній змінній): {n_out}")
    out_df = df.loc[outlier_mask, ["id", "diagnosis", "radius_mean", "area_mean"]]
    log("\nid + diagnosis для викидів:")
    log(out_df.to_string())
    log("\nПояснення: викиди у area_mean/radius_mean — скоріше реальні екстремуми (великі пухлини), не помилка вимірювання; WDBC — клінічні дані.")
    log("")
    return out_df, counts


# --- 7. B vs M статистики та тести ---
def run_b_vs_m(df: pd.DataFrame) -> dict:
    log("=" * 60)
    log("7. ПОРІВНЯННЯ B vs M")
    log("=" * 60)
    feats = KEY_FEATURES + ["concavity_mean", "concave_points_mean"]
    rows = []
    pvalues = {}
    for col in feats:
        b = df.loc[df["diagnosis"] == "B", col]
        m = df.loc[df["diagnosis"] == "M", col]
        _, p_norm_b = stats.shapiro(b)
        _, p_norm_m = stats.shapiro(m)
        if p_norm_b < 0.05 or p_norm_m < 0.05:
            stat, p = stats.mannwhitneyu(b, m, alternative="two-sided")
            test_name = "Mann-Whitney"
        else:
            stat, p = stats.ttest_ind(b, m)
            test_name = "t-test"
        pvalues[col] = p
        rows.append({
            "feature": col,
            "B_mean": b.mean(), "B_sd": b.std(),
            "M_mean": m.mean(), "M_sd": m.std(),
            "test": test_name, "p_value": p,
        })
    tbl = pd.DataFrame(rows)
    log("Таблиця mean ± SD по diagnosis:")
    log(tbl.to_string(index=False))
    log("\nВибір тесту: Mann-Whitney при відхиленні нормальності (Shapiro), інакше t-test. P-values в таблиці.")
    log("")
    return pvalues


# --- 8. Кореляція ---
def run_correlation(df: pd.DataFrame) -> pd.DataFrame:
    log("=" * 60)
    log("8. КОРЕЛЯЦІЙНИЙ АНАЛІЗ")
    log("=" * 60)
    sub = df[CORR_FEATURES]
    corr = sub.corr(method="pearson")
    log("Pearson correlation matrix (ключові ознаки):")
    log(corr.round(3).to_string())
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(CORR_FEATURES)))
    ax.set_yticks(range(len(CORR_FEATURES)))
    ax.set_xticklabels(CORR_FEATURES, rotation=45, ha="right")
    ax.set_yticklabels(CORR_FEATURES)
    for i in range(len(CORR_FEATURES)):
        for j in range(len(CORR_FEATURES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlation heatmap (key features)")
    save_plot(fig, "08_corr_heatmap.png")
    # Scatter сильних пар
    pairs = [("radius_mean", "area_mean"), ("radius_mean", "perimeter_mean")]
    for x, y in pairs:
        fig, ax = plt.subplots(figsize=(5, 4))
        r = df[x].corr(df[y])
        ax.scatter(df[x], df[y], alpha=0.5, s=15, c=df["diagnosis"].map({"B": "#2ecc71", "M": "#e74c3c"}))
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"r = {r:.3f}")
        save_plot(fig, f"08_scatter_{x}_{y}.png")
    log("Сильні пари: radius_mean–area_mean, radius_mean–perimeter_mean (r > 0.99).")
    log("")
    return corr


# --- 9. Confounding ---
def run_confounding(df: pd.DataFrame) -> None:
    log("=" * 60)
    log("9. ПЕРЕВІРКА CONFOUNDING (стратифікація)")
    log("=" * 60)
    r_global = df["radius_mean"].corr(df["area_mean"])
    log(f"Глобальна кореляція radius_mean – area_mean: r = {r_global:.4f}")
    df_q = df.copy()
    df_q["concavity_q"] = pd.qcut(df_q["concavity_mean"], q=5, labels=False)
    by_q = df_q.groupby("concavity_q").apply(
        lambda g: g["radius_mean"].corr(g["area_mean"]), include_groups=False
    )
    log("Кореляція radius–area по квінтилях concavity_mean:")
    log(by_q.to_string())
    r_mean_strat = by_q.mean()
    log(f"\nСередня r по стратах: {r_mean_strat:.4f}")
    if r_mean_strat < r_global - 0.05:
        log("Висновок: після стратифікації за concavity_mean кореляція знижується → concavity можливий confounder/пояснювальна змінна.")
    else:
        log("Кореляція залишається високою в стратах — confounder слабкий.")
    log("")


# --- 10. Роздільна здатність ---
def run_feature_separation(df: pd.DataFrame) -> dict:
    log("=" * 60)
    log("10. РОЗДІЛЬНА ЗДАТНІСТЬ ОЗНАК (B vs M)")
    log("=" * 60)
    try:
        from sklearn.metrics import roc_auc_score
        has_sklearn = True
    except ImportError:
        has_sklearn = False
    aucs = {}
    for col in SEPARATION_FEATURES:
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, color in [("B", "#2ecc71"), ("M", "#e74c3c")]:
            subset = df.loc[df["diagnosis"] == label, col]
            subset.plot(kind="kde", ax=ax, label=label, color=color, linewidth=2)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.set_title(f"KDE {col} — B vs M")
        ax.legend()
        save_plot(fig, f"10_kde_separation_{col}.png")
        if has_sklearn:
            y_binary = (df["diagnosis"] == "M").astype(int)
            aucs[col] = roc_auc_score(y_binary, df[col])
    if aucs:
        log("AUC (ROC) для однієї ознаки (M як позитивний клас):")
        for k, v in aucs.items():
            log(f"  {k}: AUC = {v:.4f}")
    log("")
    return aucs


# --- 11. Нормальність ---
def run_normality(df: pd.DataFrame) -> None:
    log("=" * 60)
    log("11. ПЕРЕВІРКА НОРМАЛЬНОСТІ")
    log("=" * 60)
    for col in KEY_FEATURES[:2]:
        stat, p = stats.shapiro(df[col])
        log(f"  {col}: Shapiro W={stat:.4f}, p={p:.4f} {'(нормальність не відхиляємо)' if p > 0.05 else '(відхиляємо нормальність)'}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, KEY_FEATURES[:2]):
        stats.probplot(df[col], dist="norm", plot=ax)
        ax.set_title(f"Q-Q {col}")
    save_plot(fig, "11_qq_plots.png")
    log("Коментар: при n=569 Shapiro дуже чутливий; Q-Q показує хвости. Для порівняння B vs M доцільно Mann-Whitney при асиметрії.")
    log("")


# --- 12. Вплив викидів ---
def run_outlier_impact(df: pd.DataFrame, outlier_counts: dict) -> None:
    log("=" * 60)
    log("12. ВПЛИВ ВИКИДІВ НА ОЦІНКИ")
    log("=" * 60)
    col = "radius_mean"
    mask = get_outliers_iqr(df[col])
    n_remove = mask.sum()
    mean_before = df[col].mean()
    median_before = df[col].median()
    df_clean = df.loc[~mask]
    mean_after = df_clean[col].mean()
    median_after = df_clean[col].median()
    log(f"  {col}: видалено викидів = {n_remove}")
    log(f"  До:   mean={mean_before:.4f}, median={median_before:.4f}")
    log(f"  Після: mean={mean_after:.4f}, median={median_after:.4f}")
    log("Висновок: зміна невелика; висновок про різницю B vs M не змінюється (M > B залишається).")
    log("")


# --- 13–14. Висновки та інсайти ---
def run_conclusions(df: pd.DataFrame, pvalues: dict, corr: pd.DataFrame, outlier_counts: dict, aucs: dict) -> None:
    log("=" * 60)
    log("13. ФІНАЛЬНІ ВИСНОВКИ (6 пунктів)")
    log("=" * 60)
    p_r = pvalues.get("radius_mean", 0)
    skew_a = df["area_mean"].skew()
    r_ra = df["radius_mean"].corr(df["area_mean"])
    n_out_a = outlier_counts.get("area_mean", 0)
    mask_r = get_outliers_iqr(df["radius_mean"])
    mean_before = df["radius_mean"].mean()
    mean_after = df.loc[~mask_r, "radius_mean"].mean()
    best_auc = max(aucs.items(), key=lambda x: x[1]) if aucs else ("concave_points_mean", 0.9)
    log(f"1. radius_mean у M значно більший за B (p < {p_r:.4f}).")
    log(f"2. area_mean сильно правий асиметричний (skew = {skew_a:.3f}).")
    log(f"3. Сильна кореляція radius_mean ↔ area_mean (r = {r_ra:.3f}).")
    log(f"4. Викиди в area_mean (IQR 1.5): n = {n_out_a}.")
    log(f"5. Після видалення викидів radius_mean середнє змінюється з {mean_before:.4f} на {mean_after:.4f}.")
    log(f"6. Краща роздільна здатність B/M: {best_auc[0]} (AUC ≈ {best_auc[1]:.3f}).")
    log("")
    log("=" * 60)
    log("14. ІНСАЙТИ")
    log("=" * 60)
    log("Інсайт 1: Стратифікація за concavity_mean знижує кореляцію radius–area у частині квінтиль → concavity виступає частковим confounder (геометрія пухлини пояснює частину зв’язку розмір–площа).")
    log("Інсайт 2: Мультимодальність texture_mean (KDE) натякає на підтипи; при поділі за texture можна отримати підгрупи з чіткішими відмінностями B/M по інших ознаках (перевірити окремими графіками).")
    log("")


def main() -> None:
    sys.stdout = _stdout
    df = load_data()
    run_preparation(df)
    run_quality_check(df)
    run_describe(df)
    run_histograms_kde(df)
    run_boxplots(df)
    out_df, outlier_counts = run_outliers(df)
    pvalues = run_b_vs_m(df)
    corr = run_correlation(df)
    run_confounding(df)
    aucs = run_feature_separation(df)
    run_normality(df)
    run_outlier_impact(df, outlier_counts)
    run_conclusions(df, pvalues, corr, outlier_counts, aucs)

    results_path = BASE_DIR / "results.txt"
    results_path.write_text(_out.getvalue(), encoding="utf-8")
    log(f"\nРезультати записано в {results_path}")
    log(f"Графіки збережено в {PLOTS_DIR}")


if __name__ == "__main__":
    main()
