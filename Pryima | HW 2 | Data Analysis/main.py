import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import os
import io
import sys

data_path = 'datasets/diagnostic/wdbc.data'

feature_names = [
    'radius', 'texture', 'perimeter', 'area', 'smoothness', 
    'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'
]

columns = ['ID', 'diagnosis']
for prefix in ['mean_', 'se_', 'worst_']:
    columns.extend([f'{prefix}{name}' for name in feature_names])

df = pd.read_csv(data_path, header=None, names=columns)

numeric_features = [col for col in df.columns if col not in ['ID', 'diagnosis']]

print(f"Total number of numerical features: {len(numeric_features)}")
print(f"Number of observations: {len(df)}")

output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

output_text = io.StringIO()
original_stdout = sys.stdout

class TeeOutput:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
        output_text.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = TeeOutput(original_stdout)

def analyze_distribution_shape(data, feature_name):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    median = np.median(data)
    mean = np.mean(data)
    std = np.std(data)
    
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    if abs(skewness) < 0.5:
        tail_description = "symmetric"
    elif skewness > 0.5:
        tail_description = "right tail longer (positive skewness)"
    else:
        tail_description = "left tail longer (negative skewness)"
    
    iqr_ratio = iqr / (np.percentile(data, 95) - np.percentile(data, 5))
    if iqr_ratio > 0.5:
        iqr_description = "large (more than 50% of interquartile range)"
    elif iqr_ratio > 0.3:
        iqr_description = "medium (30-50% of interquartile range)"
    else:
        iqr_description = "small (less than 30% of interquartile range)"
    
    if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
        normality_description = "close to normal"
    elif abs(skewness) < 1.0 and abs(kurtosis) < 1.0:
        normality_description = "moderately similar to normal"
    else:
        normality_description = "differs from normal"
    
    return {
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'median': median,
        'mean': mean,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'tail_description': tail_description,
        'iqr_description': iqr_description,
        'normality_description': normality_description,
        'iqr_ratio': iqr_ratio
    }

n_features = len(numeric_features)
n_cols = 5
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
axes = axes.flatten() if n_features > 1 else [axes]

distribution_analyses = {}
shapiro_results = {}

for idx, feature in enumerate(numeric_features):
    data = df[feature].dropna()
    
    ax = axes[idx]
    n, bins, patches = ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    median = np.median(data)
    
    ax.axvline(q1, color='red', linestyle='--', linewidth=2, label=f'Q1 = {q1:.2f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median = {median:.2f}')
    ax.axvline(q3, color='red', linestyle='--', linewidth=2, label=f'Q3 = {q3:.2f}')
    
    ax.text(0.02, 0.95, f'IQR = {iqr:.2f}', transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    analysis = analyze_distribution_shape(data, feature)
    distribution_analyses[feature] = analysis
    
    test_data = data if len(data) <= 5000 else data.sample(5000, random_state=42)
    shapiro_stat, shapiro_p = stats.shapiro(test_data)
    shapiro_results[feature] = {
        'statistic': shapiro_stat,
        'p_value': shapiro_p,
        'is_normal': shapiro_p > 0.05
    }

for idx in range(n_features, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'all_distributions.png'), dpi=300, bbox_inches='tight')
print(f"\nPlots saved to {output_dir}/all_distributions.png")
plt.close()

print("\n" + "="*80)
print("ANALYSIS OF NUMERICAL FEATURES DISTRIBUTIONS")
print("="*80)

for feature in numeric_features:
    analysis = distribution_analyses[feature]
    shapiro = shapiro_results[feature]
    
    print(f"\n{'='*80}")
    print(f"Feature: {feature}")
    print(f"{'='*80}")
    print(f"Basic statistics:")
    print(f"  Mean: {analysis['mean']:.4f}")
    print(f"  Median: {analysis['median']:.4f}")
    print(f"  Standard deviation: {analysis['std']:.4f}")
    print(f"  Q1: {analysis['q1']:.4f}")
    print(f"  Q3: {analysis['q3']:.4f}")
    print(f"  IQR: {analysis['iqr']:.4f}")
    print(f"\nDistribution shape analysis:")
    print(f"  1. Tail size and shape: {analysis['tail_description']}")
    print(f"     (Skewness coefficient: {analysis['skewness']:.4f})")
    print(f"  2. IQR size relative to the rest of the distribution: {analysis['iqr_description']}")
    print(f"     (IQR relative to 90% range: {analysis['iqr_ratio']:.2%})")
    print(f"  3. Similarity to normal distribution: {analysis['normality_description']}")
    print(f"     (Kurtosis coefficient: {analysis['kurtosis']:.4f})")
    print(f"\nShapiro-Wilk test:")
    print(f"  Statistic: {shapiro['statistic']:.6f}")
    print(f"  p-value: {shapiro['p_value']:.6f}")
    if shapiro['is_normal']:
        print(f"  Conclusion: Distribution does NOT differ from normal (p > 0.05)")
    else:
        print(f"  Conclusion: Distribution DIFFERS from normal (p ≤ 0.05)")

print("\n" + "="*80)
print("SUMMARY TABLE OF RESULTS")
print("="*80)

summary_data = []
for feature in numeric_features:
    analysis = distribution_analyses[feature]
    shapiro = shapiro_results[feature]
    summary_data.append({
        'Feature': feature,
        'Skewness': f"{analysis['skewness']:.3f}",
        'Kurtosis': f"{analysis['kurtosis']:.3f}",
        'IQR/Range': f"{analysis['iqr_ratio']:.2%}",
        'Shapiro-Wilk p-value': f"{shapiro['p_value']:.6f}",
        'Normal?': 'Yes' if shapiro['is_normal'] else 'No'
    })

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

summary_df.to_csv(os.path.join(output_dir, 'distribution_summary.csv'), index=False)
print(f"\nSummary table saved to {output_dir}/distribution_summary.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETED")
print("="*80)

sys.stdout = sys.__stdout__

def create_pdf_report(output_text, pdf_path):
    pdf = PdfPages(pdf_path)
    
    full_text = output_text.getvalue()
    lines = full_text.split('\n')
    
    fig_height = 11
    fig_width = 8.5
    lines_per_page = 55
    font_size = 7
    
    for page_start in range(0, len(lines), lines_per_page):
        page_lines = lines[page_start:page_start + lines_per_page]
        page_text = '\n'.join(page_lines)
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.05, 0.95, page_text, transform=ax.transAxes, 
                fontsize=font_size, verticalalignment='top', 
                family='monospace', wrap=False)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    pdf.close()

pdf_path = os.path.join(output_dir, 'analysis_report.pdf')
create_pdf_report(output_text, pdf_path)
print(f"\nPDF report saved to {pdf_path}")
