import pandas as pd
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def save_results_to_csv(dataset_wit_IDs, predictions, file_name: str=None):
    # assume X_test (pd.DataFrame) and preds (array-like) are defined
    # choose ID column if present, otherwise use index
    id_col = 'ID' if 'ID' in dataset_wit_IDs.columns else None
    ids = dataset_wit_IDs[id_col].values if id_col else dataset_wit_IDs.index.values
    # ensure lengths match
    print(f"IDs length: {len(ids)}, Predictions length: {len(predictions)}")
    print(len(ids) == len(predictions))
    assert len(ids) == len(predictions), "Length mismatch between IDs and predictions"

    # Ensure 1D
    preds = np.asarray(predictions)
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds.ravel()

    out_df = pd.DataFrame({'ID': ids, 'TARGET': preds})

    # ensure output dir exists (current dir used here)
    out_dir = '.'
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"{file_name}_{ts}.csv"
    out_path = os.path.join(out_dir, fname)

    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to `{out_path}`")

# ============================================================
# EDA FUNCTIONS
# ============================================================

def plot_missing_data(df, title="Missing Data Analysis"):
    """Visualize missing data patterns in the dataset."""
    missing = df.isnull()
    missing_pct = (missing.sum() / len(df) * 100).sort_values(ascending=True)
    missing_pct = missing_pct[missing_pct > 0]

    if len(missing_pct) == 0:
        print("No missing values in the dataset.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    missing_pct.plot(kind='barh', ax=ax, color='coral', edgecolor='black')
    ax.set_xlabel('Missing (%)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.bar_label(ax.containers[0], fmt='%.1f%%', padding=3)
    plt.tight_layout()
    plt.show()


def plot_target_distribution(df, target_col='TARGET'):
    """Analyze target variable distribution overall and by country."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {'Overall': 'steelblue', 'DE': '#1f77b4', 'FR': '#ff7f0e'}

    # Overall distribution
    sns.histplot(df[target_col], kde=True, ax=axes[0], color=colors['Overall'])
    axes[0].axvline(df[target_col].mean(), color='red', linestyle='--',
                    label=f'Mean: {df[target_col].mean():.3f}')
    axes[0].axvline(df[target_col].median(), color='green', linestyle='--',
                    label=f'Median: {df[target_col].median():.3f}')
    axes[0].legend()
    axes[0].set_title('All Data')

    # By country
    for i, country in enumerate(['DE', 'FR']):
        subset = df[df['COUNTRY'] == country][target_col]
        sns.histplot(subset, kde=True, ax=axes[i+1], color=colors[country])
        axes[i+1].axvline(subset.mean(), color='red', linestyle='--')
        axes[i+1].set_title(f'{country} (n={len(subset)})')

    plt.suptitle('Target Distribution: Price Variation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, target_col='TARGET', figsize=(14, 12)):
    """Create Spearman correlation heatmap with target correlations highlighted."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude ID columns
    numeric_cols = [c for c in numeric_cols if c not in ['ID', 'DAY_ID']]

    corr_matrix = df[numeric_cols].corr(method='spearman')

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Spearman Correlation'})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Print top correlations with target
    if target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
        print("\nTop 10 Features Correlated with TARGET:")
        print("-" * 40)
        for feat, corr in target_corr.head(10).items():
            print(f"  {feat:<25} {corr:.4f}")


def plot_features_by_country(df, feature_pairs):
    """Compare feature distributions between Germany and France using box plots."""
    n_pairs = len(feature_pairs)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors = {'DE': '#1f77b4', 'FR': '#ff7f0e'}

    for i, (de_col, fr_col, label) in enumerate(feature_pairs):
        if i >= 4:
            break
        data_to_plot = [df[de_col].dropna(), df[fr_col].dropna()]
        labels = ['Germany (DE)', 'France (FR)']

        bp = axes[i].boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], [colors['DE'], colors['FR']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        axes[i].set_title(label, fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Normalized Value')

    for j in range(len(feature_pairs), 4):
        axes[j].set_visible(False)

    plt.suptitle('Feature Comparison: Germany vs France', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_features_vs_target(df, features, target_col='TARGET'):
    """Scatter plots of features against target, colored by country."""
    n_features = len(features)
    ncols = min(3, n_features)
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = {'DE': '#1f77b4', 'FR': '#ff7f0e'}

    for i, feature in enumerate(features):
        for country in ['DE', 'FR']:
            subset = df[df['COUNTRY'] == country]
            axes[i].scatter(subset[feature], subset[target_col],
                           alpha=0.4, s=20, c=colors[country], label=country)

        # Calculate correlation
        valid = df[[feature, target_col]].dropna()
        corr = spearmanr(valid[feature], valid[target_col])[0]
        axes[i].set_title(f'{feature}\n(Spearman: {corr:.3f})', fontsize=10)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target_col)
        if i == 0:
            axes[i].legend(loc='upper right', fontsize=8)

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Key Features vs Target', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def evaluate_predictions_by_country(X_original, y_true, y_pred, model_name="Model"):
    """
    Create prediction vs actual scatter plots separated by country.
    X_original must contain the COUNTRY column.
    """
    eval_df = pd.DataFrame({
        'COUNTRY': X_original['COUNTRY'].values,
        'Actual': np.array(y_true).flatten(),
        'Predicted': np.array(y_pred).flatten()
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {'DE': '#1f77b4', 'FR': '#ff7f0e'}

    # Get axis limits
    all_vals = np.concatenate([eval_df['Actual'], eval_df['Predicted']])
    lims = [all_vals.min() - 0.1, all_vals.max() + 0.1]

    # Overall performance
    for country in ['DE', 'FR']:
        subset = eval_df[eval_df['COUNTRY'] == country]
        axes[0].scatter(subset['Actual'], subset['Predicted'],
                       alpha=0.5, s=30, c=colors[country], label=country)
    axes[0].plot(lims, lims, 'k--', alpha=0.75, label='Perfect')
    overall_corr = spearmanr(eval_df['Actual'], eval_df['Predicted'])[0]
    axes[0].set_title(f'Overall\nSpearman: {overall_corr:.4f}', fontweight='bold')
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].legend()
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)

    # Germany
    de_data = eval_df[eval_df['COUNTRY'] == 'DE']
    axes[1].scatter(de_data['Actual'], de_data['Predicted'],
                   alpha=0.5, s=30, c=colors['DE'])
    axes[1].plot(lims, lims, 'k--', alpha=0.75)
    de_corr = spearmanr(de_data['Actual'], de_data['Predicted'])[0]
    axes[1].set_title(f'Germany (DE)\nSpearman: {de_corr:.4f}', fontweight='bold')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)

    # France
    fr_data = eval_df[eval_df['COUNTRY'] == 'FR']
    axes[2].scatter(fr_data['Actual'], fr_data['Predicted'],
                   alpha=0.5, s=30, c=colors['FR'])
    axes[2].plot(lims, lims, 'k--', alpha=0.75)
    fr_corr = spearmanr(fr_data['Actual'], fr_data['Predicted'])[0]
    axes[2].set_title(f'France (FR)\nSpearman: {fr_corr:.4f}', fontweight='bold')
    axes[2].set_xlabel('Actual')
    axes[2].set_ylabel('Predicted')
    axes[2].set_xlim(lims)
    axes[2].set_ylim(lims)

    plt.suptitle(f'{model_name}: Predicted vs Actual', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Summary table
    print("\n" + "=" * 55)
    print(f"  {model_name} - PERFORMANCE BY COUNTRY")
    print("=" * 55)
    print(f"{'Metric':<20} {'Germany':<12} {'France':<12} {'Overall':<12}")
    print("-" * 55)
    print(f"{'Spearman Corr.':<20} {de_corr:>10.4f}   {fr_corr:>10.4f}   {overall_corr:>10.4f}")
    print(f"{'Sample Count':<20} {len(de_data):>10}   {len(fr_data):>10}   {len(eval_df):>10}")
    print("=" * 55)

    return {'overall': overall_corr, 'DE': de_corr, 'FR': fr_corr}
