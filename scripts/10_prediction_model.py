# ─── IMPORTS ───

import os
import pandas as pd
from sklearn.model_selection import LeaveOneOut
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import json
import warnings
warnings.filterwarnings("ignore")


# ─── CONFIGURATION ───

EXPR_FILE = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\outputs\gene_parcellation\group_region_gene.csv"
AMYLOID_FILE = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\outputs\amyloid_tau_parcellation\group_region_amyloid.csv"
TAU_FILE = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\outputs\amyloid_tau_parcellation\group_region_tau.csv"

OUTPUT_DIR = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\outputs\prediction_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── PARAMETERS ───

GB_N_ESTIMATORS = 100
GB_MAX_DEPTH = 3
GB_LEARNING_RATE = 0.05
GB_SUBSAMPLE = 0.8
GB_RANDOM_STATE = 42

PERM_N_REPEATS = 50

TOP_N_GENES = 15


# ─── MACHINE LEARNING MODEL PREDICTION ───

# STEP 1: LOAD AND ALIGN DATA 

def load_and_align(expr_file, modality_files):
 
    print("Loading data...")

    expr_df = pd.read_csv(expr_file, index_col=0)
    expr_df.index.name = "region"
    print(f"Expression matrix: {expr_df.shape}  (regions x genes)")

    neuro_series = {}

    for label, fpath in modality_files.items():

        if fpath is None:
            print(f"{label}: path is None — skipping.")
            continue

        if not os.path.exists(fpath):
            print(f"{label}: file not found — skipping.")
            continue

        df = pd.read_csv(fpath, index_col=0)
        df.index.name = "region"
        if "median_suvr" not in df.columns:
            print(f"WARNING: 'median_suvr' column missing in {fpath}.")
            continue

        neuro_series[label] = df["median_suvr"].rename(label)
        print(f"{label}: {len(df)} regions loaded")

    if not neuro_series:
        raise ValueError("No neurodegeneration data loaded.")

    neuro_df = pd.DataFrame(neuro_series)
    neuro_df.index.name = "region"

    common_regions = expr_df.index.intersection(neuro_df.index).tolist()

    only_expr = set(expr_df.index) - set(neuro_df.index)
    only_neuro = set(neuro_df.index) - set(expr_df.index)

    if only_expr:
        print(f"In gene expression only: {len(only_expr)} regions.")
    if only_neuro:
        print(f"In neurodegeneration SUVR only: {len(only_neuro)} regions.")

    expr_aligned = expr_df.loc[common_regions]
    neuro_aligned = neuro_df.loc[common_regions]

    valid = expr_aligned.notna().all(axis=1)
    expr_aligned  = expr_aligned[valid]
    neuro_aligned = neuro_aligned[valid]

    print(f"Final regions: {len(expr_aligned)}")
    print(f"  Genes: {expr_aligned.shape[1]}")
    print(f"  Modalities: {neuro_aligned.columns.tolist()}")

    return expr_aligned, neuro_aligned

# STEP 2: LEAVE-ONE-OUT CROSS-VALIDATION 

def run_loocv(X_scaled, y, genes):

    loo = LeaveOneOut()
    n_samples = len(y)
    ridge_preds = np.zeros(n_samples)
    gb_preds = np.zeros(n_samples)

    print(f"Running LOO-CV ({n_samples} folds)...")

    for fold, (train_idx, test_idx) in enumerate(loo.split(X_scaled)):

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train = y[train_idx]

        # RIDGE REGRESSION PREDICTION

        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
        ridge.fit(X_train, y_train)
        ridge_preds[test_idx] = ridge.predict(X_test)


        # GRADIENT BOOSTING PREDICTION

        gb = GradientBoostingRegressor(
            n_estimators=GB_N_ESTIMATORS,
            max_depth=GB_MAX_DEPTH,
            learning_rate=GB_LEARNING_RATE,
            subsample=GB_SUBSAMPLE,
            random_state=GB_RANDOM_STATE
        )
        gb.fit(X_train, y_train)
        gb_preds[test_idx] = gb.predict(X_test)

    return ridge_preds, gb_preds


# ─── FEATURE IMPORTANCE ───

def compute_feature_importance(X_scaled, y, genes):

    print("Computing feature importance...")

    # RIDGE REGRESSION ABSOLUTE COEFFICIENTS

    final_ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
    final_ridge.fit(X_scaled, y)

    ridge_coeff_abs = pd.Series(np.abs(final_ridge.coef_), index=genes)


    # GRADIENT BOOSTING PERMUTATION IMPORTANCE

    final_gb = GradientBoostingRegressor(
        n_estimators=GB_N_ESTIMATORS,
        max_depth=GB_MAX_DEPTH,
        learning_rate=GB_LEARNING_RATE,
        subsample=GB_SUBSAMPLE,
        random_state=GB_RANDOM_STATE
    )
    final_gb.fit(X_scaled, y)

    perm = permutation_importance(
        final_gb, X_scaled, y,
        n_repeats=PERM_N_REPEATS,
        random_state=GB_RANDOM_STATE
    )
    gb_importance = pd.Series(perm.importances_mean, index=genes)
    gb_importance_std = pd.Series(perm.importances_std,  index=genes)
    
    importance_df = pd.DataFrame({
        "gene": gb_importance.index,
        "gb_importance": gb_importance.values.round(6),
        "gb_importance_std": gb_importance_std.values.round(6),
        "ridge_coeff_abs": ridge_coeff_abs.values.round(6),
    })
    importance_df = importance_df.sort_values("gb_importance", ascending=False).reset_index(drop=True)
    importance_df.index += 1 
    importance_df.index.name = "rank"

    return importance_df, final_ridge, final_gb


# ─── FIGURES ───

def plot_predicted_vs_actual(y, ridge_preds, gb_preds, ridge_r2, gb_r2, modality, output_dir):

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Predicted vs Actual {modality.capitalize()} SUVR (Leave-One-Out CV, n = {len(y)} regions)", fontsize=13, fontweight="bold")

    for ax, (model_name, preds, r2) in zip(axes, [("Ridge Regression", ridge_preds, ridge_r2), ("Gradient Boosting", gb_preds, gb_r2),]):

        ax.scatter(
            y, preds,
            s=55, alpha=0.75,
            color="steelblue",
            edgecolors="white", linewidth=0.7,
            zorder=3
        )

        lims = [
            min(y.min(), preds.min()) - 0.05,
            max(y.max(), preds.max()) + 0.05,
        ]

        ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1.2, label="Perfect prediction")
        ax.set_xlabel(f"Actual {modality.capitalize()} SUVR", fontsize=11)
        ax.set_ylabel("Predicted SUVR", fontsize=11)
        ax.set_title(f"{model_name}\nR² = {r2:.3f}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "predicted_vs_actual.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def plot_feature_importance(importance_df, modality, output_dir, top_n):

    top = importance_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(top))

    ax.barh(
        y_pos,
        top["gb_importance"].values[::-1],
        xerr=top["gb_importance_std"].values[::-1],
        color="#3498db", alpha=0.85,
        error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "#2c3e50"},
        height=0.7
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["gene"].values[::-1], fontsize=10)
    ax.set_xlabel("Permutation Importance (mean decrease in R²)", fontsize=11)
    ax.set_title(
        f"Top {len(top)} Genes Predicting {modality.capitalize()} Burden\n"
        f"Gradient Boosting — permutation importance ± 1 SD ({PERM_N_REPEATS} repeats)",
        fontsize=12, fontweight="bold"
    )
    ax.axvline(0, color="#2c3e50", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


# ─── SINGLE MODALITY EXECUTION ───

def run_modality(modality, expr_df, neuro_df, output_dir):

    print(f"Modality: {modality.upper()}")

    os.makedirs(output_dir, exist_ok=True)

    genes = expr_df.columns.tolist()

    y_raw = neuro_df[modality].values
    valid = ~np.isnan(y_raw) & expr_df.notna().all(axis=1).values
    X_raw = expr_df.values[valid]
    y = y_raw[valid]
    regions = expr_df.index[valid].tolist()
    n_samples = len(y)

    print(f"Regions: {n_samples}, Genes: {len(genes)}")

    if n_samples < 15:
        print(f"Too few samples ({n_samples}) - skipping {modality}.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    ridge_preds, gb_preds = run_loocv(X_scaled, y, genes)

    ridge_r2 = r2_score(y, ridge_preds)
    ridge_mae = mean_absolute_error(y, ridge_preds)
    gb_r2 = r2_score(y, gb_preds)
    gb_mae = mean_absolute_error(y, gb_preds)

    best_model = "gradient_boosting" if gb_r2 > ridge_r2 else "ridge"

    print(f"Ridge Regression:")
    print(f"  R²  = {ridge_r2:.4f}")
    print(f"  MAE = {ridge_mae:.4f}")

    print(f"Gradient Boosting:")
    print(f"  R²  = {gb_r2:.4f}")
    print(f"  MAE = {gb_mae:.4f}")

    print(f"Best model: {best_model}")

    importance_df, final_ridge, final_gb = compute_feature_importance(X_scaled, y, genes)
    print(f"Top 5 genes by permutation importance:")
    print(importance_df.head(5)[["gene", "gb_importance", "gb_importance_std"]].to_string())

    metrics = {
        "modality": modality,
        "n_regions": n_samples,
        "n_genes": len(genes),
        "ridge": {
            "r2": round(ridge_r2,  4),
            "mae": round(ridge_mae, 4),
        },
        "gradient_boosting": {
            "r2": round(gb_r2,  4),
            "mae": round(gb_mae, 4),
        },
        "best_model": best_model,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "region": regions,
        "actual": y.round(5),
        "ridge_pred": ridge_preds.round(5),
        "gb_pred": gb_preds.round(5),
    }).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"))

    plot_predicted_vs_actual(y, ridge_preds, gb_preds, ridge_r2, gb_r2, modality, output_dir)
    plot_feature_importance(importance_df, modality, output_dir, TOP_N_GENES)

    print(f"All outputs saved to: {output_dir}")

    return metrics


# ─── MAIN ORCHESTRATOR ───

def main():

    print("Starting ML model prediction...")

    modality_files = {"amyloid": AMYLOID_FILE, "tau": TAU_FILE}
    print("Running for amyloid and tau...")

    expr_aligned, neuro_aligned = load_and_align(EXPR_FILE, modality_files)

    all_metrics = []
    for modality in neuro_aligned.columns.tolist():
        mod_output_dir = os.path.join(OUTPUT_DIR, modality)
        metrics = run_modality(
            modality, expr_aligned, neuro_aligned,
            mod_output_dir
        )
        if metrics:
            all_metrics.append(metrics)

    if all_metrics:
        summary_rows = [{
            "Modality": m["modality"],
            "N regions": m["n_regions"],
            "Ridge R²": m["ridge"]["r2"],
            "Ridge MAE": m["ridge"]["mae"],
            "GB R²": m["gradient_boosting"]["r2"],
            "GB MAE": m["gradient_boosting"]["mae"],
            "Best": m["best_model"],
        } for m in all_metrics]
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "model_summary.csv"), index=False)
        print("Model summary:")
        print(summary_df.to_string(index=False))

    print(f"ML model prediction complete.")
    print(f"Outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()