# ─── IMPORTS ───

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ─── CONFIGURATION ───

EXPR_FILE = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\outputs\gene_parcellation\group_region_gene.csv"
AMYLOID_FILE = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\outputs\amyloid_tau_parcellation\group_region_amyloid.csv"
TAU_FILE = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\outputs\amyloid_tau_parcellation\group_region_tau.csv"

OUTPUT_DIR = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\outputs\correlation_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

P_THRESHOLD = 0.05


# ─── COLOUR PALETTE ───

COL_SIG_POS     = "#1a6faf"   
COL_SIG_POS_EDG = "#0d4f8b"   

COL_SIG_NEG     = "#74b9e7"   
COL_SIG_NEG_EDG = "#4a9fd4"  


COL_NONSIG      = "#d6e8f7"   
COL_NONSIG_EDG  = "#a8cce4"

MODALITY_COLOURS = {
    "amyloid": {"sig": "#1a6faf", "nonsig": "#a8cce4", "scatter": "#1a6faf"},
    "tau":     {"sig": "#2e86c1", "nonsig": "#b3d4ea", "scatter": "#2e86c1"},
}
DEFAULT_COLOURS = {"sig": "#1565c0", "nonsig": "#90caf9", "scatter": "#1565c0"}


# ─── SPEARMAN CORRELATION ANALYSIS ───

# STEP 1: LOAD AND ALIGN DATA

def load_and_align(expr_file, modality_files):

    print("Loading data...")

    expr_df = pd.read_csv(expr_file, index_col=0)
    expr_df.index.name = "region"
    print(f"Expression matrix: {expr_df.shape}  (regions x genes)")

    neuro_series = {}

    for label, fpath in modality_files.items():

        if fpath is None:
            print(f"{label}: path is None - skipping.")
            continue

        if not os.path.exists(fpath):
            print(f"{label}: file not found at {fpath} - skipping.")
            continue

        df = pd.read_csv(fpath, index_col=0)
        df.index.name = "region"
        if "median_suvr" not in df.columns:
            print(f"WARNING: 'median_suvr' column missing in {fpath}.")
            continue

        neuro_series[label] = df["median_suvr"].rename(label)
        print(f"{label}: {len(df)} regions loaded.")

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

    expr_aligned  = expr_df.loc[common_regions]
    neuro_aligned = neuro_df.loc[common_regions]

    valid = expr_aligned.notna().all(axis=1)
    expr_aligned = expr_aligned[valid]
    neuro_aligned = neuro_aligned[valid]

    print(f"Final regions for analysis: {len(expr_aligned)}")
    print(f"  Genes: {expr_aligned.shape[1]}")
    print(f"  Modalities: {neuro_aligned.columns.tolist()}")

    return expr_aligned, neuro_aligned, expr_aligned.index.tolist()

# STEP 2: COMPUTE SPEARMAN CORRELATIONS

def run_correlation_analysis(expr_df, neuro_df, p_threshold, output_dir):

    rows = []

    for modality in neuro_df.columns.tolist():

        y_all = neuro_df[modality].values
        valid_neuro = ~np.isnan(y_all)

        if valid_neuro.sum() < 10:
            print(f"{modality}: too few valid regions - skipping.")
            continue

        print(f"{modality.upper()} — {valid_neuro.sum()} valid regions.")

        for gene in expr_df.columns.tolist():

            x_all = expr_df[gene].values
            valid = valid_neuro & ~np.isnan(x_all)
            x = x_all[valid]
            y = y_all[valid]

            if len(x) < 10:
                continue

            spearman_r, spearman_p = spearmanr(x, y)

            rows.append({
                "gene": gene,
                "modality": modality,
                "n_regions": int(len(x)),
                "spearman_r": round(spearman_r, 4),
                "spearman_p": round(spearman_p, 4),
                "significant": spearman_p < p_threshold,
            })

    if not rows:
        raise ValueError("ERROR: No results computed.")

    corr_df = pd.DataFrame(rows)
    corr_df = corr_df.sort_values(["modality", "spearman_r"], ascending=[True, False]).reset_index(drop=True)

    out_path = os.path.join(output_dir, "correlation_results.csv")
    corr_df.to_csv(out_path, index=False)
    print(f"Results saved: {out_path}")

    for modality in corr_df["modality"].unique():

        sub = corr_df[corr_df["modality"] == modality]
        n_pos = ((sub["significant"]) & (sub["spearman_r"] > 0)).sum()
        n_neg = ((sub["significant"]) & (sub["spearman_r"] < 0)).sum()
        
        print(f"{modality.upper()} nominally significant (p < {p_threshold}):")
        print(f"  Positive: {n_pos}")
        print(f"  Negative: {n_neg}")
        
        top5 = sub.nlargest(5, "spearman_r")[
            ["gene", "spearman_r", "spearman_p", "significant"]]
        print(f"Top 5 by Spearman r:\n{top5.to_string(index=False)}")

    return corr_df


# ─── DOT PLOT (PER-MODALITY + COMBINED) ───

def _draw_dotplot(ax, corr_df_subset, modalities, gene_order, p_threshold):

    n_genes = len(gene_order)
    n_modals = len(modalities)
    modal_x = {m: i for i, m in enumerate(modalities)}
    gene_y = {g: i for i, g in enumerate(gene_order[::-1])}

    for _, row in corr_df_subset.iterrows():

        gene = row["gene"]
        mod = row["modality"]
        r = row["spearman_r"]
        sig = row["significant"]

        if gene not in gene_y or mod not in modal_x:
            continue

        x = modal_x[mod]
        y = gene_y[gene]

        if sig and r > 0:
            colour, edge = COL_SIG_POS, COL_SIG_POS_EDG
        elif sig and r < 0:
            colour, edge = COL_SIG_NEG, COL_SIG_NEG_EDG
        else:
            colour, edge = COL_NONSIG, COL_NONSIG_EDG

        ax.scatter(x, y, s=abs(r) * 900 + 60,
                   c=colour, edgecolors=edge,
                   linewidths=1.5, zorder=3)
        
        ax.text(x, y, f"{r:.2f}",
                ha="center", va="center",
                fontsize=7.5,
                fontweight="bold" if sig else "normal",
                color="white" if sig else "#4a4a4a")

    ax.set_xticks(range(n_modals))
    ax.set_xticklabels([m.capitalize() for m in modalities], fontsize=12, fontweight="bold")
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(gene_order[::-1], fontsize=9)
    ax.set_xlim(-0.7, n_modals - 0.3)
    ax.set_ylim(-0.8, n_genes - 0.2)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Neurodegeneration Modality", fontsize=11, labelpad=8)
    ax.set_ylabel("AD-Associated Gene", fontsize=11, labelpad=8)

    legend_elements = [
        mpatches.Patch(color=COL_SIG_POS, label=f"Positive (p < {p_threshold})"),
        mpatches.Patch(color=COL_SIG_NEG, label=f"Negative (p < {p_threshold})"),
        mpatches.Patch(color=COL_NONSIG, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.92, edgecolor="#a8cce4")


def plot_dotplot(corr_df, output_dir, p_threshold):

    modalities = corr_df["modality"].unique().tolist()
    genes_all = corr_df["gene"].unique().tolist()

    sort_by = "amyloid" if "amyloid" in modalities else modalities[0]
    gene_order = (corr_df[corr_df["modality"] == sort_by].sort_values("spearman_r", ascending=False)["gene"].tolist())

    for g in genes_all:

        if g not in gene_order:
            gene_order.append(g)

    n_genes = len(gene_order)

    def _save_dotplot(mod_list, filename, title_suffix=""):

        n_m = len(mod_list)
        fig, ax = plt.subplots(figsize=(3.5 + n_m * 2.8, n_genes * 0.52 + 2.8))
        subset = corr_df[corr_df["modality"].isin(mod_list)]
        _draw_dotplot(ax, subset, mod_list, gene_order, p_threshold)
        ax.set_title(
            f"Spearman Correlation: Gene Expression vs Neurodegeneration"
            f"{title_suffix}\n"
            f"Dot size ∝ |ρ| — Nominal significance p < {p_threshold}",
            fontsize=12, fontweight="bold", pad=14
        )

        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Dot plot saved: {out_path}")

    _save_dotplot(modalities, "correlation_dotplot.png")

    if "amyloid" in modalities:
        _save_dotplot(["amyloid"], "correlation_dotplot_amyloid.png", title_suffix=" — Amyloid")

    if "tau" in modalities:
        _save_dotplot(["tau"], "correlation_dotplot_tau.png", title_suffix=" — Tau")


# ─── BAR CHART (PER-MODALITY + COMBINED) ───

def _draw_barchart(axes_row, modality_list, corr_df, p_threshold):

    for col_idx, modality in enumerate(modality_list):

        ax = axes_row[col_idx]
        sub = (corr_df[corr_df["modality"] == modality].sort_values("spearman_r", ascending=True).reset_index(drop=True))

        colours_map = MODALITY_COLOURS.get(modality, DEFAULT_COLOURS)
        bar_colours = [colours_map["sig"] if sig else colours_map["nonsig"] for sig in sub["significant"]]

        y_pos = range(len(sub))
        ax.barh(y_pos, sub["spearman_r"],
                color=bar_colours, edgecolor="white",
                linewidth=0.6, height=0.7)
        ax.axvline(0, color="#1a3a5c", linewidth=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["gene"], fontsize=9)
        ax.set_xlabel("Spearman ρ", fontsize=11)
        ax.set_title(f"{modality.capitalize()}\n(solid = p < {p_threshold})", fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

        for i, (r_val, p_val) in enumerate(zip(sub["spearman_r"], sub["spearman_p"])):
            offset = 0.005 if r_val >= 0 else -0.005
            ha = "left" if r_val >= 0 else "right"
            label = f"{r_val:.2f}*" if p_val < p_threshold else f"{r_val:.2f}"
            ax.text(r_val + offset, i, label, va="center", ha=ha, fontsize=7.5)

def plot_summary_barchart(corr_df, output_dir, p_threshold):

    modalities = corr_df["modality"].unique().tolist()
    n_genes = corr_df["gene"].nunique()

    def _save_barchart(mod_list, filename):

        n_m = len(mod_list)
        fig, axes = plt.subplots(
            1, n_m,
            figsize=(8 * n_m, max(6, n_genes * 0.4)),
            squeeze=False
        )

        _draw_barchart(axes[0], mod_list, corr_df, p_threshold)
        
        fig.suptitle(
            "Spearman Correlation: Gene Expression vs Neurodegeneration\n"
            f"* nominally significant (p < {p_threshold})",
            fontsize=13, fontweight="bold", y=1.01
        )

        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Bar chart saved: {out_path}")

    _save_barchart(modalities, "correlation_barchart.png")

    if "amyloid" in modalities:
        _save_barchart(["amyloid"], "correlation_barchart_amyloid.png")

    if "tau" in modalities:
        _save_barchart(["tau"], "correlation_barchart_tau.png")


# ─── SCATTER PLOTS FOR NOMINALLY SIGNIFICANT GENE-MODALITY PAIRS ───

def plot_scatter_significant(corr_df, expr_df, neuro_df, output_dir, p_threshold):

    SCATTER_DIR = os.path.join(output_dir, "correlation_scatter_plots")
    os.makedirs(SCATTER_DIR, exist_ok=True)

    sig_pairs = corr_df[corr_df["significant"]][["gene", "modality"]].values.tolist()

    if not sig_pairs:
        print("No nominally significant gene-modality pairs — no scatter plots generated.")
        return

    sig_genes = sorted(set(g for g, _ in sig_pairs))
    print(f"Generating scatter plots for {len(sig_pairs)} significant (gene, modality) pair(s) across {len(sig_genes)} gene(s)...")

    for gene, modality in sig_pairs:

        row_data = corr_df[
            (corr_df["gene"] == gene) &
            (corr_df["modality"] == modality)
        ].iloc[0]

        r = row_data["spearman_r"]
        p = row_data["spearman_p"]

        y_all = neuro_df[modality].values
        x_all = expr_df[gene].values
        valid = ~np.isnan(x_all) & ~np.isnan(y_all)
        x = x_all[valid]
        y = y_all[valid]

        colour = MODALITY_COLOURS.get(modality, DEFAULT_COLOURS)["scatter"]

        fig, ax = plt.subplots(figsize=(5.5, 4.8))

        ax.scatter(
            x, y,
            s=55, alpha=0.75,
            color=colour,
            edgecolors="white", linewidth=0.7,
            zorder=3
        )

        if len(x) > 3:
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, np.poly1d(z)(x_line),
                    color=colour, linewidth=1.8,
                    linestyle="--", alpha=0.7,
                    zorder=2)

        ax.set_xlabel(f"{gene} Expression (SRS units)", fontsize=11)
        ax.set_ylabel(f"{modality.capitalize()} SUVR",  fontsize=11)
        ax.set_title(
            f"{gene} vs {modality.capitalize()}\n"
            f"Spearman ρ = {r:.3f} p = {p:.4f}",
            fontsize=11, fontweight="bold"
        )
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()

        fname = f"scatter_plot_{modality}_{gene}.png"
        out_path = os.path.join(SCATTER_DIR, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved: {fname}")


# ─── MAIN ORCHESTRATOR ───

def main():
    print("Starting correlation analysis...")

    modality_files = {"amyloid": AMYLOID_FILE, "tau": TAU_FILE}
    print("Running for amyloid and tau...")

    expr_aligned, neuro_aligned, _ = load_and_align(EXPR_FILE, modality_files)

    corr_df = run_correlation_analysis(expr_aligned, neuro_aligned, P_THRESHOLD, OUTPUT_DIR)

    print("\nGenerating dot plots...")
    plot_dotplot(corr_df, OUTPUT_DIR, P_THRESHOLD)

    print("\nGenerating bar charts...")
    plot_summary_barchart(corr_df, OUTPUT_DIR, P_THRESHOLD)

    print("\nGenerating scatter plots...")
    plot_scatter_significant(corr_df, expr_aligned, neuro_aligned, OUTPUT_DIR, P_THRESHOLD)

    print(f"Analysis complete.")
    print(f"Outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()