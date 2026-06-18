# ─── IMPORTS ───

import pandas as pd
import os
import numpy as np
from scipy.stats import combine_pvalues
import matplotlib.pyplot as plt


# ─── PATH CONFIGURATION ───

BASE_INPUT_DIR = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\gwas_tables"
INPUT_DIR = os.path.join(BASE_INPUT_DIR, "tables_processed")
OUTPUT_DIR = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\ad_gene_prioritisation"


# ─── CONFIGURATION PARAMETERS ───

TABLES = ["seshadri", "lambert", "kunkle", "jansen", "schwartzentruber", "wightman", "rajabli"]

GWS_THRESHOLD = 5e-8
FISHER_THRESHOLD = 1e-20


# ─── GWAS TABLES LOADING ───

merged_table = pd.read_csv(os.path.join(BASE_INPUT_DIR, "AD_genes_pvalues.csv"), encoding="utf-8-sig")
merged_table = merged_table.set_index("assigned_gene")


# ─── APOE HANDLING ───

apoe_row = merged_table.loc[["APOE"]].copy() if "APOE" in merged_table.index else None
merged_table = merged_table.drop(index="APOE", errors="ignore")


# ─── ASSOCIATION AGGREGATION (FISHER'S COMBINED PROBABILITY METHOD) AND GENE RANKING ───

def fisher_p(row):

    p_values = row.dropna().values
    if len(p_values) == 0:
        return np.nan
    
    _, combined_p_value = combine_pvalues(p_values, method="fisher")
    return combined_p_value

merged_table["fisher_p"] = merged_table.apply(fisher_p, axis=1)
merged_table["n_studies_gws"] = (merged_table[TABLES] < GWS_THRESHOLD).sum(axis=1)

sorted_table = (merged_table.dropna(subset=["fisher_p"]).sort_values("fisher_p", ascending=True).reset_index())
sorted_table["rank"] = sorted_table.index + 1


# ─── DATA VISUALIZATION ───

def style_axis(ax):

    ax.set_yscale("log")
    ax.invert_yaxis()

    ax.set_xlabel("Gene rank", fontsize=12)
    ax.set_ylabel("Fisher combined p-value", fontsize=12)

    ax.grid(True, which="both", linestyle="--", alpha=0.35, color="grey")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

if FISHER_THRESHOLD is not None:

    selected = sorted_table[(sorted_table["fisher_p"] < FISHER_THRESHOLD) & (sorted_table["n_studies_gws"] >= 2)].copy()
    not_selected = sorted_table[~sorted_table.index.isin(selected.index)]

    fig2, ax2 = plt.subplots(figsize=(11, 6))

    ax2.scatter(not_selected["rank"], not_selected["fisher_p"],
                color="#C8DDE8", s=30, zorder=2, edgecolors="white", linewidths=0.3,
                label="Not selected genes")

    ax2.scatter(selected["rank"], selected["fisher_p"],
                color="#7EB8D4", s=60, zorder=3, edgecolors="white", linewidths=0.4,
                label=f"Selected genes")

    ax2.axhline(y=FISHER_THRESHOLD,
                color="#1A5276", linestyle="--", linewidth=1.5,
                label=f"Threshold: Fisher p\u200a=\u200a{FISHER_THRESHOLD:.2e}")

    for _, row in selected.iterrows():
        ax2.annotate(row["assigned_gene"],
                     xy=(row["rank"], row["fisher_p"]),
                     xytext=(4, 0), textcoords="offset points", fontsize=5, va="center", color="#1A252F")

    style_axis(ax2)
    ax2.set_title("AD-associated gene prioritisation by combined evidence", fontsize=13, fontweight="bold", pad=12)
    ax2.legend(fontsize=9, framealpha=0.9, edgecolor="#CCCCCC")

    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "AD_gene_prioritisation.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved: {plot_path}")

    plt.show()
    
    # ─── OUTPUT GENERATION ───
    
    if apoe_row is not None:
        apoe_pvalues = apoe_row[TABLES].iloc[0].dropna().values
        _, apoe_fisher = combine_pvalues(apoe_pvalues, method="fisher")
        apoe_gws = int((apoe_row[TABLES].iloc[0] < GWS_THRESHOLD).sum())
        apoe_entry = pd.DataFrame([{"assigned_gene": "APOE", "fisher_p": apoe_fisher, "n_studies_gws": apoe_gws, "rank": 0}])
        selected_out = pd.concat([apoe_entry, selected[["assigned_gene", "fisher_p", "n_studies_gws", "rank"]]], ignore_index=True)

    else:
        selected_out = selected[["assigned_gene", "fisher_p", "n_studies_gws", "rank"]]

    df_path = os.path.join(OUTPUT_DIR, "AD_prioritised_genes.csv")
    selected_out.to_csv(df_path, index=False, encoding="utf-8-sig")

    print(f"Files saved:")
    print(f"  AD_prioritised_genes.csv")
    print(f"  AD_gene_prioritisation.png")

else:
    print("FISHER_THRESHOLD is not set.")