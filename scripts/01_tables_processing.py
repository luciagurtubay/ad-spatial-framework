# ─── IMPORTS ───

import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ─── PATH CONFIGURATION ───

BASE_DIR = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\gwas_tables"
INPUT_DIR = os.path.join(BASE_DIR, "tables_raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "tables_processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── CONFIGURATION ───

FINAL_COLS = ["chr", "position", "lead_SNP", "effect_allele", "other_allele", "EAF", "OR", "p_value", "assigned_gene"]

TABLES = ["seshadri", "lambert", "kunkle", "jansen", "schwartzentruber", "wightman", "rajabli"]

TABLE_LABELS = {
    "seshadri": "Seshadri 2010",
    "lambert": "Lambert 2013",
    "kunkle": "Kunkle 2019",
    "jansen": "Jansen 2019",
    "schwartzentruber": "Schwartzentruber 2021",
    "wightman": "Wightman 2021",
    "rajabli": "Rajabli 2021",
}

HEATMAP_CAP = 100


# ─── DATA PROCESSING ───

# GENE ANNOTATION HARMONIZATION  

GENE_MAP = {
    "INPPD5": ["INPP5D"],

    "CLU/PTK2B": ["CLU", "PTK2B"],
    "SLC24A4-RIN3": ["SLC24A4", "RIN3"],
    "USP6NL/ECHDC3": ["USP6NL", "ECHDC3"],
    "MADD/SPI1": ["MADD", "SPI1"],
    "SCIMP/RABEP1": ["SCIMP", "RABEP1"],
    "ZCWPW1/NYAP1": ["ZCWPW1", "NYAP1"],

    "APOE \u03b52": ["APOE"], 
    "APOE \u03b54": ["APOE"],      
    "AC074212.3": ["APOE"],

    "EPHA1-AS1": ["EPHA1"],
    "TSPOAP1-AS1": ["TSPOAP1"],

    "HLA-DRB5\u2013HLA-DRB1": ["HLA-DRB1"],  
    "HLA-DRB5-HLA-DRB1": ["HLA-DRB1"],  
}


def expand_genes(df):

    rows = []

    for _, row in df.iterrows():
        gene = str(row["assigned_gene"]).strip()
        mapped = GENE_MAP.get(gene)
        if mapped is None:
            rows.append(row)
        else:
            for g in mapped:
                new_row = row.copy()
                new_row["assigned_gene"] = g
                rows.append(new_row)

    return pd.DataFrame(rows, columns=df.columns).reset_index(drop=True)


# DATA STANDARDIZATION

def parse_p(val):

    if pd.isna(val):
        return None
    
    s = str(val).strip().lstrip("<>~")
    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"([\d.]+)\s*[×xX]\s*10\^?\{?\s*([+\-]?\d+)\}?", lambda m: m.group(1) + "e" + m.group(2), s)
    s = s.replace(" ", "")

    try:
        return float(s)
    except ValueError:
        return None


def parse_or(val):

    if pd.isna(val):
        return None
    
    m = re.match(r"([\d.]+)", str(val).strip())

    if m:
        return float(m.group(1))
    else: 
        return None


# ─── EXPORTS ───

def save(df, name):

    df = expand_genes(df)

    path = os.path.join(OUTPUT_DIR, name + ".csv")
    df[FINAL_COLS].to_csv(path, index=False, encoding="utf-8-sig")
    
    print(f"Saved: {path}")


# ─── GWAS PROCESSOR ───

# SESHADRI

def process_seshadri():

    df = pd.read_excel(os.path.join(INPUT_DIR, "seshadri.xlsx"), header=None)
    df = df.iloc[2:].reset_index(drop=True)

    chr_pos = df.iloc[:, 1].astype(str).str.split(":", expand=True)

    out = pd.DataFrame()

    out["chr"] = chr_pos[0]
    out["position"] = chr_pos[1]
    out["lead_SNP"] = df.iloc[:, 0]
    out["effect_allele"] = df.iloc[:, 4]
    out["other_allele"] = None
    out["EAF"] = pd.to_numeric(df.iloc[:, 5], errors="coerce")
    out["OR"] = df.iloc[:, 10].apply(parse_or)
    out["p_value"] = df.iloc[:, 11].apply(parse_p)
    out["assigned_gene"] = df.iloc[:, 3]

    save(out, "seshadri")


# LAMBERT

def process_lambert():

    df = pd.read_excel(os.path.join(INPUT_DIR, "lambert.xlsx"), header=None)
    df = df.iloc[2:].reset_index(drop=True)

    alleles = df.iloc[:, 4].astype(str).str.split("/", expand=True)

    out = pd.DataFrame()

    out["chr"] = df.iloc[:, 1]
    out["position"] = df.iloc[:, 2]
    out["lead_SNP"] = df.iloc[:, 0]
    out["effect_allele"] = alleles[1]    
    out["other_allele"] = alleles[0]  
    out["EAF"] = pd.to_numeric(df.iloc[:, 5], errors="coerce")
    out["OR"] = df.iloc[:, 10].apply(parse_or)
    out["p_value"] = df.iloc[:, 11].apply(parse_p)
    out["assigned_gene"] = df.iloc[:, 3]

    save(out, "lambert")


# KUNKLE

def process_kunkle():

    df = pd.read_excel(os.path.join(INPUT_DIR, "kunkle.xlsx"), header=None)
    df = df.iloc[2:].reset_index(drop=True)

    alleles = df.iloc[:, 4].astype(str).str.split("/", expand=True)

    out = pd.DataFrame()

    out["chr"] = df.iloc[:, 1]
    out["position"] = df.iloc[:, 2]
    out["lead_SNP"] = df.iloc[:, 0]
    out["effect_allele"] = alleles[1]    
    out["other_allele"] = alleles[0] 
    out["EAF"] = pd.to_numeric(df.iloc[:, 5], errors="coerce")
    out["OR"] = df.iloc[:, 12].apply(parse_or)   
    out["p_value"] = df.iloc[:, 14].apply(parse_p)    
    out["assigned_gene"] = df.iloc[:, 3]

    save(out, "kunkle")


# JANSEN

def process_jansen():

    df = pd.read_excel(os.path.join(INPUT_DIR, "jansen.xlsx"), header=None)
    df = df.iloc[2:].reset_index(drop=True)

    out = pd.DataFrame()

    out["chr"] = df.iloc[:, 1]
    out["position"] = df.iloc[:, 8]
    out["lead_SNP"] = df.iloc[:, 7]
    out["effect_allele"] = df.iloc[:, 9]
    out["other_allele"] = df.iloc[:, 10]
    out["EAF"] = pd.to_numeric(df.iloc[:, 11], errors="coerce")
    out["OR"] = None
    out["p_value"] = df.iloc[:, 13].apply(parse_p)
    out["assigned_gene"] = df.iloc[:, 2]

    save(out, "jansen")


# SCHWARTZENTRUBER

def process_schwartzentruber():

    df = pd.read_excel(os.path.join(INPUT_DIR, "schwartzentruber.xlsx"), header=None)
    df = df.iloc[1:].reset_index(drop=True)

    out = pd.DataFrame()

    out["chr"] = None
    out["position"] = None
    out["lead_SNP"] = df.iloc[:, 1]
    out["effect_allele"] = df.iloc[:, 4]
    out["other_allele"] = None
    out["EAF"] = pd.to_numeric(df.iloc[:, 5], errors="coerce")
    out["OR"] = df.iloc[:, 3].apply(parse_or)
    out["p_value"] = df.iloc[:, 2].apply(parse_p)
    out["assigned_gene"] = df.iloc[:, 0]

    save(out, "schwartzentruber")


# WIGHTMAN

def process_wightman():

    df = pd.read_excel(os.path.join(INPUT_DIR, "wightman.xlsx"), header=None)
    df = df.iloc[1:].reset_index(drop=True)

    pos = df.iloc[:, 2].astype(str).str.replace(",", "", regex=False).str.split(":", expand=True)

    out = pd.DataFrame()

    out["chr"] = pos[0]
    out["position"] = pos[1]
    out["lead_SNP"] = df.iloc[:, 3]
    out["effect_allele"] = df.iloc[:, 4]
    out["other_allele"] = None
    out["EAF"] = pd.to_numeric(df.iloc[:, 5], errors="coerce")
    out["OR"] = None
    out["p_value"] = df.iloc[:, 6].apply(parse_p)
    out["assigned_gene"] = df.iloc[:, 1]

    save(out, "wightman")


# RAJABLI

def process_rajabli():

    df = pd.read_excel(os.path.join(INPUT_DIR, "rajabli.xlsx"), header=None)
    df = df.iloc[2:].reset_index(drop=True)

    alleles = df.iloc[:, 4].astype(str).str.split("/", expand=True)

    out = pd.DataFrame()

    out["chr"] = df.iloc[:, 1]
    out["position"] = df.iloc[:, 2].astype(str).str.replace(",", "", regex=False)
    out["lead_SNP"] = df.iloc[:, 0].astype(str).str.lstrip("-")
    out["effect_allele"] = alleles[1]
    out["other_allele"] = alleles[0]
    out["EAF"] = None
    out["OR"] = df.iloc[:, 5].apply(parse_or)
    out["p_value"] = df.iloc[:, 6].apply(parse_p)
    out["assigned_gene"] = df.iloc[:, 3]

    save(out, "rajabli")


# ─── MERGED TABLE ───

def build_merged_table():

    individual_tables = []

    for table in TABLES:

        path = os.path.join(OUTPUT_DIR, table + ".csv")
        df = pd.read_csv(path, encoding="utf-8-sig")

        df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
        df["assigned_gene"] = df["assigned_gene"].astype(str).str.strip()

        lower_p_value = df.groupby("assigned_gene")["p_value"].min().reset_index()
        lower_p_value.columns = ["assigned_gene", table]

        individual_tables.append(lower_p_value)
 
    merged_table = individual_tables[0]
    for individual_table in individual_tables[1:]:
        merged_table = pd.merge(merged_table, individual_table, on="assigned_gene", how="outer")
 
    path = os.path.join(BASE_DIR, "AD_genes_pvalues.csv")
    merged_table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path}")

    return merged_table
 
 
# ─── HEATMAPS FOR VISUALIZATION ───
 
def draw_heatmap(data, title, filename):

    neg_log_p = -np.log10(data).clip(upper=HEATMAP_CAP)
    neg_log_p.columns = [TABLE_LABELS[s] for s in TABLES]
    mask_nan = neg_log_p.isna()
 
    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(neg_log_p, mask=~mask_nan, cmap=["#D0D0D0"],
                cbar=False, linewidths=0.5, linecolor="white", ax=ax)
 
    sns.heatmap(neg_log_p, mask=mask_nan, cmap="Blues",
                vmin=0, vmax=HEATMAP_CAP, linewidths=0.5, linecolor="white", ax=ax, 
                cbar_kws={"label":  f"\u2212log\u2081\u2080(p-value) [capped at {HEATMAP_CAP}]"})
  
    cbar = ax.collections[-1].colorbar
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("GWAS meta-analysis", fontsize=9, labelpad=10)
    ax.set_ylabel("Gene", fontsize=9, labelpad=10)
 
    ax.tick_params(axis="x", labelsize=7, rotation=0)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])

    path = os.path.join(BASE_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")
 

def build_heatmaps(merged_table):
    merged_table_sorted = merged_table.set_index("assigned_gene").sort_index()[TABLES] 
 
    midpoint = len(merged_table_sorted) // 2
    heatmap_1 = merged_table_sorted.iloc[:midpoint]
    heatmap_2 = merged_table_sorted.iloc[midpoint:]
 
    draw_heatmap(
        heatmap_1,
        title = "AD gene p-values for GWAS (A\u2013K)",
        filename = "heatmap_genes_A_K.png"
        )
    draw_heatmap(
        heatmap_2,
        title = "AD gene p-values for GWAS (L\u2013Z)",
        filename = "heatmap_genes_L_Z.png"
        )


# ─── PIPELINE EXECUTION ───

if __name__ == "__main__":

    processors = {
        "seshadri": process_seshadri,
        "lambert": process_lambert,
        "kunkle": process_kunkle,
        "jansen": process_jansen,
        "schwartzentruber": process_schwartzentruber,
        "wightman": process_wightman,
        "rajabli": process_rajabli,
    }

    for name, func in processors.items():

        print(f"Processing {name}...")

        try:
            func()
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()

    print("Building merged table...")
    merged_table = build_merged_table()

    print("Building heatmaps...")
    build_heatmaps(merged_table)