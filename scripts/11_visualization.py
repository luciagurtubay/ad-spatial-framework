# ─── IMPORTS ───

import os
import pandas as pd
import numpy as np
import subprocess
from pdf2image import convert_from_path
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ─── CONFIGURATION ───

EXPR_FILE = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\outputs\group_gene_expression.csv"
AMYLOID_FILE = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\outputs\amyloid_tau_parcellation\group_region_amyloid.csv"
TAU_FILE = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\outputs\amyloid_tau_parcellation\group_region_tau.csv"
CORR_FILE = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\correlation_analysis\correlation_results.csv"

OUTPUT_DIR = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\outputs\visualizations"

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RBRAIN_2C = os.path.join(SCRIPT_DIR, "bin", "Rbrain.R")
RBRAIN_3C = os.path.join(SCRIPT_DIR, "bin", "Rbrain3c.R")

RSCRIPT = r"C:\Program Files\R\R-4.3.1\bin\Rscript.exe"   
POPPLER_PATH = r"C:\Program Files\poppler-26.02.0\Library\bin"


# ─── PARAMETERS ───

ATLAS = "dkt"
DPI = 150
P_THRESHOLD = 0.05


# ─── COLOUR PALETTE ───

CMAP_LOW = "#deebf7"
CMAP_HIGH = "#08519c"

LAB_LOW  = "#deebf7"
LAB_HIGH = "#08519c"


# ─── PARCELLATED BRAIN MAP RENDERING ───

# STEP O: ESTABLISH SETUP

TMP_DIR = os.path.join(OUTPUT_DIR, "_tmp")
for subdir in ["gene_maps", "neuro_maps", "comparison_panels", "_tmp"]:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

# STEP 1: LOAD DATA

def load_all_data():

    print("Loading data...")

    expr_df = pd.read_csv(EXPR_FILE,    index_col=0)
    amyloid_df = pd.read_csv(AMYLOID_FILE, index_col=0)
    tau_df = pd.read_csv(TAU_FILE,     index_col=0)
    corr_df = pd.read_csv(CORR_FILE)

    for df in [expr_df, amyloid_df, tau_df]:
        df.index.name = "region"

    print(f"Expression: {expr_df.shape}")
    print(f"Amyloid: {amyloid_df.shape}")
    print(f"Tau: {tau_df.shape}")
    print(f"Correlations: {len(corr_df)} rows")

    return expr_df, amyloid_df, tau_df, corr_df

# STEP 2: REGION NAME CONVERSION

def atlas_to_ggseg_cortical(region_name):
    for prefix in ("ctx-lh-", "ctx-rh-"):
        if region_name.startswith(prefix):
            return region_name[len(prefix):]
    return None

def atlas_to_ggseg_subcortical(region_name):
    name = region_name
    for prefix in ("Left-", "Right-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = name.lower().replace("-", " ")
    name = name.replace("thalamus proper", "thalamus proper")
    name = name.replace("ventraldc", "ventral DC")
    name = name.replace("accumbens area", "accumbens area")
    return name

def prepare_csvs_for_r(value_series, saveas, tmp_dir):

    SUBCORT_MAP = {
        "Left-Thalamus-Proper": "Thalamus",
        "Right-Thalamus-Proper": "Thalamus",
        "Left-Caudate": "Caudate",
        "Right-Caudate": "Caudate",
        "Left-Putamen": "Putamen",
        "Right-Putamen": "Putamen",
        "Left-Pallidum": "Pallidum",
        "Right-Pallidum": "Pallidum",
        "Left-Hippocampus": "Hippocampus",
        "Right-Hippocampus": "Hippocampus",
        "Left-Amygdala": "Amygdala",
        "Right-Amygdala": "Amygdala",
        "Left-Accumbens-area": "accumbens area",
        "Right-Accumbens-area": "accumbens area",
        "Left-VentralDC": "ventraldc",
        "Right-VentralDC": "ventraldc",
        "Left-Cerebellum-Cortex": "Cerebellum",
        "Right-Cerebellum-Cortex": "Cerebellum",
        "Brain-Stem": "Brain Stem",
    }

    cx_lh, cx_rh = {}, {}
    aseg_l, aseg_r = {}, {}

    for region, val in value_series.items(): 

        if pd.isna(val):
            continue

        if region.startswith("ctx-lh-"):
            cx_lh[region[len("ctx-lh-"):]] = float(val)

        elif region.startswith("ctx-rh-"):
            cx_rh[region[len("ctx-rh-"):]] = float(val)

        elif region.startswith("Left-"):
            ggseg_name = SUBCORT_MAP.get(region)
            if ggseg_name:
                aseg_l[ggseg_name] = float(val)

        elif region.startswith("Right-"):
            ggseg_name = SUBCORT_MAP.get(region)
            if ggseg_name:
                aseg_r[ggseg_name] = float(val)

        elif region == "Brain-Stem":
            aseg_l["Brain Stem"] = float(val)
            aseg_r["Brain Stem"] = float(val)

    def _cx_df(d):
        df = pd.DataFrame({"r": pd.Series(d)})
        df.index.name = "region"
        return df
    _cx_df(cx_lh).to_csv(os.path.join(tmp_dir, f"{saveas}_cx_lh.csv"))
    _cx_df(cx_rh).to_csv(os.path.join(tmp_dir, f"{saveas}_cx_rh.csv"))

    def _aseg_df(d):
        df = pd.DataFrame({"r": pd.Series(d)})
        df.index.name = "region"
        return df
    _aseg_df(aseg_l).to_csv(os.path.join(tmp_dir, f"{saveas}_asegl.csv"))
    _aseg_df(aseg_r).to_csv(os.path.join(tmp_dir, f"{saveas}_asegr.csv"))

    vals = value_series.dropna().values
    vmin = float(np.nanpercentile(vals, 5))
    vmax = float(np.nanpercentile(vals, 95))

    return vmin, vmax

# STEP 3: CALL R TO RENDER PDFs

def run_r(saveas, tmp_dir, vmin, vmax, low=CMAP_LOW, high=CMAP_HIGH, atlas=ATLAS):
    cmd = [
        RSCRIPT, "--vanilla", RBRAIN_2C,
        saveas,
        tmp_dir + os.sep, 
        str(vmin),
        str(vmax),
        low,
        high,
        atlas,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"R stderr: {result.stderr[-800:]}")
        raise RuntimeError(f"Rscript failed for {saveas}")

# STEP 4: APPLY IMAGE UTILITIES

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 0.5, -100)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im

def pdf_to_img(pdf_path, dpi=150, crop_right=80):
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)
    img = pages[0]
    w, h = img.size
    img = img.crop((0, 0, w - crop_right, h))
    img = trim(img)
    img = ImageOps.expand(img, border=(40, 40, 40, 40), fill="white")
    return img

def append_images(images, direction="horizontal", bg_color=(255, 255, 255), alignment="center"):

    widths, heights = zip(*(i.size for i in images))
    if direction == "horizontal":
        new_w, new_h = sum(widths), max(heights)
    else:
        new_w, new_h = max(widths), sum(heights)

    canvas = Image.new("RGB", (new_w, new_h), color=bg_color)
    offset = 0
    for im in images:
        if direction == "horizontal":
            y = int((new_h - im.size[1]) / 2) if alignment == "center" else 0
            canvas.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = int((new_w - im.size[0]) / 2) if alignment == "center" else 0
            canvas.paste(im, (x, offset))
            offset += im.size[1]
            
    return canvas

def add_title(img, text, fontsize=150, space=220):

    img = ImageOps.expand(img, border=(0, space, 0, 0), fill="white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
    except OSError:
        try:
            font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", fontsize)
        except OSError:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    w, _ = img.size
    draw.text(((w - text_w) / 2, 10), text, fill=(0, 0, 0), font=font)

    return img

def make_colorbar(vmin, vmax, cbar_label, low=CMAP_LOW, high=CMAP_HIGH, figsize=(1.2, 5), dpi=150):

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_visible(False)

    cmap = LinearSegmentedColormap.from_list("custom", [low, high])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.3, 0.05, 0.25, 0.88])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=9, color="black")
    cbar.ax.tick_params(labelsize=8, colors="black")

    tmp = os.path.join(TMP_DIR, "_cbar_tmp.png")
    plt.savefig(tmp, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close("all")

    return Image.open(tmp).copy()

# STEP 5: ASSEMBLE CORE COMBINED FIGURE

def plot_combined_map(value_series, title, out_path, cbar_label="Value", dpi=DPI):
    
    saveas = title.replace(" ", "_").replace("\n", "_").replace("/", "-")
    for ch in r'\:*?"<>|':
        saveas = saveas.replace(ch, "")

    vmin, vmax = prepare_csvs_for_r(value_series, saveas, TMP_DIR)
    run_r(saveas, TMP_DIR, vmin, vmax)

    def cp(suffix):
        return os.path.join(TMP_DIR, f"{saveas}{suffix}")

    ll = pdf_to_img(cp("_cx_left_lateral.pdf"),  dpi=150)
    rl = pdf_to_img(cp("_cx_right_lateral.pdf"), dpi=150)
    lm = pdf_to_img(cp("_cx_left_medial.pdf"),   dpi=150)
    rm = pdf_to_img(cp("_cx_right_medial.pdf"),  dpi=150)

    top_h = max(ll.size[1], rl.size[1])
    bot_h = max(lm.size[1], rm.size[1])

    def resize_h(img, h):
        w = int(img.size[0] * h / img.size[1])
        return img.resize((w, h), Image.LANCZOS)

    ll = resize_h(ll, top_h)
    rl = resize_h(rl, top_h)
    lm = resize_h(lm, bot_h)
    rm = resize_h(rm, bot_h)

    top_row = append_images([ll, rl], "horizontal")
    bottom_row = append_images([lm, rm], "horizontal")

    max_w = max(top_row.size[0], bottom_row.size[0])
    def resize_w(img, w):
        h = int(img.size[1] * w / img.size[0])
        return img.resize((w, h), Image.LANCZOS)

    top_row = resize_w(top_row, max_w)
    bottom_row = resize_w(bottom_row, max_w)

    cortex_img = append_images([top_row, bottom_row], "vertical")
    cortex_img = add_title(cortex_img, "Neocortical", fontsize=90, space=200)

    cor_img = pdf_to_img(cp("_aseg_coronal.pdf"),  dpi=150)
    sag_img = pdf_to_img(cp("_aseg_sagittal.pdf"), dpi=150)

    sub_h = max(cor_img.size[1], sag_img.size[1])
    cor_img = resize_h(cor_img, sub_h)
    sag_img = resize_h(sag_img, sub_h)

    sub_combined = append_images([cor_img, sag_img], "horizontal")

    cort_w = cortex_img.size[0]
    if sub_combined.size[0] != cort_w:
        sub_combined = resize_w(sub_combined, cort_w)

    sub_combined = add_title(sub_combined, "Subcortical", fontsize=90, space=200)

    brain_panel = append_images([cortex_img, sub_combined], "vertical")

    cbar_img = make_colorbar(vmin, vmax, cbar_label, dpi=dpi)
    bw, bh = brain_panel.size
    cw, ch = cbar_img.size
    cbar_img = cbar_img.resize((int(cw * bh / ch), bh), Image.LANCZOS)

    final = append_images([brain_panel, cbar_img], "horizontal")
    final = add_title(final, title, fontsize=150, space=250)
    final.save(out_path, dpi=(dpi, dpi))

    for suffix in [
        "_cx_lh.csv", "_cx_rh.csv", "_asegl.csv", "_asegr.csv",
        "_cx_left_lateral.pdf",  "_cx_right_lateral.pdf",
        "_cx_left_medial.pdf",   "_cx_right_medial.pdf",
        "_aseg_coronal.pdf",     "_aseg_sagittal.pdf",
    ]:
        fpath = cp(suffix)
        if os.path.exists(fpath):
            os.remove(fpath)


# ─── INDIVIDUAL GENE EXPRESSION MAPS ───

def plot_gene_expression_maps(expr_df, output_dir):

    print("Rendering individual gene expression maps...")

    gene_dir = os.path.join(output_dir, "gene_maps")

    for gene in expr_df.columns:

        series = expr_df[gene].dropna()
        if len(series) == 0:
            continue

        print(f"{gene}...")

        plot_combined_map(
            value_series=series,
            title=gene,
            out_path=os.path.join(gene_dir, f"{gene}_expression.png"),
            cbar_label="Expression (SRS units)",
        )

    print(f"Saved to: {gene_dir}")


# ─── INDIVIDUAL NEURODEGENERATION SUVR MAPS (AMYLOID AND TAU) ───

def plot_neurodegeneration_maps(amyloid_df, tau_df, output_dir):

    print("Rendering individual neurodegeneration SUVR maps...")

    neuro_dir = os.path.join(output_dir, "neuro_maps")

    for modality_label, neuro_df in [("amyloid", amyloid_df),("tau", tau_df),]:

        if neuro_df is None:
            continue

        print(f"{modality_label.upper()}...")

        series = neuro_df["median_suvr"].dropna()

        plot_combined_map(
            value_series=series,
            title=f"{modality_label.capitalize()} SUVR",
            out_path=os.path.join(
                neuro_dir, f"{modality_label}_suvr.png"
            ),
            cbar_label="SUVR",
        )

    print(f"Saved to: {neuro_dir}")


# ─── COMPARISON PANELS FOR GENE-MODALITY PAIRS REACHING NOMINAL SIGFIFICANCE ───

def plot_comparison_panels(expr_df, amyloid_df, tau_df, corr_df, output_dir, p_threshold):

    print("Rendering comparison panels for gene-modality pairs...")

    comp_dir = os.path.join(output_dir, "comparison_panels")
    sig_pairs = corr_df[corr_df["spearman_p"] < p_threshold].copy()
    neuro_map = {"amyloid": amyloid_df, "tau": tau_df}

    for _, row in sig_pairs.iterrows():
        gene = row["gene"]
        modality = row["modality"]
        r = row["spearman_r"]
        p = row["spearman_p"]

        neuro_df = neuro_map.get(modality)
        if neuro_df is None or gene not in expr_df.columns:
            continue

        print(f"{gene} vs {modality} ρ={r:.3f} p={p:.4f}") 

        gene_series = expr_df[gene].dropna()
        suvr_series = neuro_df["median_suvr"].dropna()
        common = gene_series.index.intersection(suvr_series.index)

        tmp_gene = os.path.join(TMP_DIR, f"_panel_gene_{gene}.png")
        tmp_suvr = os.path.join(TMP_DIR, f"_panel_suvr_{modality}.png")

        plot_combined_map(
            gene_series.loc[common],
            "",
            tmp_gene,
            cbar_label="Expression (SRS units)",
        )
        plot_combined_map(
            suvr_series.loc[common],
            "",
            tmp_suvr,
            cbar_label="SUVR",
        )

        img_gene = Image.open(tmp_gene)
        img_suvr = Image.open(tmp_suvr)
        combined = append_images([img_gene, img_suvr], "horizontal")
        combined = add_title(
            combined,
            f"{gene} Expression vs {modality} SUVR - Spearman ρ={r:.3f} | p={p:.4f}",
            fontsize=150, space=80,
        )

        out_path = os.path.join(comp_dir, f"{gene}_{modality}_comparison.png")
        combined.save(out_path)

        os.remove(tmp_gene)
        os.remove(tmp_suvr)

    print(f"Saved to: {comp_dir}")


# ─── SUMMARY HEATMAP ───


def plot_summary_heatmap(expr_df, corr_df, output_dir, p_threshold):

    print("Computing a summary heatmap...")

    amyloid_rows = corr_df[corr_df["modality"] == "amyloid"]
    tau_rows = corr_df[corr_df["modality"] == "tau"]

    sig_amyloid = set(amyloid_rows[amyloid_rows["spearman_p"] < p_threshold]["gene"])
    sig_tau = set(tau_rows[tau_rows["spearman_p"] < p_threshold]["gene"])

    gene_order = (amyloid_rows.set_index("gene")["spearman_r"].reindex(expr_df.columns).fillna(0).sort_values(ascending=False).index.tolist())

    for g in expr_df.columns:
        if g not in gene_order:
            gene_order.append(g)

    region_order = (expr_df.mean(axis=1).sort_values(ascending=False).index.tolist())

    heatmap_data = expr_df.loc[region_order, gene_order]

    def shorten(name):
        return (name
                .replace("ctx-lh-", "lh-")
                .replace("ctx-rh-", "rh-")
                .replace("Left-", "L-")
                .replace("Right-", "R-"))

    display_regions = [shorten(r) for r in region_order]

    col_labels = []
    for gene in gene_order:
        label = gene
        if gene in sig_amyloid:
            label += " *A"
        if gene in sig_tau:
            label += " *T"
        col_labels.append(label)

    data_vals = heatmap_data.values[~np.isnan(heatmap_data.values)]
    vmin_h = float(np.nanpercentile(data_vals, 2))
    vmax_h = float(np.nanpercentile(data_vals, 98))

    fig_w = max(14, len(gene_order) * 0.75)
    fig_h = max(18, len(region_order) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        heatmap_data, ax=ax,
        cmap="Blues", vmin=vmin_h, vmax=vmax_h,
        xticklabels=col_labels, yticklabels=display_regions,
        linewidths=0.3, linecolor="#dce8f5",
        cbar_kws={
            "label":  "Expression (SRS units)",
            "shrink": 0.5,
            "aspect": 25,
        },
        annot=False,
    )

    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9, color="black")
    ax.set_yticklabels(display_regions, rotation=0, fontsize=7, color="black")
    ax.set_title(
        "AD-Associated Gene Expression Across Brain Regions\n"
        "Sorted by amyloid Spearman ρ | *A = p<0.05 amyloid | *T = p<0.05 tau",
        fontsize=12, fontweight="bold", color="black", pad=14,
    )
    ax.set_xlabel("AD-Associated Gene", fontsize=11, color="black")
    ax.set_ylabel("Brain Region", fontsize=11, color="black")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "summary_heatmap.png")
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved to: {out_path}")


# ─── MAIN ORCHESTRATOR ───

def main():

    print("Starting AD spatial visualization pipeline...")

    for rscript_path in [RBRAIN_2C, RBRAIN_3C]:
        if not os.path.exists(rscript_path):
            raise FileNotFoundError(f"ERROR: R script not found: {rscript_path}.")

    expr_df, amyloid_df, tau_df, corr_df = load_all_data()

    plot_gene_expression_maps(expr_df, OUTPUT_DIR)

    plot_neurodegeneration_maps(amyloid_df, tau_df, OUTPUT_DIR)

    plot_comparison_panels(expr_df, amyloid_df, tau_df, corr_df, OUTPUT_DIR, P_THRESHOLD,)

    plot_summary_heatmap(expr_df, corr_df, OUTPUT_DIR, P_THRESHOLD)

    print("Visualization complete.")
    print(f"Outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()