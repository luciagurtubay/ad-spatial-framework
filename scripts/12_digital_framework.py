# ─── IMPORTS ───

import streamlit as st
from pathlib import Path
import pandas as pd
import json
import streamlit.components.v1 as components


# ─── PAGE CONFIGURATION ───

st.set_page_config(
    page_title="AD Spatial Framework",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── PATH CONFIGURATION ───

ROOT = Path(r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics")

INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"

P = {
    "methodology": ROOT / "others" / "methodology.png",
    "prisma": ROOT / "others" / "prisma_flow_diagram.jpg",
    "priority_img": INPUTS / "ad_gene_prioritisation" / "AD_gene_prioritisation.png",
    "priority_csv": INPUTS / "ad_gene_prioritisation" / "AD_prioritised_genes.csv",
    "expr_csv": OUTPUTS / "gene_parcellation" / "group_region_gene.csv",
    "gene_maps": OUTPUTS / "visualizations" / "gene_maps",
    "comp_panels": OUTPUTS / "visualizations" / "comparison_panels",
    "amyloid_csv": OUTPUTS / "amyloid_tau_parcellation" / "group_region_amyloid.csv",
    "tau_csv": OUTPUTS / "amyloid_tau_parcellation" / "group_region_tau.csv",
    "amyloid_img": OUTPUTS / "visualizations" / "neuro_maps" / "amyloid_suvr.png",
    "tau_img": OUTPUTS / "visualizations" / "neuro_maps" / "tau_suvr.png",
    "corr_csv": OUTPUTS / "correlation_analysis" / "correlation_results.csv",
    "bar_amyloid": OUTPUTS / "correlation_analysis" / "correlation_barchart_amyloid.png",
    "bar_tau": OUTPUTS / "correlation_analysis" / "correlation_barchart_tau.png",
    "bar_combined": OUTPUTS / "correlation_analysis" / "correlation_barchart.png",
    "scatter_dir": OUTPUTS / "correlation_analysis" / "correlation_scatter_plots",
    "am_metrics": OUTPUTS / "prediction_model" / "amyloid" / "metrics.json",
    "am_feat_csv": OUTPUTS / "prediction_model" / "amyloid" / "feature_importance.csv",
    "am_pred_img": OUTPUTS / "prediction_model" / "amyloid" / "predicted_vs_actual.png",
    "am_feat_img": OUTPUTS / "prediction_model" / "amyloid" / "feature_importance.png",
    "tau_metrics": OUTPUTS / "prediction_model" / "tau" / "metrics.json",
    "tau_feat_csv": OUTPUTS / "prediction_model" / "tau" / "feature_importance.csv",
    "tau_pred_img": OUTPUTS / "prediction_model" / "tau" / "predicted_vs_actual.png",
    "tau_feat_img": OUTPUTS / "prediction_model" / "tau" / "feature_importance.png",
    "deusto_logo": ROOT / "others" / "universidad_Deusto_logo.png",
    "iucpq_logo": ROOT / "others" / "IUCPQ_logo.png",
}


# ─── STATIC DATA ───

GENES_21 = [
    "APOE","BIN1","PICALM","CLU","ABCA7","TREM2","SORL1","CD2AP",
    "PTK2B","EPHA1","HLA-DRB1","INPP5D","ECHDC3","ZCWPW1","CD33",
    "FERMT2","APH1B","SLC24A4","RIN3","ACE","ADAM10",
]

GENE_FN = {
    "APOE": "Lipoprotein transport and lipid metabolism; major genetic risk factor for late-onset AD",
    "BIN1": "Endocytic trafficking, membrane tubulation, and synaptic vesicle recycling",
    "PICALM": "Clathrin-mediated endocytosis and autophagy regulation",
    "CLU": "Extracellular chaperone involved in amyloid-beta clearance and lipid transport",
    "ABCA7": "ATP-binding cassette transporter involved in lipid efflux and phagocytosis",
    "TREM2": "Microglial receptor regulating neuroinflammation and amyloid plaque clearance",
    "SORL1": "Endosomal sorting receptor regulating APP processing and amyloid-beta production",
    "CD2AP": "Actin cytoskeleton organisation and endocytic trafficking",
    "PTK2B": "Calcium-regulated non-receptor tyrosine kinase involved in synaptic plasticity",
    "EPHA1": "Ephrin receptor involved in neuronal development and immune cell migration",
    "HLA-DRB1":"MHC class II molecule regulating the adaptive immune response",
    "INPP5D": "Phosphoinositide phosphatase regulating microglial activation and phagocytosis",
    "ECHDC3": "Enoyl-CoA hydratase involved in fatty acid metabolism",
    "ZCWPW1": "Histone reader involved in DNA mismatch repair and epigenetic regulation",
    "CD33": "Sialic acid-binding immunoglobulin receptor expressed in microglia",
    "FERMT2": "Integrin activator involved in cell adhesion and APP trafficking",
    "APH1B": "Component of the gamma-secretase complex involved in APP cleavage",
    "SLC24A4": "Calcium/potassium transporter expressed in neurons",
    "RIN3": "Rab GTPase effector involved in endosomal trafficking",
    "ACE": "Angiotensin-converting enzyme involved in blood pressure regulation and amyloid-beta degradation",
    "ADAM10": "Alpha-secretase responsible for non-amyloidogenic APP cleavage",
}

PAPERS = [
    {"author":"Seshadri et al. (2010)",
     "title":"Genome-wide analysis of genetic loci associated with Alzheimer disease",
     "doi":"10.1001/jama.2010.574","url":"https://doi.org/10.1001/jama.2010.574"},
    {"author":"Lambert et al. (2013)",
     "title":"Meta-analysis of 74,046 individuals identifies 11 new susceptibility loci for Alzheimer's disease",
     "doi":"10.1038/ng.2802","url":"https://doi.org/10.1038/ng.2802"},
    {"author":"Kunkle et al. (2019)",
     "title":"Genetic meta-analysis of diagnosed Alzheimer's disease identifies new risk loci and implicates Aβ, tau, immunity and lipid processing",
     "doi":"10.1038/s41588-019-0358-2","url":"https://doi.org/10.1038/s41588-019-0358-2"},
    {"author":"Jansen et al. (2019)",
     "title":"Genome-wide meta-analysis identifies new loci and functional pathways influencing Alzheimer's disease risk",
     "doi":"10.1038/s41588-018-0311-9","url":"https://doi.org/10.1038/s41588-018-0311-9"},
    {"author":"Schwartzentruber et al. (2021)",
     "title":"Genome-wide meta-analysis, fine-mapping and integrative prioritization implicate new Alzheimer's disease risk genes",
     "doi":"10.1038/s41588-020-00776-w","url":"https://doi.org/10.1038/s41588-020-00776-w"},
    {"author":"Wightman et al. (2021)",
     "title":"A genome-wide association study with 1,126,563 individuals identifies new risk loci for Alzheimer's disease",
     "doi":"10.1038/s41588-021-00921-z","url":"https://doi.org/10.1038/s41588-021-00921-z"},
    {"author":"Rajabli et al. (2023)",
     "title":"Multi-ancestry genome-wide meta-analysis of 56,241 individuals identifies known and novel cross-population and ancestry-specific associations",
     "doi":"10.1186/s13059-025-03564-z","url":"https://doi.org/10.1186/s13059-025-03564-z"},
]


# ─── HELPERS ───

def img_bytes(path):
    p = Path(path)
    return open(p,"rb").read() if p.exists() else None

def read_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None

def read_json(path):
    p = Path(path)
    return json.load(open(p)) if p.exists() else None

def show_img(path, caption="", use_cw=True):
    b = img_bytes(path)
    if b:
        st.image(b, caption=caption, use_container_width=use_cw)
    else:
        st.markdown(
            f'<div class="missing-box">Image not yet available:<br><code>{Path(path).name}</code></div>',
            unsafe_allow_html=True)

def dl_btn(df, fname, label="Download table"):
    st.download_button(label=label,
                       data=df.to_csv(index=False).encode(),
                       file_name=fname, mime="text/csv")

def fmt_p(v):
    try:
        f = float(v)
        return "~0" if f == 0 else f"{f:.2E}"
    except:
        return str(v)

def tech_expander(render_fn):
    with st.expander("Technical Information"):
        render_fn()

def modality_selector(key):
    sk = f"_mod_{key}"
    if sk not in st.session_state:
        st.session_state[sk] = "amyloid"
    st.markdown('<p class="mod-label">Select modality</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,1,5])
    with c1:
        if st.button("Amyloid", key=f"{key}_a", use_container_width=True):
            st.session_state[sk] = "amyloid"
    with c2:
        if st.button("Tau",     key=f"{key}_t", use_container_width=True):
            st.session_state[sk] = "tau"
    m = st.session_state[sk]
    return m


# ─── GLOBAL CSS ───

st.markdown("""
<style>

/* FONTS & BASE */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    color: #1a2332;
}

.stApp { background-color: #f5f8fc; }

/* LAYOUT CHROME (TOOLBAR, FOOTER, SIDEBAR TOGGLE) */

[data-testid="stToolbar"] button:first-child { display:none !important; }
footer { visibility:hidden; }

[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebarOpenButton"]     { display: none !important; }
button[kind="header"]                   { display: none !important; }

/* HEADER BAR */

[data-testid="stHeader"] { background-color: #1a4480 !important; }
.header-title {
    position: fixed;
    top: 0;
    right: 24px;
    height: 3.5rem;
    display: flex;
    align-items: center;
    color: #ffffff;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    z-index: 999990;
    pointer-events: none;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background-color: #2171b5 !important;
    padding-top: 10px;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 2px; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 7px 10px;
    border-radius: 5px;
    cursor: pointer;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stRadio label[data-checked="true"],
[data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ label {
    background-color: #ffffff !important;
    color: #1a4480 !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background-color: rgba(255,255,255,0.18) !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] input[type="radio"] { accent-color: #ffffff; }

/* INTERACTIVE ELEMENTS (BUTTONS, TOGGLES, LINKS) */

[data-testid="stToggle"] [data-checked="true"] > div,
input[type="checkbox"]:checked + div { background-color: #1a4480 !important; }
input[type="checkbox"] { accent-color: #1a4480 !important; }

.stButton > button {
    background-color: #1a4480 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 5px !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
}
.stButton > button:hover {
    background-color: #08306b !important;
    color: #ffffff !important;
}
.stDownloadButton > button {
    background-color: #2171b5 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 5px !important;
    font-size: 0.82rem !important;
}
.stDownloadButton > button:hover { background-color: #1a4480 !important; }

a { color: #2171b5 !important; }
.stAlert { border-color: #2171b5 !important; }
[data-testid="stNotification"] { border-color: #2171b5 !important; }

.main [data-testid="stRadio"] label,
.main [data-testid="stRadio"] label p,
[data-testid="stMain"] [data-testid="stRadio"] label,
[data-testid="stMain"] [data-testid="stRadio"] label p {
    color: #1a2332 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

/* TYPOGRAPHY (TITLES, HEADERS) */

.home-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #08306b;
    line-height: 1.45;
    margin-bottom: 1.4rem;
}
.sec-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: #08306b;
    border-bottom: 2px solid #2171b5;
    padding-bottom: 8px;
    margin-bottom: 1.2rem;
}
.sub-title {
    font-size: 1.0rem;
    font-weight: 700;
    color: #1a4480;
    margin-top: 1.4rem;
    margin-bottom: 0.4rem;
}
.mod-label {
    font-size:0.78rem; font-weight:700; color:#1a4480;
    text-transform:uppercase; letter-spacing:0.05em;
    margin: 0 0 5px 0;
}

/* CONTENT BOXES */

.blank-area {
    background: #eef4fb;
    border: 1.5px dashed #90bad8;
    border-radius: 5px;
    min-height: 52px;
    margin: 6px 0 16px 0;
}
.missing-box {
    background: #f0f4f8;
    border: 1px dashed #90bad8;
    border-radius: 5px;
    padding: 18px;
    font-size: 0.82rem;
    color: #6b7b8d;
    text-align: center;
}
.notice {
    background:#ddeaf8; border-left:3px solid #2171b5;
    border-radius:4px; padding:9px 13px;
    font-size:0.86rem; color:#1a2332; margin:8px 0 14px;
}
hr.div { border:none; border-top:1px solid #c6dbef; margin:20px 0; }

/* SPECIFIC COMPONENTS AND OTHERS */

.contact-card {
    background:#ffffff; border:1px solid #c6dbef;
    border-radius:8px; padding:24px 28px; max-width:480px;
}
.contact-card h3 { color:#08306b; font-size:1.3rem; margin-bottom:3px; }
.contact-card .aff { font-size:1rem; color:#6b7b8d; margin-bottom:16px; }
.contact-card .row { font-size:1rem; color:#1a2332; margin-bottom:9px; }
.contact-card a { color:#2171b5 !important; text-decoration:none; }

[data-testid="stExpander"] summary,
.streamlit-expanderHeader {
    background-color: #2171b5 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
}
[data-testid="stExpander"] summary:hover,
.streamlit-expanderHeader:hover {
    background-color: #1a4480 !important;
}
[data-testid="stExpander"] > div > div,
.streamlit-expanderContent {
    background-color: #ddeaf8 !important;
    border: 1px solid #90bad8 !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
    padding: 14px 18px !important;
}
[data-testid="stExpander"] p,
[data-testid="stExpander"] strong { color:#1a2332 !important; font-size:0.88rem; }
[data-testid="stExpander"] svg { fill:#ffffff !important; stroke:#ffffff !important; }

.prisma-container {
    display: flex;
    align-items: stretch;
    height: 100%;
}
.prisma-container img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    object-position: top !important;
}

.paper-card-inline {
    background: #ffffff;
    border: 1px solid #c6dbef;
    border-radius: 6px;
    padding: 7px 14px;
    margin-bottom: 5px;
}
.paper-card-inline .pa { font-weight:700; color:#08306b; font-size:0.84rem; }
.paper-card-inline a   { font-size:0.79rem; color:#2171b5 !important; text-decoration:none; }
.paper-card-inline a:hover { text-decoration:underline; }

[data-testid="stDataFrame"] { border:1px solid #c6dbef; border-radius:6px; }

[data-testid="stTab"] { color:#2171b5 !important; }

</style>
""", unsafe_allow_html=True)


# ─── TITLE ───

st.markdown('<div class="header-title">AD Spatial Framework</div>', unsafe_allow_html=True)


# ─── SIDEBAR ───

with st.sidebar:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    page = st.radio("nav", [
        "HOME",
        "AD GENE PRIORITISATION",
        "BRAIN EXPRESSION MAPS",
        "ASSOCIATION ANALYSES",
        "DISCUSSION",
        "CONTACT",
    ], label_visibility="collapsed")


# ─── HOME ───

if page == "HOME":

    st.markdown(
        '<div class="home-title">Design and Development of a Digital Framework for '
        'Spatial Association Analysis between Alzheimer\'s Disease-Related Gene '
        'Expression and Neuropathology</div>',
        unsafe_allow_html=True)

    st.markdown("""
<div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; 
            padding:16px 20px; margin:6px 0 16px 0;">
    <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
        Alzheimer's disease (AD) is a progressive neurodegenerative disorder characterized by cognitive 
        decline and represents the leading cause of dementia worldwide. Despite the identification of 
        numerous implicated loci through large-scale genome-wide association studies (GWAS), the 
        distribution of AD risk genes across the human brain and its relationship to regional 
        vulnerability remain insufficiently characterized.
    </p>
    <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
        This project designs and develops a digital framework for the spatial association between 
        AD-related gene expression and neuropathology. Transcriptomic data from the Allen Human Brain 
        Atlas (AHBA) is parcellated into the 81 grey matter regions of the Desikan–Killiany–Tourville 
        (DKT) atlas using the abagen toolbox, yielding a group-level expression profile for a prioritized 
        set of 21 AD-relevant genes derived through Fisher's combined probability method applied across 
        seven GWAS meta-analyses. In parallel, amyloid and tau PET imaging data from the Alzheimer's 
        Disease Neuroimaging Initiative (ADNI) is processed through DICOM to NIfTI conversion, affine 
        registration to MNI152 standard space, and DKT parcellation to produce group-level standardized 
        uptake value ratio (SUVR) maps for both modalities. Spatial Spearman correlation analysis and 
        machine learning prediction, combining Ridge regression and Gradient Boosting with leave-one-out 
        cross-validation (LOOCV), are subsequently applied to investigate the gene-neuropathology 
        correspondence.
    </p>
    <p style="color:#1a2332; font-size:1.00rem; margin:0; line-height:1.6;">
        All analytical findings are integrated into a user-friendly Streamlit interface, enabling the 
        biomedical research community to explore AD-linked spatial imaging transcriptomic patterns, 
        thereby supporting further advances in neurogenetics and computational neuroscience.
    </p>
</div>
""", unsafe_allow_html=True)

    show_img(P["methodology"])


# ─── AD GENE PRIORITISATION ───

elif page == "AD GENE PRIORITISATION":
    
    st.markdown('<div class="sec-title">AD Gene Prioritisation</div>', unsafe_allow_html=True)

    # SYSTEMATIC SEARCH 

    st.markdown('<div class="sub-title">Systematic Literature Search</div>', unsafe_allow_html=True)

    st.markdown("""
<div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; 
            padding:16px 20px; margin:6px 0 16px 0;">
    <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
        To identify the most comprehensive and complementary sources of genetic association evidence for 
        AD, a systematic literature search is conducted in accordance with PRISMA guidelines. The screening 
        and selection process, as illustrated in the flow diagram on the left, yields seven foundational 
        GWAS meta-analyses, whose direct access is provided through the reference panel on the right.
    </p>
</div>
""", unsafe_allow_html=True)

    col_p, col_r = st.columns([1, 1], gap="large")

    with col_p:
        show_img(P["prisma"])
    
    with col_r:
        papers_html = '''
                <div style="background:#ffffff; border:1px solid #c6dbef; border-radius:6px;
                            padding:12px 16px; height:100%; box-sizing:border-box;">
                    <div style="font-weight:700; color:#08306b; font-size:0.88rem;
                                margin-bottom:10px; padding-bottom:6px;
                                border-bottom:1px solid #c6dbef;">Reference Studies</div>
                    <table style="width:100%; border-collapse:collapse; font-size:0.82rem;">
                '''
        for paper in PAPERS:
            papers_html += (
                f'<tr style="vertical-align:top;">'
                f'<td style="color:#1a4480; font-weight:600; padding:4px 10px 4px 0;'
                f'white-space:nowrap; font-size:0.86rem;">{paper["author"]}</td>'
                f'<td style="padding:4px 0; color:#374151; font-size:0.76rem;">'
                f'<a href="{paper["url"]}" target="_blank" '
                f'style="color:#2171b5; text-decoration:none;">'
                f'DOI: {paper["doi"]} ↗</a></td>'
                f'</tr>'
            )
        papers_html += '</table></div>'
        st.markdown(papers_html, unsafe_allow_html=True)
    
    st.markdown('<hr class="div">', unsafe_allow_html=True)
    

    # GENE PRIORIZATION

    st.markdown('<div class="sub-title">Gene Prioritisation by Combined Evidence</div>', unsafe_allow_html=True)

    st.markdown("""
<div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; 
            padding:16px 20px; margin:6px 0 16px 0;">
    <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
            Fisher's combined probability method is applied to aggregate genetic association significance 
            across the seven selected studies, deriving a single ranked composite p-value per gene.
    </p>
</div>
""", unsafe_allow_html=True)
    
    col_txt, col_pl = st.columns([0.75, 0.75], gap="large")

    with col_txt:
        st.markdown("""
    <div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; 
                padding:16px 20px; margin:6px 0 16px 0;">
        <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
            The scatter plot illustrates the power-law distribution of cumulative evidence across genes, with 
            the prioritisation threshold set at the elbow of the curve beyond which a minor subset of genes 
            accounts for the majority of the total evidence. Only genes reaching this threshold and confirmed 
            by genome-wide significance (GWS) in at least two of the studies are retained.
        </p>
    </div>
    """, unsafe_allow_html=True)
        
    with col_pl:
        show_img(P["priority_img"])

    df_p = read_csv(P["priority_csv"])
    if df_p is not None:
        df_p = df_p[df_p["assigned_gene"].isin(GENES_21)].copy()
        df_p = df_p.sort_values("rank")
        df_p["Biological Function"] = df_p["assigned_gene"].map(GENE_FN)
        df_p["Fisher p-value"] = df_p["fisher_p"].apply(fmt_p)
        df_p = df_p.rename(columns={"assigned_gene":"Gene"})[
            ["Gene","Fisher p-value","Biological Function"]]
    else:
        df_p = pd.DataFrame([
            {"Gene":g,"Fisher p-value":"—","Biological Function":GENE_FN[g]}
            for g in GENES_21])

    st.markdown("""
<div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; 
            padding:16px 20px; margin:6px 0 16px 0;">
    <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
            The following table presents the final prioritised gene set, which includes the corresponding 
            Fisher p-values and biological functions, and can be downloaded using the button below it. 
    </p>
</div>
""", unsafe_allow_html=True)
    
    st.dataframe(df_p, use_container_width=True, hide_index=True, height=600)

    st.markdown('<hr class="div">', unsafe_allow_html=True)


# ─── BRAIN EXPRESSION MAPS ───

elif page == "BRAIN EXPRESSION MAPS":

    st.markdown('<div class="sec-title">Brain Expression Maps</div>', unsafe_allow_html=True)

    expr_df = read_csv(P["expr_csv"])
    amyloid_df = read_csv(P["amyloid_csv"])
    tau_df = read_csv(P["tau_csv"])

    # GENE EXPRESSION 

    st.markdown('<div class="sub-title">Gene Expression — Regional Map</div>', unsafe_allow_html=True)
    
    st.markdown("""
<div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; 
            padding:16px 20px; margin:6px 0 16px 0;">
    <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
        Regional gene expression profiles are derived from the AHBA and parcellated across the grey matter 
        anatomical structures of the DKT atlas using the abagen toolbox. All values represent group-level 
        SRS-normalised medians aggregated across six post-mortem donors.
    </p>
    <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
        Select a gene from the dropdown to explore its spatial expression pattern across brain regions both 
        quantitatively and visually, and download the complete group-level gene expression matrix using the 
        button below the table.
    </p>
</div>
""", unsafe_allow_html=True)

    sel_gene = st.selectbox("Select gene", GENES_21, key="gene_sel")

    col_gi, col_gt = st.columns([1, 1], gap="large")

    with col_gi:
        show_img(P["gene_maps"] / f"{sel_gene}_expression.png",
                 caption=f"{sel_gene} — SRS-normalised expression")
        
    with col_gt:
        if expr_df is not None:
            rc = expr_df.columns[0]
            if sel_gene in expr_df.columns:
                tbl = expr_df[[rc, sel_gene]].copy()
                tbl.columns = ["Region", "Expression (SRS)"]
                tbl["Expression (SRS)"] = tbl["Expression (SRS)"].round(4)
                tbl = tbl.sort_values("Expression (SRS)", ascending=False)
                st.dataframe(tbl, use_container_width=True, hide_index=True, height=420)
            else:
                st.info(f"{sel_gene} not found in expression matrix.")
            dl_btn(expr_df, "group_region_gene.csv", "Download group-level gene expression matrix")
        else:
            st.info("Expression data not available.")

    st.markdown('<hr class="div">', unsafe_allow_html=True)


    # NEURODEGENERATION 

    st.markdown('<div class="sub-title">Neuropathological Burden — Regional Map</div>', unsafe_allow_html=True)
    
    st.markdown("""
<div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; 
            padding:16px 20px; margin:6px 0 16px 0;">
    <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
        Regional amyloid and tau burden are derived from ADNI Fully Processed PET imaging data, parcellated 
        across the DKT atlas regions using the cerebellar cortex as the SUVR normalization reference. All 
        values represent group-level medians aggregated across subjects.
    </p>
    <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
        Select a modality, Amyloid or Tau, to explore the spatial distribution of neuropathological 
        vulnerability made available through automatically updated regional tables and brain maps. The overall 
        SUVR matrix can also be downloaded by clicking the corresponding button. 
    </p>
</div>
""", unsafe_allow_html=True)

    neuro_mod = modality_selector("neuro")
    n_img = P["tau_img"] if neuro_mod == "tau" else P["amyloid_img"]
    n_df = tau_df if neuro_mod == "tau" else amyloid_df
    n_csv = P["tau_csv"] if neuro_mod == "tau" else P["amyloid_csv"]
    n_lbl = neuro_mod.capitalize()

    col_ni, col_nt = st.columns([1, 1], gap="large")
    
    with col_ni:
        show_img(n_img, caption=f"{n_lbl} SUVR — group-level regional map")
    
    with col_nt:
        if n_df is not None:
            rc = n_df.columns[0]
            disp_suvr = n_df[[rc, "median_suvr"]].copy()
            disp_suvr.columns = ["Region", f"{n_lbl} SUVR (median)"]
            disp_suvr[f"{n_lbl} SUVR (median)"] = disp_suvr[f"{n_lbl} SUVR (median)"].round(4)
            disp_suvr = disp_suvr.sort_values(f"{n_lbl} SUVR (median)", ascending=False)
            st.dataframe(disp_suvr, use_container_width=True, hide_index=True, height=420)
            dl_btn(n_df, f"group_region_{neuro_mod}.csv",
                   f"Download group-level {n_lbl} SUVR matrix")
        else:
            st.info(f"{n_lbl} SUVR data not available.")

    st.markdown('<hr class="div">', unsafe_allow_html=True)


# ─── ASSOCIATION ANALYSES ───

elif page == "ASSOCIATION ANALYSES":

    st.markdown('<div class="sec-title">Association Analyses</div>', unsafe_allow_html=True)
    
    st.markdown("""
<div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; 
            padding:16px 20px; margin:6px 0 16px 0;">
    <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
        This section characterises the spatial relationship between AD-associated gene expression and 
        neuropathological burden across the DKT atlas brain regions through two complementary analytical 
        approaches.
    </p>
    <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
        Select an analysis to get started:
    </p>
    <ul style="color:#1a2332; font-size:0.95rem; line-height:1.8; margin:0; padding-left:20px;">
        <li><strong>Spearman Correlation:</strong> univariate exploration of individual gene–neuropathology pair spatial associations.</li>
        <li><strong>ML Prediction:</strong> multivariate prediction of neuropathological burden from the collective gene expression profile.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

    st.markdown('<p class="mod-label">Select Analysis</p>', unsafe_allow_html=True)
    assoc_section = st.radio(
        "section", ["Spearman Correlation", "ML Prediction"],
        horizontal=True, label_visibility="collapsed",
        key="assoc_section"
    )

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    corr_df = read_csv(P["corr_csv"])

    # SPEARMAN CORRELATION

    if assoc_section == "Spearman Correlation":

        st.markdown('<div class="sub-title">Spearman Correlation</div>', unsafe_allow_html=True)
        
        st.markdown("""
    <div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; 
                padding:16px 20px; margin:6px 0 16px 0;">
        <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
            Spearman correlation coefficients are computed between the regional expression profile of each 
            AD-associated gene and the regional neuropathological SUVR map, independently for amyloid and 
            tau. All findings are reported at nominal significance (p < 0.05, uncorrected) and are therefore 
            exploratory in nature.
        </p>
        <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
            Select a modality, Amyloid or Tau, to display its corresponding results.
        </p>
    </div>
    """, unsafe_allow_html=True)

        corr_mod = modality_selector("corr")
        mod_lbl  = corr_mod.capitalize()

        bar_path = P["bar_amyloid"] if corr_mod == "amyloid" else P["bar_tau"]
        if not bar_path.exists():
            bar_path = P["bar_combined"]

        st.markdown("""
    <div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; 
                padding:16px 20px; margin:6px 0 16px 0;">
        <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
            The table presents the complete Spearman correlation results for all genes under the selected 
            modality, ranked by evidence strength. The bar chart provides a visual summary of the spatial 
            association landscape across the full prioritised gene set, with nominally significant gene-
            modality pairs highlighted for subsequent investigations.
        </p>
        <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
            Download the complete correlation results using the button below the table.
        </p>
    </div>
    """, unsafe_allow_html=True)

        col_t, col_b = st.columns([1, 1], gap="large")

        with col_t:
            if corr_df is not None:
                filt = corr_df[corr_df["modality"] == corr_mod].copy()
                filt = filt.sort_values("spearman_r", ascending=False)
                filt["spearman_r"] = filt["spearman_r"].round(4)
                filt["spearman_p"] = filt["spearman_p"].apply(
                    lambda x: f"{float(x):.4f}" if float(x) >= 0.0001 else f"{float(x):.2E}")
                filt["Significant"] = filt["significant"].map(
                    lambda x: "Yes" if str(x).upper() in ["TRUE","1","YES"] else "No")
                disp = filt[["gene","spearman_r","spearman_p","Significant"]].rename(columns={
                    "gene":"Gene","spearman_r":"Spearman ρ","spearman_p":"p-value"})
                st.dataframe(disp, use_container_width=True, hide_index=True, height=420)
                dl_btn(corr_df, "correlation_results.csv", "Download correlation analysis results")
            else:
                st.info("Correlation results not available.")
        
        with col_b:
            show_img(bar_path, caption=f"Spearman ρ — {mod_lbl}")

        st.markdown('<hr class="div">', unsafe_allow_html=True)
        
        st.markdown("""
    <div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; 
                padding:16px 20px; margin:6px 0 16px 0;">
        <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
            For gene–modality pairs reaching nominal significance, a scatter plot illustrating the spatial 
            relationship between gene expression and neuropathological burden together with a side-by-side 
            brain map comparison panel are displayed. 
        </p>
        <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
            To do so, select a gene from the dropdown to generate the corresponding visualisations for the 
            active modality. 
        </p>
    </div>
    """, unsafe_allow_html=True)

        sc_gene = st.selectbox("Select gene", GENES_21, key="sc_gene_new")
        sc_path = P["scatter_dir"] / f"scatter_plot_{corr_mod}_{sc_gene}.png"
        cp_path = P["comp_panels"] / f"{sc_gene}_{corr_mod}_comparison.png"

        col_sc, col_cp = st.columns([0.85, 1.20], gap="large")
        
        with col_sc:
            st.markdown(f"**Scatter plot — {sc_gene} vs {mod_lbl}**")
            if sc_path.exists():
                show_img(sc_path, caption=f"{sc_gene} vs {mod_lbl} SUVR")
            else:
                st.markdown(
                    f'<div class="notice">No scatter plot available for '
                    f'<strong>{sc_gene}</strong> / <strong>{mod_lbl}</strong>. '
                    f'This gene–modality pair may not have reached nominal significance.</div>',
                    unsafe_allow_html=True)
        
        with col_cp:
            st.markdown(f"**Brain map comparison — {sc_gene} vs {mod_lbl}**")
            if cp_path.exists():
                show_img(cp_path, caption=f"{sc_gene} — {mod_lbl} comparison map")
            else:
                st.markdown(
                    f'<div class="notice">No comparison map available for '
                    f'<strong>{sc_gene}</strong> / <strong>{mod_lbl}</strong>.</div>',
                    unsafe_allow_html=True)

        st.markdown('<hr class="div">', unsafe_allow_html=True)


    # ML PREDICTION

    elif assoc_section == "ML Prediction":
        
        st.markdown('<div class="sub-title">Machine Learning Prediction</div>', unsafe_allow_html=True)
        
        st.markdown("""
    <div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; 
                padding:16px 20px; margin:6px 0 16px 0;">
        <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
            Two complementary ML models are trained to predict neuropathological burden from the collective 
            AD-associated gene expression profile across all the DKT atlas regions, evaluated through leave-
            one-out cross-validation (LOOCV). Ridge regression provides a regularised linear baseline with 
            directly interpretable gene-level coefficients, while Gradient Boosting captures non-linear gene 
            interactions that linear models cannot detect.
        </p>
        <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
            Select a modality, Amyloid or Tau, to display its corresponding results.
        </p>
    </div>
    """, unsafe_allow_html=True)
        
        pred_mod = modality_selector("pred")
        mod_lbl = pred_mod.capitalize()
        mk = "am" if pred_mod == "amyloid" else "tau"

        met = read_json(P[f"{mk}_metrics"])

        st.markdown("""
    <div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; 
                padding:16px 20px; margin:6px 0 16px 0;">
        <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
            The table summarises the LOOCV performance of both models under the selected modality, reporting 
            the results of R² and MAE metrics. The predicted versus actual scatter plot illustrates how closely 
            each model's regional predictions align with the actual SUVR values, allowing direct visual comparison 
            of linear versus non-linear model fit across the DKT brain regions.
        </p>
    </div>
    """, unsafe_allow_html=True)

        col_m, col_pi = st.columns([1, 2], gap="large")
        
        with col_m:
            if met:
                ridge = met.get("ridge", {})
                gb = met.get("gradient_boosting", {})
                best = met.get("best_model", "—")
                mdf = pd.DataFrame([
                    {"Model":"Ridge Regression",  "R²":round(ridge.get("r2",0),3), "MAE":round(ridge.get("mae",0),3)},
                    {"Model":"Gradient Boosting", "R²":round(gb.get("r2",0),3),    "MAE":round(gb.get("mae",0),3)},
                ])
                st.markdown(f"Best performing model: **{best.replace('_',' ').title()}**")
                st.dataframe(mdf, use_container_width=True, hide_index=True)
            else:
                st.info("Model metrics not available.")
        
        with col_pi:
            show_img(P[f"{mk}_pred_img"], caption=f"Predicted vs Actual SUVR — {mod_lbl}")

        st.markdown('<hr class="div">', unsafe_allow_html=True)

        st.markdown("""
    <div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; 
                padding:16px 20px; margin:6px 0 16px 0;">
        <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
            Gene-level feature importance rankings identify which genes contribute most to neuropathological 
            burden prediction in the multivariate context, complementing the univariate findings of the Spearman 
            correlation analysis. 
        </p>
        <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
            The table displays absolute coefficients and permutation importance scores for all genes under both 
            models. Alternatively, the bar chart exclusively displays the permutation importance rankings for 
            Gradient Boosting, as it is the best-performing model across both modalities.
        </p>
        <p style="color:#1a2332; font-size:1.00rem; margin-bottom:10px; line-height:1.6;">
            Download the complete feature importance results using the button below the table.
        </p>
    </div>
    """, unsafe_allow_html=True)

        col_fi, col_fp = st.columns([1, 1], gap="large")
        feat_df = read_csv(P[f"{mk}_feat_csv"])
        
        with col_fi:
            if feat_df is not None:
                st.dataframe(feat_df, use_container_width=True, hide_index=True, height=420)
                dl_btn(feat_df, f"feature_importance_{pred_mod}.csv", "Download ML prediction results")
            else:
                st.info("Feature importance data not available.")
        
        with col_fp:
            show_img(P[f"{mk}_feat_img"], caption=f"Permutation Importance (GB) — {mod_lbl}")

        st.markdown('<hr class="div">', unsafe_allow_html=True)


# ─── DISCUSSION ───

elif page == "DISCUSSION":

    st.markdown('<div class="sec-title">Discussion</div>', unsafe_allow_html=True)
    
    # LIMITATIONS

    st.markdown('<div class="sub-title">Limitations</div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; '
        'padding:16px 20px; margin:6px 0 16px 0;">'
        '<p style="color:#1a2332; font-size:1rem; margin-bottom:10px; line-height:1.6;">'
        'The limitations outlined below are inherent to the project design and the characteristics of the '
        'employed data resources, which are explicitly acknowledged throughout this work:'
        '</p>'
        '<p style="color:#1a2332; font-size:1rem; margin-bottom:4px; line-height:1.6;">'
        '<strong>AHBA dataset composition:</strong>'
        '</p>'
        '<ul style="color:#1a2332; font-size:1rem; line-height:1.8; margin:0 0 10px 0; padding-left:20px;">'
        '<li><strong>Small cohort:</strong> The limited sample size of six donors restricts the statistical generalizability of group-level regional gene expression profiles and prevents formal demographic stratification analyses.</li>'
        '<li><strong>Hemispheric coverage asymmetry:</strong> Four of the six donors provide predominantly left-hemisphere rather than right-hemisphere samples, resulting in reduced or incomplete spatial representation for those individuals.</li>'
        '<li><strong>Demographic imbalance:</strong> The donor cohort only contains a female participant and spans an age range of 24–57 years, thereby failing to capture sex-related differences and the older age groups most relevant to late-onset AD (LOAD).</li>'
        '<li><strong>Bulk transcriptomic resolution:</strong> Microarray-derived expression measurements reflect the aggregate transcriptional signal across all cell types within a tissue sample, limiting the ability to resolve specific expression patterns.</li>'
        '</ul>'
        '<p style="color:#1a2332; font-size:1rem; margin-bottom:10px; line-height:1.6;">'
        '<strong>Mixed ADNI diagnostic population:</strong> The group-level regional neurodegeneration matrices '
        'for amyloid and tau are computed across a mixed population of cognitively normal, mild cognitive '
        'impairment (MCI), and AD subjects. This population heterogeneity may attenuate SUVR contrasts '
        'relative to those observed in analyses restricted to clinically diagnosed AD cases.'
        '</p>'
        '<p style="color:#1a2332; font-size:1rem; margin:0; line-height:1.6;">'
        '<strong>Limited statistical power:</strong> With DKT atlas brain regions as the unit of analysis for '
        'spatial associations, the statistical power after multiple false discovery rate (FDR) testing '
        'correction is limited. Therefore, findings are reported at nominal significance and explicitly '
        'framed as exploratory.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )


    # FUTURE DIRECTIONS

    st.markdown('<div class="sub-title">Future Directions</div>', unsafe_allow_html=True)
    
    st.markdown("""
<div style="background:#eef4fb; border:1.5px solid #90bad8; border-radius:5px; 
            padding:16px 20px; margin:6px 0 16px 0;">
    <p style="color:#1a2332; font-size:1rem; margin-bottom:10px; line-height:1.6;">
        Future work centres on five directions. First, integrating the Seattle Alzheimer's Disease Brain 
        Cell Atlas (SEA-AD) single-nucleus atlas, or applying deconvolution methods like CIBERSORT or 
        MuSiC to existing AHBA data, would address the current sample size, demographic imbalance, and 
        bulk-resolution limitations and recover cell-type-specific signals. Second, stratifying ADNI 
        subjects by diagnostic group (cognitive normal, MCI, and AD) would yield higher-contrast 
        neuropathological maps, which could be further incorporated with longitudinal transcriptomic data
        to allow gene-neuropathology relationships to be tracked across disease stages. Third, extending 
        the gene prioritisation pipeline to the full Bellenguez et al. (2022) locus catalogue (75+ loci 
        versus the current 25) would broaden transcriptomic coverage and could reveal new spatial associations.
        Finally, formally integrating the framework's outputs as empirical constraints within the MEDICS 
        multiscale AD model would represent the most direct translational continuation of this work, enabling
        simulation of disease progression informed by real spatial data.
    </p>
</div>
""", unsafe_allow_html=True) 


# ─── CONTACT ───

elif page == "CONTACT":

    st.markdown('<div class="sec-title">Contact</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="contact-card">
  <h3 style="font-size:1.3rem;">Lucia Gurtubay Loyo</h3>
  <div class="aff" style="font-size:1rem;">Universidad de Deusto &amp; IUCPQ</div>
  <div class="row" style="font-size:1rem;"><a href="mailto:lucia.gurtubay@opendeusto.es">lucia.gurtubay@opendeusto.es</a></div>
  <div class="row" style="font-size:1rem;"><a href="https://www.linkedin.com/in/lucia-gurtubay-loyo/" target="_blank">LinkedIn Profile</a></div>
  <div class="row" style="font-size:1rem;"><a href="https://git.valeria.science/medics/markers/genetics.git" target="_blank">GitLab Repository</a></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    
    st.markdown(
        '<p style="font-size:1rem;color:#6b7b8d;margin-bottom:10px">Institutional Affiliations</p>',
        unsafe_allow_html=True)

    lc, rc, _ = st.columns([1, 1, 7])
    d = img_bytes(P["deusto_logo"])
    i = img_bytes(P["iucpq_logo"])
    
    with lc:
        if d:
            st.image(d, width=120)
        else:
            st.markdown("Universidad de Deusto")
    
    with rc:
        if i:
            st.image(i, width=120)
        else:
            st.markdown("IUCPQ")