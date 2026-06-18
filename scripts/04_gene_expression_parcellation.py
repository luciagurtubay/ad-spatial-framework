# ─── IMPORTS ───

import os
import time
import logging
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import argparse
import warnings
warnings.filterwarnings('ignore')


# ─── CONFIGURATION ───

CONFIG = {
    "BASE_DIR": r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics",
    "AHBA_DIR": r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\ahba",
    "AD_GENES_FILE": r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\ad_gene_prioritisation\AD_prioritised_genes.csv",
    "DKT_ATLAS_FILE": r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\atlas\DKT_MNI152_atlas_filtered.nii",
    "DKT_LABELS_FILE": r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\atlas\DKT_MNI152_atlas_label.csv",
    "OUTPUT_DIR": r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\outputs\gene_parcellation"
}


def get_paths(config):

    out = config["OUTPUT_DIR"]

    return {
        "individual_dir": os.path.join(out, "individual_region_gene"),
        "group_file": os.path.join(out, "group_region_gene.csv"),
        "validation_dir": os.path.join(out, "validation_results"),
    }


# ─── LOGGING ───

def setup_logging(output_dir):

    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger(__name__)


# ─── CACHE MANAGER ───

def cache_existance(paths):

    required = [
        paths["individual_dir"],
        paths["group_file"],
        paths["validation_dir"],
    ]

    return all(os.path.exists(p) for p in required)


def load_cached_outputs(paths, logger):

    logger.info("Loading cached outputs...")

    group_median = pd.read_csv(paths["group_file"], index_col=0)
    group_median.index.name = "region"

    logger.info("Group-level region gene expression loaded.")

    return group_median


# ─── GENE EXPRESSION PARCELLATION ───

# STEP 1: RUN ABAGEN

def run_abagen(config, paths, logger):

    import abagen

    logger.info("Running abagen...")

    os.makedirs(paths["individual_dir"], exist_ok=True)

    abagen_output = abagen.get_expression_data(
        atlas=config["DKT_ATLAS_FILE"],
        probe_selection="rnaseq",       
        lr_mirror="bidirectional",     
        sample_norm="srs",             
        gene_norm=None,                 
        region_agg="donors",          
        agg_metric="median",           
        return_donors=True,
        data_dir=config["AHBA_DIR"]
    )

    if not isinstance(abagen_output, dict):
        logger.error("abagen did not return a dictionary per donor.")
        raise ValueError("Unexpected abagen output.")

    logger.info(f"abagen returned a dictionary per donor: {list(abagen_output.keys())}")

    ad_genes_df = pd.read_csv(config["AD_GENES_FILE"])
    ad_genes = ad_genes_df["assigned_gene"].astype(str).unique().tolist()

    atlas_labels_df = pd.read_csv(config["DKT_LABELS_FILE"])
    atlas_labels_df["id"] = atlas_labels_df["id"].astype(int)
    id_to_name = dict(zip(atlas_labels_df["id"], atlas_labels_df["structure"]))

    donor_data = {}

    for donor_id, df in abagen_output.items():

        df.index = df.index.map(id_to_name)
        df.index.name = "region"

        available_genes = [g for g in ad_genes if g in df.columns]
        df_filtered = df[available_genes]

        out_path = os.path.join(paths["individual_dir"], f"{donor_id}_region_gene.csv")
        df_filtered.to_csv(out_path)
        donor_data[donor_id] = df_filtered

    logger.info(f"Individual-level region gene experession saved to: {paths['individual_dir']}")

    return donor_data


# STEP 2: LOAD INDIVIDUAL-LEVEL REGION GENE EXPRESSION PROFILES 

def load_individual_region_gene_expression(paths, logger):

    logger.info("Loading individual-level region gene experession...")

    donor_data = {}

    individual_dir = paths["individual_dir"]

    if not os.path.exists(individual_dir):
        raise FileNotFoundError(f"Individual directory not found: {individual_dir}. Run the pipeline once with --rerun.")

    for file_name in sorted(os.listdir(individual_dir)):

        if file_name.endswith("_region_gene.csv"):
            donor_id = file_name.replace("_region_gene.csv", "")

            individual_df = pd.read_csv(os.path.join(individual_dir, file_name), index_col=0)
            individual_df.index.name = "region"

            donor_data[donor_id] = individual_df

            logger.info(f"Loaded donor {donor_id}.")

    logger.info(f"Total donors loaded: {len(donor_data)}.")

    return donor_data


# STEP 3: GROUP-LEVEL AGGREGATION

def compute_group_region_gene_expression(donor_data, paths, logger):

    logger.info("Computing group-level region gene expression...")  

    all_dfs = list(donor_data.values())

    common_regions = all_dfs[0].index
    common_genes = all_dfs[0].columns

    for individual_df in all_dfs[1:]:
        common_regions = common_regions.intersection(individual_df.index)
        common_genes = common_genes.intersection(individual_df.columns)

    logger.info(
        f"Common regions: {len(common_regions)}, "
        f"common genes: {len(common_genes)}."
    )

    stacked = np.stack([individual_df.loc[common_regions, common_genes].values for individual_df in all_dfs], axis=0)
    group_median_values = np.nanmedian(stacked, axis=0)

    group_df = pd.DataFrame(
        group_median_values,
        index=common_regions,
        columns=common_genes
    )
    group_df.index.name = "region"

    group_df.to_csv(paths["group_file"])
    logger.info(f"Group.level region gene experession saved to: {paths['group_file']}")
    
    return group_df


# STEP 4: VALIDATION (SPEARMAN CORRELATION)

def implement_validation(donor_data, group_median, paths, logger):

    logger.info("Running LOO Spearman correlation...")

    os.makedirs(paths["validation_dir"], exist_ok=True)

    all_ids = list(donor_data.keys())

    loo_results = {}

    for test_id in all_ids:

        other_dfs = [donor_data[d] for d in all_ids if d != test_id]

        common_regions = other_dfs[0].index
        common_genes   = other_dfs[0].columns

        for df in other_dfs[1:]:
            common_regions = common_regions.intersection(df.index)
            common_genes = common_genes.intersection(df.columns)

        loo_stacked = np.stack([df.loc[common_regions, common_genes].values for df in other_dfs], axis=0)
        loo_group_median_values = np.nanmedian(loo_stacked, axis=0)
        loo_df = pd.DataFrame(loo_group_median_values, index=common_regions, columns=common_genes)

        test_df = donor_data[test_id]
        test_common_regions = test_df.index.intersection(common_regions)
        test_common_genes = test_df.columns.intersection(common_genes)

        test_vec = test_df.loc[test_common_regions, test_common_genes].values.flatten()
        loo_group_vec = loo_df.loc[test_common_regions, test_common_genes].values.flatten()

        mask = ~np.isnan(test_vec) & ~np.isnan(loo_group_vec)
        corr, pval = spearmanr(test_vec[mask], loo_group_vec[mask])

        loo_results[test_id] = {
            "Spearman_rho": round(corr, 4),
            "p_value": float(pval)
        }

        logger.info(f"Donor {test_id} (LOO): rho = {corr:.4f}, p = {pval:.2e}")

    loo_out = pd.DataFrame.from_dict(loo_results, orient="index")
    loo_out.index.name = "donor_id"

    loo_path = os.path.join(paths["validation_dir"], "donor_LOO_group_spearman.csv")
    loo_out.to_csv(loo_path)
    logger.info(f"LOO results saved to: {loo_path}")

    logger.info("Running donor vs group Spearman correlation...")

    donor_vs_group_results = {}

    common_regions = group_median.index
    common_genes = group_median.columns

    for donor_id, donor_df in donor_data.items():

        shared_regions = donor_df.index.intersection(common_regions)
        shared_genes = donor_df.columns.intersection(common_genes)

        donor_vec = donor_df.loc[shared_regions, shared_genes].values.flatten()
        group_vec = group_median.loc[shared_regions, shared_genes].values.flatten()

        mask = ~np.isnan(donor_vec) & ~np.isnan(group_vec)
        corr, pval = spearmanr(donor_vec[mask], group_vec[mask])

        donor_vs_group_results[donor_id] = {
            "Spearman_rho": round(corr, 4),
            "p_value": float(pval),
        }

        logger.info(f"Donor {donor_id} vs group: rho = {corr:.4f}, p = {pval:.2e}")

    donor_vs_group_out = pd.DataFrame.from_dict(donor_vs_group_results, orient="index")
    donor_vs_group_out.index.name = "donor_id"

    donor_vs_group_path = os.path.join(paths["validation_dir"], "donor_vs_group_spearman.csv")
    donor_vs_group_out.to_csv(donor_vs_group_path)
    logger.info(f"Donor vs group results saved: {donor_vs_group_path}")

    return loo_out, donor_vs_group_out


# ─── MAIN ORCHESTRATOR ───

def run_pipeline(force_rerun=False):

    paths = get_paths(CONFIG)
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    logger = setup_logging(CONFIG["OUTPUT_DIR"])

    t0 = time.time()

    use_cache = (not force_rerun) and cache_existance(paths)

    if use_cache:

        logger.info("Cached outputs found - loading from disk.")
        logger.info("Pass --rerun to force full reprocessing.")

        donor_output = load_individual_region_gene_expression(paths, logger)
        group_output = load_cached_outputs(paths, logger)
        
    else:

        if force_rerun:
            logger.info("--rerun flag set: reprocessing all steps.")
        else:
            logger.info("No cached outputs found - running full pipeline.")

        donor_output = run_abagen(CONFIG, paths, logger)
        group_output = compute_group_region_gene_expression(donor_output, paths, logger)
        implement_validation(donor_output, group_output, paths, logger)

    elapsed = time.time() - t0
    logger.info(f"Pipeline finished. Total time: {elapsed / 60:.1f} minutes.")

    return donor_output, group_output


# ─── COMMAND-LINE INTERFACE ───

def main():

    parser = argparse.ArgumentParser(
        description="AD Gene Expression Parcellation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            python pipeline.py: Run full pipeline or load from cache if available.
            python pipeline.py --rerun: Force rerun all steps, ignoring any cached outputs.
        """
    )

    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force rerun all steps even if cached outputs exist."
    )

    args = parser.parse_args()
    run_pipeline(force_rerun=args.rerun)

if __name__ == "__main__":
    main()