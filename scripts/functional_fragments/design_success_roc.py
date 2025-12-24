import argparse
import json
import multiprocessing as mp
import os
import typing as t
from functools import partial
from multiprocessing.pool import ThreadPool  # Import ThreadPool
from pathlib import Path
from typing import List, Tuple
import ampal
from tessera.difference_fn.difference_processing import select_first_ampal_assembly
from tessera.difference_fn.shape_difference import RmsdBiopythonStrategy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from sklearn.metrics import hamming_loss
from tqdm import tqdm

from tessera.fragments.fragments_graph import StructureFragmentGraph
from tessera.function_prediction.uniprot_processing import Ontology
from tessera.scripts.functional_fragments.foldseek_check_proteins import (
    fetch_go_terms,
    get_uniprot_from_pdb,
)
from multiprocessing import Pool

from tessera.visualization.fold_coverage import load_graph_creator
import gmatch4py as gm

# Mapping from Category components to binding column names
category_to_binding = {
    "DNA": "DNA binding",
    "RNA": "RNA binding",
    "GTP": "GTP binding",
    "ATP": "ATP binding",
    "Metal": "metal ion binding",
}


def add_go_terms(df: pd.DataFrame, ontology: Ontology) -> pd.DataFrame:
    uniprot_ids = df["Uniprot"].tolist()

    # ThreadPool to avoid daemonic process restriction
    with ThreadPool(mp.cpu_count()) as pool:
        go_terms_list = list(
            tqdm(
                pool.imap(partial(fetch_go_terms, ontology=ontology), uniprot_ids),
                total=len(uniprot_ids),
                desc="Fetching GO terms",
            )
        )

    go_terms_df = pd.DataFrame(go_terms_list)
    df = pd.concat([df, go_terms_df], axis=1)
    return df


def create_reference_category_dict(
    reference_path: Path,
) -> t.Dict[str, t.Dict[str, t.Dict[str, t.Any]]]:
    assert reference_path.exists(), f"Input file {reference_path} does not exist."

    # Extract all the .pdb1 files from the reference_path
    reference_pdbs = list(reference_path.glob("*.pdb1"))

    category_to_pdb_to_json_path = {}

    # For each of the PDB, load the json results
    for ref_pdb in reference_pdbs:
        ref_pdb_name = ref_pdb.stem
        # Split by "_" and extract the second element
        pdb, ref_category = ref_pdb_name.split("_")
        results_pdb_name = f"{ref_pdb_name}_results.json"
        results_pdb_path = reference_path / results_pdb_name

        # Check if the results file exists
        assert (
            results_pdb_path.exists()
        ), f"Results file {results_pdb_path} does not exist."

        # Update the dictionary structure
        if ref_category not in category_to_pdb_to_json_path:
            category_to_pdb_to_json_path[ref_category] = {}

        category_to_pdb_to_json_path[ref_category][pdb] = {
            "reference": results_pdb_path,
            "designs": {},
        }

    return category_to_pdb_to_json_path


def add_designs_to_category_dict(
    category_dict: t.Dict[str, t.Dict[str, t.Dict[str, t.Any]]],
    design_paths: t.List[Path],
):
    for design_path in design_paths:
        folder_name = (
            design_path.stem
        )  # Extract the folder name, e.g., fragment_design_experiment_2
        design_files = list(design_path.glob("*_design_*_results.json"))

        for design_file in design_files:
            design_pdb_name = design_file.stem
            pdb, ref_category, _, _, _, _ = design_pdb_name.split("_")

            if ref_category in category_dict and pdb in category_dict[ref_category]:
                # Add the design under the appropriate category and pdb
                if folder_name not in category_dict[ref_category][pdb]["designs"]:
                    category_dict[ref_category][pdb]["designs"][folder_name] = []
                category_dict[ref_category][pdb]["designs"][folder_name].append(
                    design_file
                )
            else:
                raise KeyError(
                    f"PDB {pdb} with category {ref_category} not found in reference dictionary."
                )


def parse_results(
    results: dict,
    category: str,
    pdb_file_path: Path,
    alignment_slice: t.Union[slice, t.Tuple[int, int]],
) -> pd.DataFrame:
    """
    Parse the results from the API and extract relevant information, including the category.
    """
    data = []
    split_design_path = pdb_file_path.stem.split("_")
    design_id = split_design_path[0]
    if "design" in pdb_file_path.stem:
        design_n = split_design_path[-2]
    else:
        design_n = "0"

    for result in results["results"]:
        db = result["db"]
        if result["alignments"]:
            for i, alignment in enumerate(result["alignments"][0][alignment_slice]):
                target = alignment["target"]
                prob = alignment["prob"]
                score = alignment["score"]

                if db == "afdb-swissprot":
                    uniprot, rest = target.split("-")[1], target.split("_")[1]
                    protein_name = rest.split(" ", 1)[1].rsplit(" ", 1)[0]
                    id_ = rest.split(" ")[-1]
                elif db == "pdb100":
                    parts = target.split("_")
                    pdb_id = parts[0].split("-")[0]
                    chain = parts[1][0]
                    id_ = pdb_id + "_" + chain
                    protein_name = target.split(" ", 1)[1]
                    pdb_directory = pdb_file_path.parent
                    uniprot = get_uniprot_from_pdb(pdb_id, chain, pdb_directory)

                data.append(
                    [
                        design_id,
                        design_n,
                        category,
                        db,
                        uniprot,
                        protein_name,
                        id_,
                        prob,
                        score,
                    ]
                )

    df = pd.DataFrame(
        data,
        columns=[
            "Design ID",
            "Experiment Number",
            "Category",
            "DB",
            "Uniprot",
            "Protein Name",
            "ID",
            "Probability",
            "Score",
        ],
    )
    return df


def parse_reference(category, pdb, paths, num_results):
    with open(paths["reference"], "r") as ref_file:
        ref_results = json.load(ref_file)
        return parse_results(
            ref_results, category, paths["reference"], slice(0, num_results)
        )


def parse_design(category, pdb, design_name, design_files, num_results):
    design_data = []
    for design_file in design_files:
        with open(design_file, "r") as des_file:
            des_results = json.load(des_file)
            design_df = parse_results(
                des_results, category, design_file, slice(0, num_results)
            )
            design_data.append(design_df)
    return pd.concat(design_data, ignore_index=True)


# Move starmap_helper function out of parse_all_results
def starmap_helper(func, *args):
    return func(*args)


def parse_all_results(
    category_dict: t.Dict[str, t.Dict[str, t.Dict[str, t.Any]]],
    num_results: int,
    ontology: Ontology,
) -> t.Dict[str, pd.DataFrame]:
    reference_jobs = [
        (category, pdb, paths, num_results)
        for category, pdb_dict in category_dict.items()
        for pdb, paths in pdb_dict.items()
    ]

    design_jobs = [
        (category, pdb, design_name, design_files, num_results)
        for category, pdb_dict in category_dict.items()
        for pdb, paths in pdb_dict.items()
        for design_name, design_files in paths["designs"].items()
    ]

    with mp.Pool(mp.cpu_count()) as pool:
        # Process reference jobs
        reference_data = list(
            tqdm(
                pool.starmap(
                    starmap_helper, [(parse_reference, *job) for job in reference_jobs]
                ),
                total=len(reference_jobs),
                desc="Parsing reference results",
            )
        )

        # Process design jobs
        design_data = list(
            tqdm(
                pool.starmap(
                    starmap_helper, [(parse_design, *job) for job in design_jobs]
                ),
                total=len(design_jobs),
                desc="Parsing design results",
            )
        )

    # Combine all reference data into a single DataFrame
    reference_df_combined = pd.concat(reference_data, ignore_index=True)
    reference_df_combined = add_go_terms(reference_df_combined, ontology)

    # Combine all design data into separate DataFrames per design path
    design_dfs_combined = {}
    for design_job, design_df in zip(design_jobs, design_data):
        _, _, design_name, _, _ = design_job
        if design_name not in design_dfs_combined:
            design_dfs_combined[design_name] = []
        design_dfs_combined[design_name].append(design_df)

    # Using ThreadPool to add GO terms to design DataFrames
    with ThreadPool(mp.cpu_count()) as pool:
        design_dfs_combined = {
            design_name: df
            for design_name, df in zip(
                design_dfs_combined.keys(),
                tqdm(
                    pool.imap(
                        partial(add_go_terms, ontology=ontology),
                        [
                            pd.concat(design_list, ignore_index=True)
                            for design_list in design_dfs_combined.values()
                        ],
                    ),
                    total=len(design_dfs_combined),
                    desc="Adding GO terms to design DataFrames",
                ),
            )
        }

    return {"reference": reference_df_combined, **design_dfs_combined}


def check_paths_exist(paths: t.List[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Error: The file or directory '{path}' does not exist."
            )


def load_or_parse_results(
    category_dict: t.Dict, num_results: int, ontology: Ontology
) -> t.Dict[str, pd.DataFrame]:
    result_dfs = {}
    # Load or parse reference results
    if Path("reference_results.csv").exists():
        result_dfs["reference"] = pd.read_csv("reference_results.csv")
    else:
        result_dfs = parse_all_results(category_dict, num_results, ontology)
        result_dfs["reference"].to_csv("reference_results.csv", index=False)

    return result_dfs


def save_experiment_results(
    experiment_name: str, result_dfs: t.Dict[str, pd.DataFrame]
) -> None:
    if Path(f"{experiment_name}_results.csv").exists():
        result_dfs[experiment_name] = pd.read_csv(f"{experiment_name}_results.csv")
    else:
        result_dfs[experiment_name].to_csv(
            f"{experiment_name}_results.csv", index=False
        )


def merge_experiment_dfs(experiments_to_dfs: t.Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged_df = pd.DataFrame()  # Initialize empty dataframe
    for experiment_name in experiments_to_dfs:
        if experiment_name in experiments_to_dfs:
            df = experiments_to_dfs[
                experiment_name
            ].copy()  # Ensure no in-place modification
            df["Experiment"] = experiment_name  # Add the 'Experiment' column
            merged_df = pd.concat([merged_df, df], ignore_index=True)
    return merged_df


def check_success(row: pd.Series) -> bool:
    # Split the Category by '+' and strip any whitespace
    categories = [cat.strip() for cat in row["Category"].split("+")]
    for cat in categories:
        binding_col = category_to_binding.get(cat)
        if binding_col is None:
            # If the category component is unknown, consider it as a failure
            return False
        if not row.get(binding_col, False):
            # If the required binding is not True, mark as failure
            return False
    return True


def add_success_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'success' column to the DataFrame based on the 'Category' and binding columns.
    A row is marked as successful if all binding columns corresponding to its Category are True.
    """

    # Apply the check_success function to each row to determine 'success'
    df["success"] = df.apply(check_success, axis=1)
    return df


def adjust_lightness(color, amount=1.0):
    """
    Adjust the lightness of a color in HLS space.
    amount > 1 makes the color lighter
    amount < 1 makes the color darker
    """
    import colorsys

    c = np.array(to_rgb(color))
    h, l, s = colorsys.rgb_to_hls(*c)
    l = max(0, min(1, l * amount))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return r, g, b


def compute_success_rates(top_df, min_df, min_count):
    # Group by Experiment
    grouped_top_10 = top_df.groupby("Experiment")
    grouped_min = min_df.groupby("Experiment")

    # Total designs per experiment in top 10 and top min
    total_top_10 = grouped_top_10.size()
    total_min = grouped_min.size()

    # Successful designs in top 10 and top min
    success_top_10 = grouped_top_10["success"].sum()
    success_min = grouped_min["success"].sum()

    # Calculate success rates
    success_rate_top_10 = (success_top_10 / 10) * 100
    success_rate_min = (success_min / min_count) * 100

    # Combine into DataFrame
    success_rates = pd.DataFrame(
        {
            "Success Rate @10 (%)": success_rate_top_10,
            f"Success Rate @{min_count} (%)": success_rate_min,
        }
    )

    # Reset index to have 'Experiment' as a column
    success_rates = success_rates.reset_index()

    return success_rates


def plot_cumulative_graphs_per_pdb(
    df: pd.DataFrame, output_dir: str = "output_plots"
) -> None:
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of all unique experiments across the entire DataFrame
    all_experiments = df["Experiment"].unique()
    n_experiments = len(all_experiments)

    # Create a consistent color mapping for experiments
    base_colors = sns.color_palette("colorblind", n_colors=n_experiments)
    experiment_color_mapping = dict(zip(all_experiments, base_colors))

    # Get unique PDB codes and loop over each
    for pdb_code in df["Design ID"].unique():
        # Filter for the current PDB code
        pdb100_df = df[(df["Design ID"] == pdb_code) & (df["DB"] == "pdb100")]
        afdb_df = df[(df["Design ID"] == pdb_code) & (df["DB"] == "afdb-swissprot")]

        # Skip if there's no data for this PDB code
        if pdb100_df.empty or afdb_df.empty:
            print(f"No data for PDB {pdb_code}. Skipping.")
            continue

        category = pdb100_df["Category"].iloc[
            0
        ]  # Assume the first category is representative

        # Sort by Score in descending order
        pdb100_df = pdb100_df.sort_values(by="Score", ascending=False)
        afdb_df = afdb_df.sort_values(by="Score", ascending=False)

        # Determine the minimum number of results to select
        min_pdb100 = min(100, pdb100_df.groupby("Experiment").size().min())
        min_afdb = min(100, afdb_df.groupby("Experiment").size().min())

        # Select the top 10 and the top min rows
        top_10_pdb100_df = pdb100_df.groupby("Experiment").head(10).copy()
        min_pdb100_df = pdb100_df.groupby("Experiment").head(min_pdb100).copy()
        top_10_afdb_df = afdb_df.groupby("Experiment").head(10).copy()
        min_afdb_df = afdb_df.groupby("Experiment").head(min_afdb).copy()

        # Compute success rates for pdb100
        success_rates_pdb100 = compute_success_rates(
            top_10_pdb100_df, min_pdb100_df, min_pdb100
        )

        # Compute success rates for afdb
        success_rates_afdb = compute_success_rates(
            top_10_afdb_df, min_afdb_df, min_afdb
        )

        # Filter only successful designs for plotting cumulative distributions
        min_pdb100_df_success = min_pdb100_df[min_pdb100_df["success"]]
        min_afdb_df_success = min_afdb_df[min_afdb_df["success"]]

        # Skip if no successful entries after filtering
        if min_pdb100_df_success.empty and min_afdb_df_success.empty:
            print(
                f"No successful entries for PDB {pdb_code} in category {category}. Skipping."
            )
            continue

        # Prepare data for cumulative counts
        def prepare_cumulative_data(df):
            df["cumulative_count_all"] = df.groupby("Experiment")["Score"].rank(
                method="max", ascending=True
            )
            df["cumulative_count_all"] = df.groupby("Experiment")[
                "cumulative_count_all"
            ].transform(lambda x: (x / x.max()) * 100)
            return df

        min_pdb100_df_success = prepare_cumulative_data(min_pdb100_df_success)
        min_afdb_df_success = prepare_cumulative_data(min_afdb_df_success)

        # Set up the figure and axes
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Cumulative Distribution (pdb100)
        sns.lineplot(
            data=min_pdb100_df_success,
            x="Score",
            y="cumulative_count_all",
            ax=axes[0, 0],
            hue="Experiment",
            palette=experiment_color_mapping,
            legend=False,
        )
        axes[0, 0].set_title(
            f"Cumulative Distribution of Successful Designs (pdb100)\nPDB: {pdb_code}, Category: {category}, N={min_pdb100}"
        )
        axes[0, 0].set_xlabel("Score")
        axes[0, 0].set_ylabel("Cumulative Percentage (%)")
        axes[0, 0].set_xlim(0, 100)

        # Plot 2: Success Rates (pdb100) with adjusted colors
        success_rates = success_rates_pdb100.set_index("Experiment")
        experiments = all_experiments  # Use the consistent experiment list
        n_exp = len(experiments)
        indices = np.arange(n_exp)
        bar_width = 0.35

        for i, experiment in enumerate(experiments):
            base_color = experiment_color_mapping.get(
                experiment, (0.7, 0.7, 0.7)
            )  # Default to gray if missing

            # Adjust colors: lighter for Success Rate @10, darker for Success Rate @N
            color_at_10 = adjust_lightness(base_color, 1.2)  # Lighter color
            color_at_N = adjust_lightness(base_color, 0.8)  # Darker color

            rate_at_10 = (
                success_rates.loc[experiment, "Success Rate @10 (%)"]
                if experiment in success_rates.index
                else 0
            )
            rate_at_N = (
                success_rates.loc[experiment, f"Success Rate @{min_pdb100} (%)"]
                if experiment in success_rates.index
                else 0
            )

            # Plot Success Rate @10 (lighter bar) first
            axes[0, 1].bar(
                indices[i] - bar_width / 2,
                rate_at_10,
                width=bar_width,
                color=color_at_10,
            )
            axes[0, 1].bar(
                indices[i] + bar_width / 2, rate_at_N, width=bar_width, color=color_at_N
            )

        axes[0, 1].set_xticks(indices)
        axes[0, 1].set_xticklabels([])  # Remove x-tick labels
        axes[0, 1].set_xlabel("")
        axes[0, 1].set_ylabel("Success Rate (%)")
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].set_title(
            f"Success Rates (pdb100)\nPDB: {pdb_code}, Category: {category}"
        )

        # Plot 3: Cumulative Distribution (afdb-swissprot)
        sns.lineplot(
            data=min_afdb_df_success,
            x="Score",
            y="cumulative_count_all",
            ax=axes[1, 0],
            hue="Experiment",
            palette=experiment_color_mapping,
            legend=False,
        )
        axes[1, 0].set_title(
            f"Cumulative Distribution of Successful Designs (afdb-swissprot)\nPDB: {pdb_code}, Category: {category}, N={min_afdb}"
        )
        axes[1, 0].set_xlabel("Score")
        axes[1, 0].set_ylabel("Cumulative Percentage (%)")
        axes[1, 0].set_xlim(0, 100)

        # Plot 4: Success Rates (afdb-swissprot) with adjusted colors
        success_rates = success_rates_afdb.set_index("Experiment")
        for i, experiment in enumerate(experiments):
            base_color = experiment_color_mapping.get(
                experiment, (0.7, 0.7, 0.7)
            )  # Default to gray if missing

            # Adjust colors: lighter for Success Rate @10, darker for Success Rate @N
            color_at_10 = adjust_lightness(base_color, 1.2)  # Lighter color
            color_at_N = adjust_lightness(base_color, 0.8)  # Darker color

            rate_at_10 = (
                success_rates.loc[experiment, "Success Rate @10 (%)"]
                if experiment in success_rates.index
                else 0
            )
            rate_at_N = (
                success_rates.loc[experiment, f"Success Rate @{min_afdb} (%)"]
                if experiment in success_rates.index
                else 0
            )

            # Plot Success Rate @10 (lighter bar) first
            axes[1, 1].bar(
                indices[i] - bar_width / 2,
                rate_at_10,
                width=bar_width,
                color=color_at_10,
            )
            axes[1, 1].bar(
                indices[i] + bar_width / 2, rate_at_N, width=bar_width, color=color_at_N
            )

        axes[1, 1].set_xticks(indices)
        axes[1, 1].set_xticklabels([])  # Remove x-tick labels
        axes[1, 1].set_xlabel("")
        axes[1, 1].set_ylabel("Success Rate (%)")
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].set_title(
            f"Success Rates (afdb-swissprot)\nPDB: {pdb_code}, Category: {category}"
        )

        # Prepare legend handles and labels
        # Metrics legend with lighter and darker greys
        metric_labels = ["Success Rate @10 (%)", f"Success Rate @{min_pdb100} (%)"]
        metric_patches = [
            mpatches.Patch(
                facecolor=(0.8, 0.8, 0.8), label=metric_labels[0]
            ),  # Lighter grey
            mpatches.Patch(
                facecolor=(0.4, 0.4, 0.4), label=metric_labels[1]
            ),  # Darker grey
        ]

        # Experiment legend
        experiment_patches = [
            mpatches.Patch(color=experiment_color_mapping[exp], label=exp)
            for exp in experiments
        ]

        # Combine legends
        handles = metric_patches + experiment_patches
        labels = [patch.get_label() for patch in handles]

        # Place the combined legend below all plots
        fig.legend(
            handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05)
        )

        # Adjust layout to accommodate the legend
        plt.tight_layout(
            rect=[0, 0.08, 1, 1]
        )  # Adjust rect to leave space at the bottom

        # Save the plot as a PDF
        plot_filename = os.path.join(
            output_dir, f"cumulative_success_rate_plot_{pdb_code}.pdf"
        )
        plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
        print(f"Plot saved to {plot_filename}")

        # Close the figure
        plt.close(fig)


def calculate_hamming_distance(real_labels, predicted_labels, label_types):
    """
    Calculate the Hamming distance between real and predicted labels, both represented as sets.
    """
    real_binary = np.array([1 if label in real_labels else 0 for label in label_types])
    predicted_binary = np.array(
        [1 if label in predicted_labels else 0 for label in label_types]
    )
    return hamming_loss(real_binary, predicted_binary) * len(label_types)


def plot_boxplot_hamming_vs_score(
    df: pd.DataFrame, label_types: list, category_to_binding: dict
) -> None:
    # Initialize lists to hold results
    hamming_distances = []

    # Calculate Hamming distance for each row
    for idx, row in df.iterrows():
        # Real labels from the Category column
        real_labels = set(row["Category"].split("+"))

        # Predicted labels from the binding columns
        predicted_labels = set(
            [
                label
                for label, binding_col in category_to_binding.items()
                if row[binding_col]
            ]
        )

        # Calculate Hamming distance
        hamming_dist = calculate_hamming_distance(
            real_labels, predicted_labels, label_types
        )
        hamming_distances.append(hamming_dist)

    df["Hamming Distance"] = hamming_distances

    # Create the boxplot for all experiments in the same plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Hamming Distance", y="Score", hue="Experiment")

    # Add titles and labels
    plt.title("Score Distribution by Hamming Distance and Experiment")
    plt.xlabel("Hamming Distance")
    plt.ylabel("Score")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_zero_hamming(df: pd.DataFrame) -> None:
    """
    Plot a violin plot for entries with Hamming distance 0.
    """
    df_zero_hamming = df[df["Hamming Distance"] == 0]

    # Set up a grid for the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the violin plot, restricted to Hamming distance of 0
    sns.violinplot(data=df_zero_hamming, x="Experiment", y="Score", ax=ax)
    ax.set_title("Scores for Entries with Hamming Distance 0")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Score")

    # Show the plot
    plt.tight_layout()
    plt.show()


import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
import typing as t


def generate_distance_matrix(
    binding_freq_dict: t.Dict[str, t.Dict[str, np.ndarray]]
) -> pd.DataFrame:

    # Get design identifiers
    pdb_experiments = list(binding_freq_dict.keys())
    # Create an empty distance matrix (DataFrame) with PDB_Experiment pairs as indices/columns
    distance_matrix = pd.DataFrame(
        np.zeros((len(pdb_experiments), len(pdb_experiments))),
        index=pdb_experiments,
        columns=pdb_experiments,
    )

    # Compute cosine distances between all pairs of PDB_Experiment
    for i, pdb1 in enumerate(pdb_experiments):
        for j in range(i + 1, len(pdb_experiments)):
            pdb2 = pdb_experiments[j]
            vector1 = binding_freq_dict[pdb1]  # Frequency vector for PDB1
            vector2 = binding_freq_dict[pdb2]  # Frequency vector for PDB2
            dist = cosine(vector1, vector2)  # Compute cosine distance

            # Set distance in both [i, j] and [j, i] due to symmetry
            distance_matrix.at[pdb1, pdb2] = dist
            distance_matrix.at[pdb2, pdb1] = dist

    # Set diagonal to 0 (self-comparison)
    np.fill_diagonal(distance_matrix.values, 0)

    return distance_matrix


def generate_functional_motif_matrices(
    df: t.Any,
) -> t.Dict[str, t.Dict[str, np.ndarray]]:
    # Initialize the resulting dictionary
    db_experiment_pdb_matrices: t.Dict[str, t.Dict[str, np.ndarray]] = {}

    # Combine 'Design ID' and 'Experiment Number' into a new column 'PDB_Experiment'
    df["PDB_Experiment"] = (
        df["Design ID"]
        + "_"
        + df["Experiment Number"].astype(str)
        + "_"
        + df["Experiment"].astype(str)
    )

    # Group by DB
    grouped_by_db = df.groupby("DB")

    # Iterate over each database
    for db, group_data_by_db in grouped_by_db:
        # Initialize a dictionary to store results for this DB
        experiment_pdb_vectors = {}

        # Group by PDB_Experiment to calculate frequency for each binding column
        grouped_by_pdb_experiment = group_data_by_db.groupby("PDB_Experiment")

        for pdb_experiment, group_data in grouped_by_pdb_experiment:
            # List of binding columns to analyze
            binding_columns = [
                "DNA binding",
                "RNA binding",
                "GTP binding",
                "ATP binding",
                "metal ion binding",
            ]

            # Calculate the frequency of True values for each binding column
            true_frequencies = group_data[binding_columns].mean().to_numpy()

            # Store the results
            experiment_pdb_vectors[pdb_experiment] = true_frequencies

        # Store results for the current DB
        db_experiment_pdb_matrices[db] = experiment_pdb_vectors

    return db_experiment_pdb_matrices


def flatten_experiments(
    binding_freq_dict: t.Dict[str, t.Dict[str, np.ndarray]]
) -> t.Dict[str, np.ndarray]:
    """
    Flatten the experiments for each database by keeping the original 'PDB_Experiment' as the key and the numpy array as the value.
    """
    flattened_dict: t.Dict[str, np.ndarray] = {}

    for db, experiment_dict in binding_freq_dict.items():
        for pdb_experiment_name, freq_vector in experiment_dict.items():
            flattened_dict[pdb_experiment_name] = freq_vector

    return flattened_dict


def calculate_rmsd_for_pair(args: Tuple[str, str]) -> t.Optional[float]:
    pdb_path1, pdb_path2 = args
    try:
        # Load original and design PDB
        original_pdb = ampal.load_pdb(pdb_path1)
        original_pdb = select_first_ampal_assembly(original_pdb)

        design_pdb = ampal.load_pdb(pdb_path2)
        design_pdb = select_first_ampal_assembly(design_pdb)

        # Calculate RMSD
        rmsd = RmsdBiopythonStrategy.biopython_calculate_rmsd(original_pdb, design_pdb)
        return rmsd
    except Exception as e:
        # Handle any loading errors or RMSD calculation errors
        print(f"Error calculating RMSD for {pdb_path1} vs {pdb_path2}: {e}")
        return None


# Main function to create the distance matrix
def create_rmsd_distance_matrix(
    pdb_paths: List[str], pdb_names: List[str]
) -> pd.DataFrame:
    # Initialize the matrix with zeros for diagonal elements
    matrix_size = len(pdb_paths)
    rmsd_matrix = pd.DataFrame(0, index=pdb_names, columns=pdb_names)

    # Create all unique pairs (i, j) where i < j to avoid redundant computations
    pdb_pairs = [
        (pdb_paths[i], pdb_paths[j])
        for i in range(matrix_size)
        for j in range(i + 1, matrix_size)
    ]

    # Use multiprocessing to calculate RMSD in parallel
    with Pool(mp.cpu_count() - 4) as pool:
        results = pool.map(calculate_rmsd_for_pair, pdb_pairs)

    # Fill the distance matrix
    for (pdb1, pdb2), rmsd in results:
        if rmsd is not None:
            name1 = pdb_names[pdb_paths.index(pdb1)]
            name2 = pdb_names[pdb_paths.index(pdb2)]
            rmsd_matrix.at[name1, name2] = rmsd
            rmsd_matrix.at[name2, name1] = rmsd  # Symmetry

    return rmsd_matrix


import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import spearmanr


def get_pdb_to_category_mapping(category_dict):
    """Extracts PDB to category mapping from the category dictionary."""
    pdb_to_category = {}
    for category, pdbs in category_dict.items():
        for pdb in pdbs:
            pdb_to_category[pdb] = category
    return pdb_to_category


def load_graph_fg(fg_file):
    fragment_graph = StructureFragmentGraph.load(fg_file).graph
    # Iterate through nodes and convert fragment_class to str
    for node in fragment_graph.nodes(data=True):
        node[1]["fragment_class"] = str(node[1]["fragment_class"])
    # Iterate through edges and convert peptide_bond to str
    for edge in fragment_graph.edges(data=True):
        edge[2]["peptide_bond"] = str(edge[2]["peptide_bond"])
    return fragment_graph


from tqdm import tqdm

def create_distance_vs_reference(
    dataset_name, reference_path, functional_motif_dict, fragment_path
):
    data_path = reference_path.parent
    # Extract Function Data
    binding_data = functional_motif_dict[dataset_name]
    design_to_distance = {}
    experiment_to_graphcreator = (
        {}
    )  # Store graph creators for each experiment to avoid reloading them
    for curr_design, motif_matrix in tqdm(binding_data.items(), desc="Processing designs"):
        pdb_name, experiment, folder = curr_design.split("_", 2)
        # Skip the reference
        if folder == "reference":
            continue
        exp_path = data_path / folder
        search_pattern = f"{pdb_name}*{experiment}.pdb"
        matching_files = list(exp_path.glob(search_pattern))
        if len(matching_files) == 0:
            print(f"No file found for {search_pattern}")
            continue
        else:
            pdb_path = matching_files[0]
        # Calculate RMSD with reference
        ref_search = f"{pdb_name}*.pdb1"
        ref_matching_files = list(reference_path.glob(ref_search))
        if len(ref_matching_files) == 0:
            print(f"No file found for {ref_search}")
            continue
        else:
            curr_ref = ref_matching_files[0]
        # Check if the reference file exists
        if not curr_ref.exists():
            print(f"Reference file {curr_ref} does not exist.")
            continue
        # Calculate RMSD
        curr_rmsd = calculate_rmsd_for_pair((pdb_path, curr_ref))
        # Calculate cosine distance with reference
        motif_reference = binding_data[f"{pdb_name}_0_reference"]
        curr_cos = cosine(motif_matrix, motif_reference)

        # Check if .fg file exists at path / [1:3] / name.fg
        design_fragment_path = pdb_path.parent / pdb_name[1:3] / f"{pdb_path.stem}.fg"
        # Check if reference .fg file exists
        ref_fragment_path =  curr_ref.parent / pdb_name[1:3] / f"{curr_ref.stem}.fg"
        if not ref_fragment_path.exists():
            if "reference" not in experiment_to_graphcreator:
                reference_graph_creator = load_graph_creator(
                    fragment_path, workers=10, pdb_path=reference_path
                )
                experiment_to_graphcreator["reference"] = reference_graph_creator
            else:
                reference_graph_creator = experiment_to_graphcreator["reference"]
            ref_fragment_path = reference_graph_creator.classify_and_save_graph(
                curr_ref
            )
            if not ref_fragment_path.exists():
                print(f"Error creating fragment graph for reference {curr_ref}")
                continue
        if not design_fragment_path.exists():
            # Load current experiment graph creator
            if folder not in experiment_to_graphcreator:
                exp_graph_creator = load_graph_creator(
                    fragment_path, workers=10, pdb_path=exp_path
                )
                experiment_to_graphcreator[folder] = exp_graph_creator
            else:
                exp_graph_creator = experiment_to_graphcreator[folder]
            design_fragment_path = exp_graph_creator.classify_and_save_graph(
                pdb_path
            )
            if not design_fragment_path.exists():
                print(f"Error creating fragment graph for {pdb_path}")
                continue
        # Load fragment graphs
        ref_graph = load_graph_fg(ref_fragment_path)
        design_graph = load_graph_fg(design_fragment_path)
        # Calculate graph distance
        comparator = gm.GraphEditDistance(1, 1, 1, 1)
        comparator.set_attr_graph_used(node_attr_key="fragment_class", edge_attr_key="peptide_bond")
        # Compute distance or similarity matrix
        curr_ged = comparator.compare([ref_graph, design_graph], None)
        # Add to dict
        design_to_distance[curr_design] = {
            "RMSD": curr_rmsd,
            "Cosine": curr_cos,
            "Graph Edit Distance": curr_ged.flatten()[1],
            "Experiment": folder,
        }
    return design_to_distance



def plot_and_calculate_correlations(
    pdb_to_category: t.Dict[str, str], dist_vs_ref_dict: t.Dict[str, t.Dict[str, float]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Prepare data for DataFrame
    data = []
    for entry, values in dist_vs_ref_dict.items():
        pdb_id = entry[:4]  # Extract PDB ID
        category = pdb_to_category.get(pdb_id, "Unknown")  # Get category from pdb_to_category
        cosine = values["Cosine"]
        rmsd = values["RMSD"]
        ged = values["Graph Edit Distance"]  # Added GED
        experiment = values["Experiment"]
        data.append(
            {
                "PDB_ID": pdb_id,
                "Category": category,
                "Cosine": cosine,
                "RMSD": rmsd,
                "GED": ged,
                "Experiment": experiment,
            }
        )

    df = pd.DataFrame(data)

    # Helper function to calculate Spearman correlation and format title
    def calc_spearman_title(x_col: str, y_col: str) -> str:
        corr, p_val = spearmanr(df[x_col], df[y_col])
        return f"Spearman: {corr:.2f}, p-value: {p_val:.2e}"

    ### First plot set: hue by Category
    fig1, axes1 = plt.subplots(1, 3, figsize=(21, 6))

    # Left plot: Cosine vs RMSD, colored by Category
    sns.scatterplot(ax=axes1[0], data=df, x="Cosine", y="RMSD", hue="Category")
    axes1[0].set_xlim(0, None)  # X-axis starts at 0
    axes1[0].set_ylim(0, None)  # Y-axis starts at 0
    axes1[0].set_title(f"Cosine vs RMSD (Colored by Category)\n{calc_spearman_title('Cosine', 'RMSD')}")

    # Middle plot: Cosine vs GED, colored by Category
    sns.scatterplot(ax=axes1[1], data=df, x="Cosine", y="GED", hue="Category")
    axes1[1].set_xlim(0, None)  # X-axis starts at 0
    axes1[1].set_ylim(0, None)  # Y-axis starts at 0
    axes1[1].set_title(f"Cosine vs GED (Colored by Category)\n{calc_spearman_title('Cosine', 'GED')}")

    # Right plot: RMSD vs GED, colored by Category
    sns.scatterplot(ax=axes1[2], data=df, x="RMSD", y="GED", hue="Category")
    axes1[2].set_xlim(0, None)  # X-axis starts at 0
    axes1[2].set_ylim(0, None)  # Y-axis starts at 0
    axes1[2].set_title(f"RMSD vs GED (Colored by Category)\n{calc_spearman_title('RMSD', 'GED')}")

    plt.tight_layout()
    plt.show()

    ### Second plot set: hue by Experiment
    fig2, axes2 = plt.subplots(1, 3, figsize=(21, 6))

    # Left plot: Cosine vs RMSD, colored by Experiment
    sns.scatterplot(ax=axes2[0], data=df, x="Cosine", y="RMSD", hue="Experiment")
    axes2[0].set_xlim(0, None)  # X-axis starts at 0
    axes2[0].set_ylim(0, None)  # Y-axis starts at 0
    axes2[0].set_title(f"Cosine vs RMSD (Colored by Experiment)\n{calc_spearman_title('Cosine', 'RMSD')}")

    # Middle plot: Cosine vs GED, colored by Experiment
    sns.scatterplot(ax=axes2[1], data=df, x="Cosine", y="GED", hue="Experiment")
    axes2[1].set_xlim(0, None)  # X-axis starts at 0
    axes2[1].set_ylim(0, None)  # Y-axis starts at 0
    axes2[1].set_title(f"Cosine vs GED (Colored by Experiment)\n{calc_spearman_title('Cosine', 'GED')}")

    # Right plot: RMSD vs GED, colored by Experiment
    sns.scatterplot(ax=axes2[2], data=df, x="RMSD", y="GED", hue="Experiment")
    axes2[2].set_xlim(0, None)  # X-axis starts at 0
    axes2[2].set_ylim(0, None)  # Y-axis starts at 0
    axes2[2].set_title(f"RMSD vs GED (Colored by Experiment)\n{calc_spearman_title('RMSD', 'GED')}")

    plt.tight_layout()
    plt.show()

    # Function to calculate Spearman correlations for all pairs
    def calculate_correlations(group_df: pd.DataFrame) -> pd.Series:
        spearman_cos_rmsd, spearman_p_cos_rmsd = spearmanr(group_df["Cosine"], group_df["RMSD"])
        spearman_cos_ged, spearman_p_cos_ged = spearmanr(group_df["Cosine"], group_df["GED"])
        spearman_rmsd_ged, spearman_p_rmsd_ged = spearmanr(group_df["RMSD"], group_df["GED"])

        return pd.Series(
            {
                "Spearman Correlation (Cosine vs RMSD)": spearman_cos_rmsd,
                "Spearman p-value (Cosine vs RMSD)": spearman_p_cos_rmsd,
                "Spearman Correlation (Cosine vs GED)": spearman_cos_ged,
                "Spearman p-value (Cosine vs GED)": spearman_p_cos_ged,
                "Spearman Correlation (RMSD vs GED)": spearman_rmsd_ged,
                "Spearman p-value (RMSD vs GED)": spearman_p_rmsd_ged,
            }
        )

    # Calculate correlations by Category
    category_correlations = df.groupby("Category").apply(calculate_correlations)

    # Calculate correlations by Experiment
    experiment_correlations = df.groupby("Experiment").apply(calculate_correlations)

    return category_correlations, experiment_correlations


def main(args):
    # Check if design and reference paths exist
    check_paths_exist(args.design_paths)
    assert (
        args.reference_path.exists()
    ), f"Input file {args.reference_path} does not exist."

    # Initialize Ontology and category dictionary
    ontology = Ontology(Path("data/"))
    category_dict = create_reference_category_dict(args.reference_path)
    # Add designs to category dictionary
    add_designs_to_category_dict(category_dict, args.design_paths)

    # Load or parse results
    experiments_to_dfs = load_or_parse_results(
        category_dict, args.num_results, ontology
    )

    # Get experiments names from first category and PDB
    first_category = next(iter(category_dict))
    first_pdb = next(iter(category_dict[first_category]))
    experiments_names = category_dict[first_category][first_pdb]["designs"]

    # Save experiment results
    for experiment_name in experiments_names:
        save_experiment_results(experiment_name, experiments_to_dfs)

    merged_df = merge_experiment_dfs(experiments_to_dfs)
    merged_df = add_success_column(merged_df)

    # plot_cumulative_graphs_per_pdb(merged_df)

    # save merged_df to a csv file
    # merged_df.to_csv("merged_df.csv", index=False)
    functional_motif_dict = generate_functional_motif_matrices(merged_df)
    # dist_vs_ref_dict = create_distance_vs_reference(
    #     "pdb100", args.reference_path, functional_motif_dict, args.fragment_path
    # )
    # Define the file path to save/load the dictionary
    dict_filename = "dist_vs_ref_dict.pkl"
    import pickle
    # Check if the file already exists
    if os.path.exists(dict_filename):
        # Load the dictionary from the file
        with open(dict_filename, 'rb') as f:
            dist_vs_ref_dict = pickle.load(f)
        print(f"Loaded dictionary from {dict_filename}")
    else:
        # Run the function and create the dictionary
        dist_vs_ref_dict = create_distance_vs_reference("pdb100", args.reference_path, functional_motif_dict, args.fragment_path)

        # Save the dictionary to a file
        with open(dict_filename, 'wb') as f:
            pickle.dump(dist_vs_ref_dict, f)
        print(f"Saved dictionary to {dict_filename}")

    # # Create RMSD vs Reference
    # data_path = args.reference_path.parent
    # pbd_paths = []
    # pdb_names = []
    # for pdb in cosine_distance_matrix.columns:
    #     pdb_name, experiment, folder = pdb.split("_", 2)
    #     exp_path = data_path / folder
    #     search_pattern = f"{pdb_name}*{experiment}.pdb"
    #     matching_files = list(exp_path.glob(search_pattern))
    #     if len(matching_files) == 0:
    #         print(f"No file found for {search_pattern}")
    #         continue
    #     else:
    #         pdb_path = matching_files[0]
    #         pbd_paths.append(pdb_path)
    #         pdb_names.append(pdb)
    #
    # rmsd_distance_matrix = (
    #     create_RMSD_distance_matrix(pbd_paths, pdb_names) if not os.path.exists("rmsd_matrix.csv") else pd.read_csv(
    #         "rmsd_matrix.csv", index_col=0))
    # if not os.path.exists("rmsd_matrix.csv"):
    #     rmsd_distance_matrix.to_csv("rmsd_matrix.csv", index=True)
    #
    # # Select the common columns between rmsd_distance_matrix and cosine_distance_matrix
    # common_columns = rmsd_distance_matrix.columns.intersection(cosine_distance_matrix.columns)
    # rmsd_common = rmsd_distance_matrix[common_columns].loc[common_columns]
    # cosine_common = cosine_distance_matrix[common_columns].loc[common_columns]

    # cosine_distance_matrix = generate_distance_matrix(binding_freq_dict["pdb100"])

    # Create RMSD distance matrix
    # data_path = args.reference_path.parent
    # pbd_paths = []
    # pdb_names = []
    # for pdb in cosine_distance_matrix.columns:
    #     pdb_name, experiment, folder = pdb.split("_", 2)
    #     exp_path = data_path / folder
    #     search_pattern = f"{pdb_name}*{experiment}.pdb"
    #     matching_files = list(exp_path.glob(search_pattern))
    #     if len(matching_files) == 0:
    #         print(f"No file found for {search_pattern}")
    #         continue
    #     else:
    #         pdb_path = matching_files[0]
    #         pbd_paths.append(pdb_path)
    #         pdb_names.append(pdb)
    #
    # rmsd_distance_matrix = (
    #     create_RMSD_distance_matrix(pbd_paths, pdb_names) if not os.path.exists("rmsd_matrix.csv") else pd.read_csv(
    #         "rmsd_matrix.csv", index_col=0))
    # if not os.path.exists("rmsd_matrix.csv"):
    #     rmsd_distance_matrix.to_csv("rmsd_matrix.csv", index=True)
    #
    # # Select the common columns between rmsd_distance_matrix and cosine_distance_matrix
    # common_columns = rmsd_distance_matrix.columns.intersection(cosine_distance_matrix.columns)
    # rmsd_common = rmsd_distance_matrix[common_columns].loc[common_columns]
    # cosine_common = cosine_distance_matrix[common_columns].loc[common_columns]
    #
    # Get the PDB to category mapping
    pdb_to_category = get_pdb_to_category_mapping(category_dict)
    #
    # # Filter for within-category pairs
    # rmsd_within_category = []
    # cosine_within_category = []
    # categories_within = []
    #
    # for i, pdb1 in enumerate(common_columns):
    #     for j, pdb2 in enumerate(common_columns):
    #         if i >= j:
    #             continue  # Skip symmetric pairs and self-comparisons
    #
    #         # Check if pdb1 and pdb2 are in the same category
    #         pdb1_category = pdb_to_category[pdb1.split("_")[0]]
    #         pdb2_category = pdb_to_category[pdb2.split("_")[0]]
    #         if pdb1_category == pdb2_category:
    #             rmsd_within_category.append(rmsd_common.iloc[i, j])
    #             cosine_within_category.append(cosine_common.iloc[i, j])
    #             categories_within.append(pdb1_category)
    #
    # # Convert lists to numpy arrays for consistency
    # rmsd_within_category = np.array(rmsd_within_category)
    # cosine_within_category = np.array(cosine_within_category)
    #
    # # Ensure the lengths match
    # assert len(rmsd_within_category) == len(cosine_within_category) == len(
    #     categories_within), "Mismatched lengths for within-category data"
    #
    # # Calculate the Spearman correlation between RMSD and cosine distance for within-category pairs
    # spearman_corr, pvalue = spearmanr(rmsd_within_category, cosine_within_category)
    # print(f"Spearman correlation between RMSD and cosine distance (within-category): {spearman_corr} pvalue {pvalue}")
    #
    # # Plot RMSD vs Cosine Distance for within-category pairs with Seaborn
    # plt.figure(figsize=(8, 6))
    # sns.set(style="whitegrid")
    # sns.scatterplot(x=rmsd_within_category, y=cosine_within_category, hue=categories_within, palette="Set2", legend="full")
    #
    # plt.xlabel('RMSD (Within Category)')
    # plt.ylabel('Cosine Distance (Within Category)')
    # plt.title('RMSD vs Cosine Distance (Spearman correlation) - Within Category')
    # plt.grid(True)
    # plt.legend(title='Category')
    # plt.show()

    category_corr, experiment_corr = plot_and_calculate_correlations(
        pdb_to_category, dist_vs_ref_dict
    )
    raise ValueError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--reference_path", type=Path, help="Path to input file")
    parser.add_argument(
        "--design_paths",
        type=Path,
        nargs="+",
        help="Paths to input files (one or more)",
    )
    parser.add_argument(
        "--fragment_path", type=Path, required=True, help="Path to fragment classifier"
    )
    parser.add_argument(
        "--num_results", type=int, default=10, help="Number of results to analyse"
    )
    params = parser.parse_args()
    main(params)
