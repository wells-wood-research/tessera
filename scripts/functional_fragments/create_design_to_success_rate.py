import pickle
import sys
import typing as t
import pandas as pd
from requests import get, post
from time import sleep
import requests
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.fragments.classification_config import (
    UniprotResults,
    go_to_prosite,
    selected_pdbs,
)
from src.function_prediction.uniprot_processing import Ontology, UniprotDownloader
from src.scripts.functional_fragments.foldseek_check_proteins import (
    add_go_terms,
    get_uniprot_from_pdb,
    process_all_pdb_files,
    slice_arg,
)
from tqdm import tqdm
from pathlib import Path
import typing as t
import pandas as pd
import json
import matplotlib.cm as cm


def parse_results(
    results: dict,
    design_id: str,
    pdb_file_path: Path,
    alignment_slice: t.Union[slice, t.Tuple[int, int]],
) -> pd.DataFrame:
    """
    Parse the results from the API and extract relevant information, including the category.
    """
    data = []
    category = pdb_file_path.stem.split("_")[1]

    for result in results["results"]:
        db = result["db"]
        if result["alignments"]:
            for i, alignment in enumerate(result["alignments"][alignment_slice]):
                target = alignment["target"]
                prob = alignment["prob"]
                score = alignment["score"]

                if db == "swissprot":
                    uniprot, rest = target.split("-")[1], target.split("_")[1]
                    id_ = target
                elif db == "pdb":
                    parts = target.split("_")
                    pdb_id = parts[0].split("-")[0]
                    chain = parts[1][0]
                    id_ = target
                    pdb_directory = pdb_file_path.parent
                    uniprot = get_uniprot_from_pdb(pdb_id, chain, pdb_directory)
                else:
                    raise ValueError(f"Unknown db: {db}")

                data.append([design_id, category, db, uniprot, id_, prob, score])

    df = pd.DataFrame(
        data,
        columns=[
            "Design ID",
            "Category",
            "DB",
            "Uniprot",
            "ID",
            "Probability",
            "Score",
        ],
    )
    return df


def process_foldseek_results(
    input_foldseek: Path,
    alignment_slice: t.Union[slice, t.Tuple[int, int]],
) -> pd.DataFrame:
    """
    Processes FoldSeek results by iterating over JSON files in a directory, parsing results,
    and combining them into a single DataFrame.

    Args:
        input_foldseek (Path): Path to the directory containing FoldSeek JSON files.
        alignment_slice (t.Union[slice, t.Tuple[int, int]]): Slice or tuple to specify alignment slicing.

    Returns:
        pd.DataFrame: Combined DataFrame containing results from all JSON files.
    """
    # List all JSON files in the input directory
    json_files = list(input_foldseek.glob("*.json"))
    # Initialize the Ontology object once
    ontology = Ontology(Path("data/"))

    dfs = []

    # Process each JSON file with a progress bar
    for json_file in tqdm(json_files, desc="Processing FoldSeek JSON files"):
        # Extract category and pdb from the file name
        pdb_code, category, _ = json_file.stem.split("_", 2)
        # Load JSON file
        with open(json_file, "r") as f:
            results = json.load(f)[0]
        # Extract results
        df = parse_results(
            results,
            design_id=json_file.stem,
            pdb_file_path=json_file,
            alignment_slice=alignment_slice,
        )
        df = add_go_terms(df, ontology)
        if not df.empty:
            dfs.append(df)

    # Combine all DataFrames into one
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    return combined_df


# Split the categories from the "Category" column for each row
def check_success(row):
    categories = row["Category"].split("+")  # Split categories like DNA+RNA
    return all(row[f"{cat} binding"] for cat in categories if f"{cat} binding" in row)


# def sort_categories(df, column_name):
#     """
#     Sorts the categories in the specified column of the dataframe:
#     - Categories without "+" come first.
#     - Categories with one "+" come next.
#     - Categories with two "+" come last.
#     Within each group, sorts alphabetically.
#     """
#
#     def category_sort_key(category):
#         num_plus = category.count("+")
#         return (
#             num_plus,
#             category.lower(),
#         )  # Sort by count of "+" first, then alphabetically
#
#     sorted_categories = sorted(df[column_name].unique(), key=category_sort_key)
#     df[column_name] = pd.Categorical(
#         df[column_name], categories=sorted_categories, ordered=True
#     )
#     return df


def plot_barplot_with_control_and_design(
    per_category_success_stats, per_category_success_stats_control, output_path: Path
) -> None:
    """
    Plots barplots for both control and design data in two subplots: one for pdb and one for swissprot.
    The y-axis is converted to percentages, and the control bars always appear first.
    """
    # Combine design and control data
    per_category_success_stats["Type"] = "Design"
    per_category_success_stats_control["Type"] = "Control"
    combined_stats = pd.concat(
        [per_category_success_stats, per_category_success_stats_control],
        ignore_index=True,
    )

    # Sort categories for consistent ordering and ensure Control is plotted first
    combined_stats = sort_categories(combined_stats, "Category")
    combined_stats["Type"] = pd.Categorical(
        combined_stats["Type"], categories=["Control", "Design"], ordered=True
    )

    # Convert success rates to percentages
    combined_stats["success_rate"] *= 100
    combined_stats["std_dev"] *= 100

    # Create subplots: one for pdb and one for swissprot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    db_values = ["pdb", "swissprot"]

    for i, db in enumerate(db_values):
        ax = axes[i]
        db_stats = combined_stats[combined_stats["DB"] == db]
        sns.barplot(
            data=db_stats,
            x="Category",
            y="success_rate",
            hue="Type",
            ci=None,  # Disable confidence intervals; std will be added as error bars
            ax=ax,
            palette="viridis",
        )
        #
        # # Add standard deviation as error bars
        # for bar, (_, row) in zip(ax.patches, db_stats.iterrows()):
        #     x = bar.get_x() + bar.get_width() / 2
        #     y = bar.get_height()
        #     std_dev = row["std_dev"]
        #     ax.errorbar(x, y, yerr=std_dev, fmt="none", color="black", capsize=4)

        # Set subplot details
        ax.set_title(f"Success Rates for {db.upper()}")
        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(0, 100)  # Start y-axis from 0 and go up to 100%
        if i == len(db_values) - 1:
            ax.set_xlabel("Category")
        else:
            ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=90)

    # Add legend and adjust layout
    axes[0].legend(title="Type", loc="upper left")
    axes[1].legend().remove()
    plt.tight_layout()
    plt.savefig(output_path / "success_rates.pdf")


def sort_categories(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Sorts the categories in the specified column of the dataframe:
    - Categories without "+" come first.
    - Categories with one "+" come next.
    - Categories with two "+" come last.
    Within each group, sorts alphabetically.
    """

    def category_sort_key(category: str) -> t.Tuple[int, str]:
        return (category.count("+"), category.lower())

    sorted_categories = sorted(df[column_name].unique(), key=category_sort_key)
    df[column_name] = pd.Categorical(
        df[column_name], categories=sorted_categories, ordered=True
    )
    return df


# Use the viridis colormap for consistent coloring
viridis = cm.get_cmap("viridis")


def get_color(category: str) -> t.Any:
    num_plus = category.count("+")
    if num_plus == 0:
        return viridis(0.25)  # Single-function: same for all categories with no '+'
    elif num_plus == 1:
        return viridis(0.5)  # Double-function: same for all categories with one '+'
    elif num_plus == 2:
        return viridis(0.75)  # Triple-function: same for all categories with two '+'
    else:
        return viridis(1.0)


def plot_relative_success_rate(per_category_success_stats: pd.DataFrame, per_category_success_stats_control: pd.DataFrame,
        design_success_stats: pd.DataFrame, output_path: Path, ) -> None:
    """
    Plots a barplot and a boxplot for the relative success rate:
      - Barplot: Shows aggregated relative success rates.
      - Boxplot: Shows the per-design distribution of relative success rates.
    """

    # --- 1. MERGE & CALCULATE ---
    agg_merged = pd.merge(per_category_success_stats.rename(columns={"success_rate": "success_rate_design"}),
        per_category_success_stats_control.rename(columns={"success_rate": "success_rate_control"}), on=["Category", "DB"],
        how="left", )

    agg_merged["relative_success_rate"] = (agg_merged["success_rate_design"] / agg_merged["success_rate_control"]).fillna(
        0) * 100

    # --- 2. DEFINE ALL TARGET CATEGORIES ---
    # The complete list of categories you want to appear on the axis
    target_categories = ["METAL", "ATP", "GTP", "DNA", "RNA", "ATP+GTP", "ATP+METAL", "DNA+ATP", "DNA+GTP", "DNA+METAL",
        "DNA+RNA", "GTP+METAL", "RNA+ATP", "RNA+GTP", "RNA+METAL", "ATP+GTP+METAL", "DNA+ATP+GTP", "DNA+ATP+METAL",
        "DNA+RNA+ATP", "DNA+RNA+METAL", "RNA+ATP+GTP", "RNA+ATP+METAL", "RNA+GTP+METAL"]

    # --- 3. INJECT MISSING DATA (CRITICAL STEP) ---
    # We must ensure every category in target_categories exists for DB='pdb'
    # otherwise the bar won't be drawn.

    # Filter for PDB entries currently in data
    existing_pdb = agg_merged[agg_merged["DB"] == "pdb"]["Category"].unique()

    missing_rows = []
    for cat in target_categories:
        if cat not in existing_pdb:
            # Create a dummy row for this missing category
            missing_rows.append({"Category": cat, "DB": "pdb", "success_rate_design": 0.0, "success_rate_control": 1.0,
                # Dummy value to prevent div/0 issues
                "relative_success_rate": 0.0  # The important part: 0% height
            })

    if missing_rows:
        agg_merged = pd.concat([agg_merged, pd.DataFrame(missing_rows)], ignore_index=True)

    # --- 4. SORTING LOGIC ---
    # Define fixed order for single-function categories
    single_function_order = ["METAL", "ATP", "GTP", "DNA", "RNA"]

    # Split targets into single and multi
    single_cats = [c for c in target_categories if "+" not in c]
    multi_cats = [c for c in target_categories if "+" in c]

    # Sort single: explicitly follow your preferred order
    single_cats_sorted = sorted(single_cats, key=lambda x: single_function_order.index(x) if x in single_function_order else 999)

    # Sort multi: by number of '+' then alphabetical
    multi_cats_sorted = sorted(multi_cats, key=lambda cat: (cat.count("+"), cat.lower()))

    final_order = single_cats_sorted + multi_cats_sorted

    # Apply the categorical type to the dataframe
    agg_merged["Category"] = pd.Categorical(agg_merged["Category"], categories=final_order, ordered=True)

    # --- 5. PLOTTING ---
    fig, ax = plt.subplots(figsize=(4, 4))  # Increased width slightly for legibility

    # Setup Colors
    palette = sns.color_palette("colorblind", n_colors=3)
    function_colors = {"single": palette[0], "dual": palette[1], "triple": palette[2], }

    category_colors = {}
    for cat in final_order:
        if "+" not in cat:
            category_colors[cat] = function_colors["single"]
        elif cat.count("+") == 1:
            category_colors[cat] = function_colors["dual"]
        else:
            category_colors[cat] = function_colors["triple"]

    # Plot (Data must be sorted by the category for Seaborn to respect order perfectly)
    plot_data = agg_merged[agg_merged["DB"] == "pdb"].sort_values("Category")

    sns.barplot(data=plot_data, x="Category", y="relative_success_rate", palette=category_colors, edgecolor="black",
        errorbar=None, ax=ax)

    # Aesthetics
    ax.set_ylabel("Relative Recovery Rate (%)", fontsize=14)
    # ax.set_title("Alva et al. Fragments", fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Category", fontsize=14)
    ax.tick_params(axis="x", rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(output_path / "relative_success_rates_barplot.pdf", format="pdf", bbox_inches="tight", )

    # --- LEGEND ---
    legend_handles = [plt.Line2D([0], [0], color=function_colors["single"], lw=4, label="Single"),
        plt.Line2D([0], [0], color=function_colors["dual"], lw=4, label="Dual"),
        plt.Line2D([0], [0], color=function_colors["triple"], lw=4, label="Triple"), ]
    fig_legend, ax_legend = plt.subplots(figsize=(4, 1))
    ax_legend.axis("off")
    ax_legend.legend(handles=legend_handles, title="Function Type", loc="center", ncol=3, fontsize=12, title_fontsize=12,
        frameon=True, )
    fig_legend.savefig(output_path / "relative_success_rates_legend.pdf", format="pdf", bbox_inches="tight", )
    plt.close(fig_legend)

    # --- BOXPLOT ---
    box_data = pd.merge(design_success_stats.rename(columns={"success_rate": "success_rate_design"}),
        per_category_success_stats_control[["Category", "DB", "success_rate"]].rename(
            columns={"success_rate": "success_rate_control"}), on=["Category", "DB"], how="left", )
    box_data["relative_success_rate"] = (box_data["success_rate_design"] / box_data["success_rate_control"]).fillna(0) * 100

    # Apply same categorical order
    box_data["Category"] = pd.Categorical(box_data["Category"], categories=final_order, ordered=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    db_values = ["pdb", "swissprot"]
    for i, db in enumerate(db_values):
        ax = axes[i]
        db_box = box_data[box_data["DB"] == db]
        sns.boxplot(data=db_box, x="Category", y="relative_success_rate", palette=category_colors, ax=ax, linewidth=1.5,
            fliersize=3, )
        ax.set_title(f"Relative Success Rate for {db.upper()}", fontsize=14)
        ax.set_ylabel("Relative Success Rate (%)", fontsize=12)
        ax.tick_params(axis="x", rotation=90)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        if i == len(db_values) - 1:
            ax.set_xlabel("Category", fontsize=14)

    plt.tight_layout()
    fig.savefig(output_path / "relative_success_rates_boxplot.pdf", format="pdf", bbox_inches="tight", )

def load_data(input_foldseek, csv_output_path, slice):
    if csv_output_path.exists():
        combined_df = pd.read_csv(csv_output_path)
    else:
        # Process json files
        combined_df = process_foldseek_results(input_foldseek, alignment_slice=slice)
        # Add success column
        combined_df["Success"] = combined_df.apply(check_success, axis=1)
        # Save to CSV
        combined_df.to_csv(csv_output_path, index=False)

    # Add column for PDB by splitting design ID by _ and selecting the first element
    combined_df["PDB"] = combined_df["Design ID"].apply(lambda x: x.split("_")[0])
    # Filter out rows where the PDB is not in selected_pdbs
    combined_df = combined_df[combined_df["PDB"].isin(selected_pdbs)]

    # Calculate success rates by Design ID and then Binding Combination
    success_stats = (
        combined_df.groupby(["Design ID", "Category", "DB"])
        .agg(success_rate=("Success", "mean"), std_dev=("Success", "std"))
        .reset_index()
    )
    # Calculate the mean success rate for each category
    per_category_success_stats = (
        success_stats.groupby(["Category", "DB"])
        .agg(success_rate=("success_rate", "mean"), std_dev=("success_rate", "std"))
        .reset_index()
    )
    return combined_df, success_stats, per_category_success_stats


def main(args):
    assert (
        args.input_foldseek_design.exists()
    ), f"Path does not exist: {args.input_foldseek_design}"
    assert (
        args.input_foldseek_control.exists()
    ), f"Path does not exist: {args.input_foldseek_control}"
    assert args.output_path.exists(), f"Path does not exist: {args.output_path}"
    csv_output_path = args.output_path / "design_combined_results.csv"
    csv_output_path_control = args.output_path / "control_combined_results.csv"
    combined_df, success_stats, per_category_success_stats = load_data(
        args.input_foldseek_design, csv_output_path, args.slice
    )
    (
        combined_df_control,
        success_stats_control,
        per_category_success_stats_control,
    ) = load_data(args.input_foldseek_control, csv_output_path_control, args.slice)

    plot_barplot_with_control_and_design(
        per_category_success_stats, per_category_success_stats_control, args.output_path
    )
    plot_relative_success_rate(
        per_category_success_stats,
        per_category_success_stats_control,
        success_stats,
        args.output_path,
    )

    raise ValueError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input_foldseek_design", type=Path, help="Path to input file (designs)"
    )
    parser.add_argument(
        "--input_foldseek_control", type=Path, help="Path to input file (controls)"
    )
    parser.add_argument(
        "--slice",
        required=True,
        type=slice_arg,
        default=slice(None, 1),
        help="Slice for result['alignments'][0]",
    )
    parser.add_argument("--output_path", type=Path, help="Path to output file")
    parser.add_argument(
        "--ignore_orginal_pdb",
        action="store_true",
        help="Ignore the original PDB used to generate the design",
    )
    params = parser.parse_args()
    main(params)
