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

from tessera.fragments.classification_config import UniprotResults, go_to_prosite
from tessera.function_prediction.uniprot_processing import Ontology, UniprotDownloader


def submit_job(pdb_file_path: Path) -> dict:
    """
    Submit a job for a given PDB file.
    """
    with pdb_file_path.open("r") as pdb_file:
        ticket = post(
            "https://search.foldseek.com/api/ticket",
            files={"q": pdb_file},
            data={
                "mode": "tmalign",
                "database[]": ["afdb-swissprot", "pdb100"],
            },
        ).json()
    return ticket


def poll_job(ticket: dict) -> None:
    """
    Poll the job until it's complete.
    """
    repeat = True
    while repeat:
        response = get(f"https://search.foldseek.com/api/ticket/{ticket['id']}")
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            sys.exit(1)

        status = response.json()
        if status.get("status") == "ERROR":
            sys.exit(0)

        sleep(1)
        repeat = status.get("status") != "COMPLETE"


def load_results(ticket: dict) -> dict:
    """
    Load the results of the job.
    """
    json_results = get(f"https://search.foldseek.com/api/result/{ticket['id']}/0")
    results = json_results.json()
    return results


def save_results(results: dict, design_id: str, output_dir: Path) -> None:
    """
    Save the results to a file.
    """
    output_file = output_dir / f"{design_id}_results.json"
    with output_file.open("w") as f:
        json.dump(results, f)


def get_uniprot_from_pdb(pdb_id: str, chain: str, pdb_directory: Path) -> str:
    """
    Get the UniProt ID from the PDBe API using the PDB ID and chain.
    """
    # Check if file already exists
    response_file = pdb_directory / f"{pdb_id}_uniprot_mapping.json"
    if response_file.exists():
        with response_file.open("r") as f:
            data = json.load(f)
    else:
        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
        response = requests.get(url)

        if response.status_code != 200:
            return "NA"

        data = response.json().get(pdb_id.lower(), {}).get("UniProt", {})
        # Save to a file for future reference
        with open(f"{pdb_id}_uniprot_mapping.json", "w") as f:
            json.dump(data, f)
    # Find the matching chain for the UniProt entry
    for uniprot_id, details in data.items():
        for mapping in details.get("mappings", []):
            if mapping["chain_id"] == chain:
                return uniprot_id

    return "NA"


def slice_arg(arg: str) -> t.Union[slice, t.Tuple[int, int]]:
    """Parses the input string to convert it into a slice object or tuple."""
    parts = arg.split(":")

    if len(parts) == 1:  # Single index case
        return int(parts[0])

    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if parts[1] else None
    return slice(start, end)


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
                    [design_id, category, db, uniprot, protein_name, id_, prob, score]
                )

    df = pd.DataFrame(
        data,
        columns=[
            "Design ID",
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

import os
# Save function for Uniprot entry
from multiprocessing import Lock

# Define a global lock
file_lock = Lock()

def save_uniprot_entry(uniprot_id: str, data: UniprotResults) -> None:
    pickle_path = Path(f"{uniprot_id}.pkl")
    temp_path = pickle_path.with_suffix(".tmp")

    # Acquire the global lock to ensure that only one process is writing the file at a time
    with file_lock:
        try:
            with temp_path.open("wb") as f:
                pickle.dump(data._asdict(), f)  # Save as dict

            # Rename temp file to the final .pkl file (atomic operation)
            temp_path.rename(pickle_path)

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()  # Cleanup temp file if an error occurs
            raise

# Load function with checks for file validity
def load_uniprot_entry(uniprot_id: str) -> t.Optional[UniprotResults]:
    pickle_path = Path(f"{uniprot_id}.pkl")
    if pickle_path.exists() and os.path.getsize(pickle_path) > 0:  # Ensure file is not empty
        with pickle_path.open("rb") as f:
            try:
                data_dict = pickle.load(f)  # Load as dict
                return UniprotResults(**data_dict)  # Convert back to namedtuple
            except EOFError:
                print(f"Error: The pickle file {pickle_path} is corrupted or empty.")
    return None


# Fetch GO terms with option to load/save cached data
def fetch_go_terms(uniprot_id: str, ontology: Ontology) -> t.Dict[str, t.Any]:
    if uniprot_id == "NA":
        return {
            **{go_to_prosite[go]["description"]: False for go in go_to_prosite},
            "GO Codes": [],
        }

    # Try loading saved UniprotResults
    uniprot_results = load_uniprot_entry(uniprot_id)
    if not uniprot_results:
        # Fetch from UniprotDownloader if not available locally
        try:
            uniprot_results = UniprotDownloader.get_uniprot_entry(uniprot_id)
        except Exception as e:
            print(f"Error: {e}")
            return {
                **{go_to_prosite[go]["description"]: False for go in go_to_prosite},
                "GO Codes": [],
            }
        save_uniprot_entry(uniprot_id, uniprot_results)

    # Process GO terms
    go_codes = set(uniprot_results.go_codes)
    for g in uniprot_results.go_codes:
        extra_go_codes = ontology.get_ancestors(g)
        go_codes.update(extra_go_codes)

    go_terms = {
        go_to_prosite[go]["description"]: go in go_codes for go in go_to_prosite
    }
    go_terms["GO Codes"] = go_codes
    return go_terms


def add_go_terms(df: pd.DataFrame, ontology: Ontology) -> pd.DataFrame:
    uniprot_ids = df["Uniprot"].tolist()
    go_terms_list = []
    for uniprot_id in uniprot_ids:
        go_terms = fetch_go_terms(uniprot_id, ontology)
        go_terms_list.append(go_terms)
    go_terms_df = pd.DataFrame(go_terms_list)
    df = pd.concat([df, go_terms_df], axis=1)
    return df


def process_pdb_file(
    pdb_file_path: Path,
    ontology: Ontology,
    alignment_slice: t.Union[slice, t.Tuple[int, int]],
) -> pd.DataFrame:
    """
    Process a PDB file, submitting it to the FoldSeek API, polling for results,
    and parsing the results to extract relevant information.
    """
    design_id = pdb_file_path.stem  # Get the file name without the extension
    json_file = pdb_file_path.parent / f"{design_id}_results.json"

    if json_file.exists():
        results = json.load(json_file.open())
    else:
        ticket = submit_job(pdb_file_path)
        poll_job(ticket)
        results = load_results(ticket)
        save_results(results, design_id, output_dir=pdb_file_path.parent)

    # Pass the alignment_slice to parse_results
    df = parse_results(results, design_id, pdb_file_path, alignment_slice)
    df = add_go_terms(df, ontology)
    return df


def process_all_pdb_files(
    input_folder: Path, alignment_slice: t.Union[slice, t.Tuple[int, int]]
) -> pd.DataFrame:
    # Use Path.glob to list all .pdb files or .pdb1 files
    pdb_files = list(input_folder.glob("*.pdb*"))

    # Initialize the Ontology object once
    ontology = Ontology(Path("data/"))

    dfs = []
    for pdb in tqdm(pdb_files, desc="Processing PDB files"):
        df = process_pdb_file(pdb, ontology, alignment_slice)
        if not df.empty:
            dfs.append(df)

    # Combine all DataFrames into one
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
    return combined_df


def plot_percentage_hits_with_function(data: pd.DataFrame, output_path: Path) -> None:
    """
    For each category, create a subplot showing the percentage of hits with each binding function.
    Include an "Others" category at the end of the x-axis.

    :param data: DataFrame containing the binding columns and 'Category' column.
    :param output_path: Path to save the plot.
    """
    # Get all binding combinations present in the data
    all_combinations = data["Binding Combination"].unique()
    # Ensure 'Others' is at the end
    all_combinations = [bc for bc in all_combinations if bc != "Others"] + ["Others"]

    categories = data["Category"].unique()
    num_categories = len(categories)

    fig, axes = plt.subplots(num_categories, 1, figsize=(12, 6 * num_categories))

    if num_categories == 1:
        axes = [axes]  # Ensure axes is a list

    for idx, category in enumerate(categories):
        ax = axes[idx]
        category_data = data[data["Category"] == category]

        total_hits = len(category_data)
        if total_hits == 0:
            print(f"No data for category {category}.")
            continue

        # Count occurrences of each binding combination
        combination_counts = (
            category_data["Binding Combination"]
            .value_counts()
            .reindex(all_combinations, fill_value=0)
        )

        # Calculate percentages
        percentages = (combination_counts / total_hits) * 100

        # Create DataFrame for plotting
        plot_df = pd.DataFrame(
            {"Binding Combination": percentages.index, "Percentage": percentages.values}
        )

        sns.barplot(x="Binding Combination", y="Percentage", data=plot_df, ax=ax)
        ax.set_title(f"Percentage of Hits with Function in Category: {category}")
        ax.set_xlabel("Binding Combination")
        ax.set_ylabel("Percentage (%)")
        ax.tick_params(axis="x", rotation=90)
        ax.set_ylim(0, 100)  # Since percentages

    plt.tight_layout()
    plt.savefig(output_path / "percentage_hits_function.pdf")
    plt.show()


def plot_success_rate_per_category(data: pd.DataFrame, output_path: Path) -> None:
    """
    For each category, create one bar representing the success rate.

    Success Rate (per category) = (Number of positive hits in that category) / (Number of hits in that category)

    Proteins with combined functions are counted for each category they represent.

    :param data: DataFrame containing the 'Category' and binding combination columns.
    :param output_path: Path to save the plot.
    """
    # Calculate total number of hits for each category
    total_hits_per_category = data["Category"].value_counts()

    # Modify the calculation for positive_hits_per_category
    positive_hits_per_category = (
        data[data["Category"] == data["Binding Combination"]].groupby("Category").size()
    )

    # Calculate success rate for each category
    success_rates = positive_hits_per_category / total_hits_per_category

    # Create DataFrame for plotting
    success_rate_df = pd.DataFrame(
        {"Category": success_rates.index, "Success Rate": success_rates.values}
    )

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Category", y="Success Rate", data=success_rate_df)
    plt.title("Success Rate per Category")
    plt.ylim(0, 1)
    plt.ylabel("Success Rate")
    plt.xlabel("Category")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path / "success_rate_per_category.pdf")
    plt.show()


def plot_confusion_matrix(data: pd.DataFrame, output_path: Path) -> None:
    """
    Plot a confusion matrix where the Y axis represents the design categories (Category column),
    and the X axis represents the binding combinations. The values are percentages of hits.
    The X-axis order is: single binding combinations, combined binding combinations (with "+"),
    and "Others" at the end. "binding" is removed from the labels for simplicity.

    :param data: DataFrame containing the binding columns and 'Category' column.
    """
    # Get all binding combinations present in the data (for the X-axis)
    binding_combinations = data["Binding Combination"].unique()

    # Sort the binding combinations: first no "+", then those with "+", and "Others" at the end
    single_bindings = sorted(
        [bc for bc in binding_combinations if "+" not in bc and bc != "Others"]
    )
    combined_bindings = sorted([bc for bc in binding_combinations if "+" in bc])
    binding_combinations = single_bindings + combined_bindings + ["Others"]

    # Initialize the confusion matrix DataFrame
    design_categories = data[
        "Category"
    ].unique()  # Ensure y-axis only contains design categories
    # Sprt the design categories: first no "+", then those with "+", and "Others" at the end
    single_categories = sorted(
        [bc for bc in design_categories if "+" not in bc and bc != "Others"]
    )
    combined_categories = sorted([bc for bc in design_categories if "+" in bc])
    design_categories = single_categories + combined_categories

    confusion_matrix = pd.DataFrame(
        index=design_categories, columns=binding_combinations, data=0
    )

    # Calculate percentage of hits for each combination in each category
    for category in confusion_matrix.index:
        category_data = data[data["Category"] == category]
        total_hits = len(category_data)

        if total_hits > 0:
            # Calculate the percentage for each binding combination
            combination_counts = category_data["Binding Combination"].value_counts()
            confusion_matrix.loc[category, combination_counts.index] = (
                combination_counts / total_hits
            ) * 100

    # Plot the confusion matrix as a heatmap with the viridis color map
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        cbar_kws={"label": "Percentage (%)"},
    )

    # Set the title and labels
    plt.title("% Functional Hits ")
    plt.xlabel("Function")
    plt.ylabel("Design Category")

    # Rotate the x-axis labels by 90 degrees and align them properly
    plt.xticks(rotation=90, ha="right")

    # Ensure tight layout to fit the labels
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.pdf")
    plt.show()


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Process PDB files and retrieve UniProt & GO terms."
    )
    parser.add_argument(
        "--input_pdb", required=True, type=Path, help="Folder containing PDB files"
    )
    parser.add_argument(
        "--slice",
        required=True,
        type=slice_arg,
        default=slice(None, 1),
        help="Slice for result['alignments'][0]",
    )
    args = parser.parse_args()
    csv_output_path = args.input_pdb / "combined_results.csv"
    if csv_output_path.exists():
        combined_df = pd.read_csv(csv_output_path)
    else:
        # Process PDB files
        combined_df = process_all_pdb_files(args.input_pdb, args.slice)
        # Ensure the binding columns are boolean
        binding_columns = [
            "DNA binding",
            "RNA binding",
            "ATP binding",
            "GTP binding",
            "metal ion binding",
        ]
        for col in binding_columns:
            combined_df[col] = combined_df[col].astype(bool)

        # Create 'Binding Combination' column if not already present
        def get_binding_combination(row):
            bindings = [
                binding.split(" ")[0].upper()
                if binding != "metal ion binding"
                else "Metal"
                for binding in binding_columns
                if row[binding]
            ]
            if bindings:
                return "+".join(bindings)
            else:
                return "Others"

        combined_df["Binding Combination"] = combined_df.apply(
            get_binding_combination, axis=1
        )
        combined_df.to_csv(csv_output_path, index=False)
        print("Processing complete. Results saved to 'combined_results.csv'.")

    # Plot 1: Percentage of hits with function per category
    plot_percentage_hits_with_function(combined_df, args.input_pdb)
    # Plot 2: Success rate per category
    plot_success_rate_per_category(combined_df, args.input_pdb)
    # Plot 3:
    plot_confusion_matrix(combined_df, args.input_pdb)


if __name__ == "__main__":
    main()
