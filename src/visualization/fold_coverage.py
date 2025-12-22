import argparse
import pickle
import typing as t
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import ampal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from ampal.amino_acids import polarity_Zimmerman, residue_charge
from scipy.stats import linregress, pearsonr, spearmanr
from tqdm import tqdm

from src.difference_fn.difference_processing import select_first_ampal_assembly
from src.fragments.fragments_classifier import EnsembleFragmentClassifier
from src.fragments.fragments_graph import StructureFragmentGraph
from src.training.data_processing.dataset import GraphCreator
import mdtraj as md  # Add this import to the top of your script
from src.fragments.fragments_classifier import FragmentDetail

# Mapping of classes to categories
classes = {
    1: "Mainly Alpha",
    2: "Mainly Beta",
    3: "Alpha Beta",
    4: "Few Structures/Special",
}

hbond_coverage = namedtuple(
    "HbondCoverage",
    [
        "within_fragments",
        "between_fragments",
    ],
)


def get_fragment(
    graph_creator: GraphCreator,
    pdb_file: Path,
    sequence: str,
    pdb_code: str,
    pdb_path: Path,
    check_sequence_length: bool = True,
) -> StructureFragmentGraph:
    fragment_path = pdb_path / pdb_code[1:3] / f"{pdb_code}.fg"
    if not fragment_path.exists():
        print(fragment_path)
        print(pdb_file)
        raise ValueError
        try:
            fragment_path = graph_creator.classify_and_save_graph(Path(pdb_file))
        except Exception as e:
            print(f"Error processing {fragment_path}: {e}")
            return None
    structure_fragment_graph = StructureFragmentGraph.load(fragment_path)
    if check_sequence_length:
        assert len(sequence) == len(
            structure_fragment_graph.structure_fragment.classification
        ), f"Length mismatch between sequence ({len(sequence)}) and classification ({len(structure_fragment_graph.structure_fragment.classification)})"

    return structure_fragment_graph


def load_pdbench(pdbench_path: Path) -> pd.DataFrame:
    pdbench = pd.read_csv(pdbench_path)
    pdbench["class_architecture"] = (
        pdbench["class"].astype(str) + "." + pdbench["architecture"].astype(str)
    )
    pdbench["Category"] = pdbench.apply(
        lambda row: classes.get(row["class"], "Few Structures/Special"), axis=1
    )
    return pdbench


def load_graph_creator(
    fragment_path: Path, workers: int, pdb_path: Path, verbose: bool = True
) -> GraphCreator:
    assert fragment_path.exists(), f"Fragment classifier not found: {fragment_path}"
    classifier = EnsembleFragmentClassifier(
        fragment_path,
        difference_names=["LogPr", "RamRmsd"],
        n_processes=workers,
    )
    graph_creator = GraphCreator(classifier, graph_dir=pdb_path, verbose=verbose)
    return graph_creator


def process_pdb_file(
    pdb_code: str, pdb_path: Path
) -> t.Dict[str, t.Union[Path, ampal.Assembly, str, torch.Tensor, torch.Tensor]]:
    pdb_file = pdb_path / f"{pdb_code}.pdb"
    assert pdb_file.exists(), f"PDB file not found: {pdb_file}"

    # Set cache_dir to be the same as pdb_path
    cache_file = pdb_path / f"{pdb_code}_dssp.pkl"

    # Check if cached DSSP results exist
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)
        return cache_data

    structure = ampal.load_pdb(pdb_file)
    structure = select_first_ampal_assembly(structure)
    sequence = structure.sequence

    # Calculate charge and polarity for each residue
    seq_charge = [residue_charge.get(aa, 0) for aa in sequence]
    seq_polarity = [-1 if polarity_Zimmerman.get(aa, 0) < 20 else 1 for aa in sequence]

    # Parse the structure using Bio.PDB
    p = PDBParser(QUIET=True)
    structure = p.get_structure(pdb_code, pdb_file)
    model = structure[0]
    # Run DSSP
    dssp = DSSP(model, pdb_file)

    seq_dssp = []
    seq_accessibility = []
    for dssp_res in dssp:
        _, _, ss, rel_acc, *_ = dssp_res
        seq_dssp.append(1 if ss in {"H", "B", "E", "G", "I", "T", "S"} else 0)
        try:
            rel_acc = float(rel_acc)
        except ValueError:
            rel_acc = 0.0
        seq_accessibility.append(rel_acc)

    # Load structure with MDTraj
    traj = md.load(pdb_file)

    # Select only the polypeptide atoms
    protein_atoms = traj.topology.select("protein")
    traj_protein = traj.atom_slice(protein_atoms)

    # Compute H-bonding via Kabsch-Sander
    hbond_matrices = md.kabsch_sander(traj_protein)[0]

    results_dict = {
        "pdb_code": pdb_code,
        "structure": structure,
        "sequence": sequence,
        "charge": seq_charge,
        "polarity": seq_polarity,
        "dssp": seq_dssp,
        "accessibility": seq_accessibility,
        "hbond_matrices": hbond_matrices,
    }

    # Ensure the lengths match (note: the assert on hbond_matrices might need to be adapted if it returns multiple matrices)
    assert len(sequence) == len(seq_dssp), "Length mismatch between sequence and DSSP"
    assert len(sequence) == len(
        seq_accessibility
    ), "Length mismatch between sequence and accessibility"
    assert (
        len(sequence) == hbond_matrices.shape[0]
    ), "Length mismatch between sequence and H-bond matrices"
    # Save results to cache
    with open(cache_file, "wb") as f:
        pickle.dump(results_dict, f)

    return results_dict


def calculate_overlap_for_continuous_property(
    fragment_map: t.List[int], property_map: t.List[float]
) -> float:
    """
    Calculate the overlap of indeces corresponding to property and the fragment classification

    ie. Union

    Parameters
    ----------
    fragment_map: t.List[int]
        Fragment classification map
    property_map: t.List[int]
        Property classification map, binary

    Returns
    -------
    property_coverage: float
        Overlap between the fragment and property classification
    """
    assert len(fragment_map) == len(
        property_map
    ), "Length mismatch between fragment and property maps"
    fragment_map_np = np.array(fragment_map)
    property_map_np = np.array(property_map)

    # Get the indices of non-zero elements in classification
    non_zero_indices = fragment_map_np != 0
    # Sum the properties corresponding to non-zero indices
    selected_sum = property_map_np[non_zero_indices].sum()
    # Calculate the total sum of property_map
    total_sum = property_map_np.sum()

    # Calculate the coverage
    property_coverage = selected_sum / total_sum
    return property_coverage


def calculate_overlap_for_binary_property(
    fragment_map: t.List[int], property_map: t.List[int]
) -> float:
    """
    Calculate the overlap of indeces corresponding to property and the fragment classification

    ie. Union

    Parameters
    ----------
    fragment_map: t.List[int]
        Fragment classification map
    property_map: t.List[int]
        Property classification map, binary

    Returns
    -------
    property_coverage: float
        Overlap between the fragment and property classification
    """
    assert len(fragment_map) == len(
        property_map
    ), "Length mismatch between fragment and property maps"
    # Get the indices of non-zero elements in classification
    classified_indices = set(np.nonzero(fragment_map)[0])
    # Property indices
    property_indices = set(np.nonzero(property_map)[0])
    # Property Coverage
    property_coverage = len(classified_indices & property_indices) / len(
        property_indices
    )

    return property_coverage


def calculate_overlap_for_interaction_property(
    fragment_classification_map: t.List[FragmentDetail], property_map: np.ndarray
) -> hbond_coverage:
    # Initialization
    total_sum = np.sum(property_map)
    hbonds_within_fragments = []
    hbonds_within_non_fragments = []
    fragment_idxs = []
    non_fragment_idxs = []

    # Single loop to handle all fragment details
    for fragment_detail in fragment_classification_map:
        start, end = fragment_detail.start_idx, fragment_detail.end_idx
        fragment_property = property_map[start:end, start:end]
        total_hbond = np.sum(fragment_property)

        if fragment_detail.fragment_class > 0:
            hbonds_within_fragments.append(total_hbond)
            fragment_idxs.extend(range(start, end))
        else:
            hbonds_within_non_fragments.append(total_hbond)
            non_fragment_idxs.extend(range(start, end))

    # Calculating coverage within fragments and non-fragments
    total_within_fragments = np.sum(hbonds_within_fragments)
    total_within_non_fragments = np.sum(hbonds_within_non_fragments)
    hbond_within_coverage = total_within_fragments / (
        total_within_fragments + total_within_non_fragments
    )
    # hbond_within_units_coverage = (total_within_fragments+total_within_non_fragments) / total_sum
    # Calculating between fragments and non-fragments
    hbond_between_fragments = (
        np.sum(property_map[fragment_idxs]) - total_within_fragments
    )
    hbond_between_fragments_coverage = hbond_between_fragments / (total_sum - total_within_fragments - total_within_non_fragments)

    return hbond_coverage(
        within_fragments=hbond_within_coverage,
        between_fragments=hbond_between_fragments_coverage,
    )




def plot_coverage(
    pdbench: pd.DataFrame, coverage_column: str, output_path: str, title: str
):
    plt.figure(figsize=(12, 6))
    coverage_stats = (
        pdbench.groupby(["Fold", "Category", "class_architecture"])[coverage_column]
        .agg(["mean", "std"])
        .reset_index()
    )
    coverage_stats = coverage_stats.sort_values(by="class_architecture")

    # Use Seaborn's color palette
    seaborn_palette = sns.color_palette("deep", len(classes) * 2)
    palette = [seaborn_palette[i] for i in [3, 0, 4, 7]]
    # Map categories to colors
    color_mapping = {
        category: palette[i] for i, category in enumerate(classes.values())
    }

    # Plot the bars in the order of the sorted class_architecture
    bars = plt.bar(
        coverage_stats["Fold"],
        coverage_stats["mean"],
        yerr=coverage_stats["std"],
        capsize=5,
        color=[color_mapping[cat] for cat in coverage_stats["Category"]],
        alpha=0.8,
    )

    # Set y-axis limit to 1
    plt.ylim(0, 1)

    # Place the x tick labels starting from the x-axis (bottom of the bars)
    for bar, fold_label in zip(bars, coverage_stats["Fold"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            0.01,
            fold_label,
            ha="center",
            va="bottom",
            fontsize=12,
            rotation=90,
            color="black",
        )

    # Hide the original x-axis ticks
    plt.xticks([])

    # Increase font size of labels and title
    plt.xlabel("Fold", fontsize=18)
    plt.ylabel(f"% {coverage_column.replace('_', ' ').capitalize()}", fontsize=18)
    plt.title(title, fontsize=16)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_mapping[category])
        for category in classes.values()
    ]
    labels = classes.values()
    plt.legend(
        handles,
        labels,
        title="Class",
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
    )

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # Save the plot as a PDF file
    plt.savefig(output_path, format="pdf")


def plot_coverage_vs_resolution(pdbench: pd.DataFrame, output_path: str):
    plt.figure(figsize=(25, 5))
    categories = pdbench["Category"].unique()

    for i, category in enumerate(categories):
        subset = pdbench[pdbench["Category"] == category].dropna(
            subset=["resolution", "fold_coverage"]
        )
        plt.subplot(1, len(categories) + 1, i + 1)
        plt.scatter(
            subset["resolution"],
            subset["fold_coverage"],
            label=f"{category} (n={len(subset)})",
            alpha=0.7,
        )
        if len(subset) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(
                subset["resolution"], subset["fold_coverage"]
            )
            plt.plot(
                subset["resolution"],
                intercept + slope * subset["resolution"],
                color="red",
                linestyle="-",
                linewidth=2,
            )

            # Pearson correlation
            pearson_corr, pearson_p = pearsonr(
                subset["resolution"], subset["fold_coverage"]
            )
            # Spearman correlation
            spearman_corr, spearman_p = spearmanr(
                subset["resolution"], subset["fold_coverage"]
            )
            title = f"{category}\nPearson: {pearson_corr:.2f} (p={pearson_p:.1e})\nSpearman: {spearman_corr:.2f} (p={spearman_p:.1e})"
        else:
            title = f"{category}\nInsufficient data"

        plt.xlabel("Resolution (Å)", fontsize=18)
        plt.ylabel("Coverage (%)", fontsize=18)
        plt.title(title, fontsize=18)
        plt.ylim(0, 1)

    overall_subset = pdbench.dropna(subset=["resolution", "fold_coverage"])
    plt.subplot(1, len(categories) + 1, len(categories) + 1)
    plt.scatter(
        overall_subset["resolution"],
        overall_subset["fold_coverage"],
        label="Overall",
        alpha=0.7,
    )
    if len(overall_subset) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            overall_subset["resolution"], overall_subset["fold_coverage"]
        )
        plt.plot(
            overall_subset["resolution"],
            intercept + slope * overall_subset["resolution"],
            color="red",
            linestyle="-",
            linewidth=2,
        )

        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(
            overall_subset["resolution"], overall_subset["fold_coverage"]
        )
        # Spearman correlation
        spearman_corr, spearman_p = spearmanr(
            overall_subset["resolution"], overall_subset["fold_coverage"]
        )
        title = f"Overall\nPearson: {pearson_corr:.2f} (p={pearson_p:.1e})\nSpearman: {spearman_corr:.2f} (p={spearman_p:.1e})"
    else:
        title = "Overall\nInsufficient data"

    plt.xlabel("Resolution (Å)", fontsize=18)
    plt.ylabel("Coverage (%)", fontsize=18)
    plt.title(title, fontsize=18)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf")
    plt.close()


def plot_coverage_scatter(
    subset: pd.DataFrame,
    x_column: str,
    y_column: str,
    category: str,
    ylabel: str,
    title_suffix: str,
    ax: plt.Axes,
):
    ax.scatter(
        subset[x_column],
        subset[y_column],
        label=f"{category} - {title_suffix} (n={len(subset)})",
        alpha=0.7,
    )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)

    if len(subset) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            subset[x_column], subset[y_column]
        )
        ax.plot(
            subset[x_column],
            intercept + slope * subset[x_column],
            color="red",
            linestyle="-",
            linewidth=2,
        )

        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(subset[x_column], subset[y_column])
        # Spearman correlation
        spearman_corr, spearman_p = spearmanr(subset[x_column], subset[y_column])
        title = f"{category} - {title_suffix}\nPearson: {pearson_corr:.2f} (p={pearson_p:.1e})\nSpearman: {spearman_corr:.2f} (p={spearman_p:.1e})"
    else:
        title = f"{category} - {title_suffix}\nInsufficient data"

    ax.set_xlabel("Fold Coverage (%)", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=18)
    ax.set_ylim(0, 1)
    plt.close()



def plot_coverage_vs_fold_coverage(
    pdbench: pd.DataFrame,
    coverage_column: str,
    ylabel: str,
    title_suffix: str,
    output_path: str,
):
    plt.figure(figsize=(25, 5))
    categories = pdbench["Category"].unique()

    for i, category in enumerate(categories):
        subset = pdbench[pdbench["Category"] == category].dropna(
            subset=["fold_coverage", coverage_column]
        )

        plt.subplot(1, len(categories) + 1, i + 1)
        plt.scatter(
            subset["fold_coverage"],
            subset[coverage_column],
            label=f"{category} (n={len(subset)})",
            alpha=0.7,
        )
        plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)

        if len(subset) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(
                subset["fold_coverage"], subset[coverage_column]
            )
            plt.plot(
                subset["fold_coverage"],
                intercept + slope * subset["fold_coverage"],
                color="red",
                linestyle="-",
                linewidth=2,
            )

            # Pearson correlation
            pearson_corr, pearson_p = pearsonr(
                subset["fold_coverage"], subset[coverage_column]
            )
            # Spearman correlation
            spearman_corr, spearman_p = spearmanr(
                subset["fold_coverage"], subset[coverage_column]
            )
            title = f"{category}\nPearson: {pearson_corr:.2f} (p={pearson_p:.1e})\nSpearman: {spearman_corr:.2f} (p={spearman_p:.1e})"
        else:
            title = f"{category}\nInsufficient data"

        plt.xlabel("Fold Coverage (%)", fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.title(title, fontsize=18)
        plt.ylim(0, 1)

    overall_subset = pdbench.dropna(subset=["fold_coverage", coverage_column])
    plt.subplot(1, len(categories) + 1, len(categories) + 1)
    plt.scatter(
        overall_subset["fold_coverage"],
        overall_subset[coverage_column],
        label="Overall",
        alpha=0.7,
    )
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)

    if len(overall_subset) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            overall_subset["fold_coverage"], overall_subset[coverage_column]
        )
        plt.plot(
            overall_subset["fold_coverage"],
            intercept + slope * overall_subset["fold_coverage"],
            color="red",
            linestyle="-",
            linewidth=2,
        )

        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(
            overall_subset["fold_coverage"], overall_subset[coverage_column]
        )
        # Spearman correlation
        spearman_corr, spearman_p = spearmanr(
            overall_subset["fold_coverage"], overall_subset[coverage_column]
        )
        title = f"Overall\nPearson: {pearson_corr:.2f} (p={pearson_p:.1e})\nSpearman: {spearman_corr:.2f} (p={spearman_p:.1e})"
    else:
        title = "Overall\nInsufficient data"

    plt.xlabel("Fold Coverage (%)", fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=18)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf")
    plt.close()


def plot_ratio_coverage(
    pdbench: pd.DataFrame, property_column: str, output_path: str, title: str
):
    plt.figure(figsize=(12, 6))

    # Calculate the ratio (% property coverage / % fold coverage)
    pdbench["coverage_ratio"] = pdbench[property_column] / pdbench["fold_coverage"]

    coverage_stats = (
        pdbench.groupby(["Fold", "Category", "class_architecture"])["coverage_ratio"]
        .agg(["mean", "std"])
        .reset_index()
    )
    coverage_stats = coverage_stats.sort_values(by="class_architecture")

    # Use Seaborn's color palette
    seaborn_palette = sns.color_palette("deep", len(classes) * 2)
    palette = [seaborn_palette[i] for i in [3, 0, 4, 7]]
    # Map categories to colors
    color_mapping = {
        category: palette[i] for i, category in enumerate(classes.values())
    }

    # Plot the bars in the order of the sorted class_architecture
    bars = plt.bar(
        coverage_stats["Fold"],
        coverage_stats["mean"],
        yerr=coverage_stats["std"],
        capsize=5,
        color=[color_mapping[cat] for cat in coverage_stats["Category"]],
        alpha=0.8,
    )

    # Set y-axis limit to a reasonable range
    plt.ylim(0, 2)  # Adjust the upper limit based on your expected range

    # Place the x tick labels starting from the x-axis (bottom of the bars)
    for bar, fold_label in zip(bars, coverage_stats["Fold"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            0.05,
            f"{fold_label}",
            ha="center",
            va="bottom",
            fontsize=12,
            rotation=90,
            color="black",
        )

    # Hide the original x-axis ticks
    plt.xticks([])

    # Increase font size of labels and title
    plt.xlabel("Fold", fontsize=18)
    plt.ylabel(
        f"{property_column.replace('_', ' ').capitalize()} / Fold Coverage", fontsize=18
    )
    plt.title(title, fontsize=16)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_mapping[category])
        for category in classes.values()
    ]
    labels = classes.values()
    plt.legend(
        handles,
        labels,
        title="Class",
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
    )

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # Save the plot as a PDF file
    plt.savefig(output_path, format="pdf")
    plt.show()


def main(args):
    args.pdbench_path = Path(args.pdbench_path)
    args.pdb_path = Path(args.pdb_path)
    args.fragment_path = Path(args.fragment_path)
    assert args.pdbench_path.exists(), f"PDBench file not found: {args.pdbench_path}"
    assert args.pdb_path.exists(), f"PDB path not found: {args.pdb_path}"
    assert (
        args.fragment_path.exists()
    ), f"Fragment classifier not found: {args.fragment_path}"

    pdbench = load_pdbench(args.pdbench_path)
    graph_creator = load_graph_creator(
        args.fragment_path, args.workers, args.pdb_path
    )

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_pdb_file, pdb_code, args.pdb_path): pdb_code
            for pdb_code in pdbench["PDB"]
        }
        # Initialize tqdm with the total number of PDB files
        with tqdm(total=len(futures), desc="Processing PDB files") as pbar:
            for future in as_completed(futures):
                pdb_code = futures[future]
                try:
                    results_dict = future.result()
                    if results_dict is None:
                        print(f"Error processing {pdb_code}")
                        continue

                    structure_fragment_graph = get_fragment(
                        graph_creator,
                        results_dict["pdb_code"],
                        results_dict["sequence"],
                        pdb_code,
                        args.pdb_path,
                    )

                    # Skip fragment if it is not found
                    if structure_fragment_graph is None:
                        continue

                    # Calculate coverage metrics
                    fragment_classification = (
                        structure_fragment_graph.structure_fragment.classification
                    )

                    # Fold coverage
                    classified_indices = set(np.nonzero(fragment_classification)[0])
                    fold_coverage = len(classified_indices) / len(
                        fragment_classification
                    )

                    # Charge coverage
                    charge_coverage_val = calculate_overlap_for_binary_property(
                        fragment_classification, results_dict["charge"]
                    )

                    # Polarity coverage
                    polarity_coverage_val = calculate_overlap_for_binary_property(
                        fragment_classification, results_dict["polarity"]
                    )

                    # DSSP coverage
                    dssp_coverage = calculate_overlap_for_binary_property(
                        fragment_classification, results_dict["dssp"]
                    )

                    # Accessiblity coverage
                    accessiblity_coverage = calculate_overlap_for_continuous_property(
                        fragment_classification, results_dict["accessibility"]
                    )

                    (
                        hbond_within_fragments,
                        hbond_between_fragments,
                    ) = calculate_overlap_for_interaction_property(
                        structure_fragment_graph.structure_fragment.classification_map,
                        results_dict["hbond_matrices"],
                    )

                    # Update pdbench DataFrame
                    pdbench.loc[
                        pdbench["PDB"] == pdb_code, "fold_coverage"
                    ] = fold_coverage
                    pdbench.loc[
                        pdbench["PDB"] == pdb_code, "charge_coverage"
                    ] = charge_coverage_val
                    pdbench.loc[
                        pdbench["PDB"] == pdb_code, "polarity_coverage"
                    ] = polarity_coverage_val
                    pdbench.loc[
                        pdbench["PDB"] == pdb_code, "dssp_coverage"
                    ] = dssp_coverage
                    pdbench.loc[
                        pdbench["PDB"] == pdb_code, "accessibility_coverage"
                    ] = accessiblity_coverage
                    pdbench.loc[
                        pdbench["PDB"] == pdb_code, "hbond_within_fragments"
                    ] = hbond_within_fragments
                    pdbench.loc[
                        pdbench["PDB"] == pdb_code, "hbond_between_fragments"
                    ] = hbond_between_fragments

                except Exception as exc:
                    print(f"PDB code {pdb_code} generated an exception: {exc}")
                finally:
                    # Update progress bar for each completed task
                    pbar.update(1)

    plot_coverage(
        pdbench,
        "fold_coverage",
        "coverage_by_fold.pdf",
        "Coverage by Fold",
    )
    plot_coverage(
        pdbench,
        "charge_coverage",
        "coverage_by_charge.pdf",
        "Charge Coverage by Fold",
    )
    plot_coverage(
        pdbench,
        "polarity_coverage",
        "coverage_by_polarity.pdf",
        "Polarity Coverage by Fold",
    )
    plot_coverage(
        pdbench,
        "dssp_coverage",
        "coverage_by_secondary_structure.pdf",
        "Secondary Structure Coverage by Fold",
    )
    plot_coverage(
        pdbench,
        "accessibility_coverage",
        "coverage_by_accessibility.pdf",
        "Accessibility Coverage by Fold",
    )
    plot_coverage(
        pdbench,
        "hbond_within_fragments",
        "coverage_by_hbond_within_fragments.pdf",
        "Hbond Within Fragments Coverage by Fold",
    )
    plot_coverage(
        pdbench,
        "hbond_between_fragments",
        "hbond_between_fragments_coverage_by_fold.pdf",
        "Hbond Between Fragments Coverage by Fold",
    )

    plot_coverage_vs_resolution(pdbench, "coverage_vs_resolution.pdf")
    plot_coverage_vs_fold_coverage(
        pdbench,
        "charge_coverage",
        "Charge Coverage (%)",
        "Charge",
        "charge_vs_fold_coverage.pdf",
    )
    plot_coverage_vs_fold_coverage(
        pdbench,
        "polarity_coverage",
        "Polarity Coverage (%)",
        "Polarity",
        "polarity_vs_fold_coverage.pdf",
    )
    plot_coverage_vs_fold_coverage(
        pdbench,
        "dssp_coverage",
        "Secondary Structure Coverage (%)",
        "Secondary Structure",
        "secondary_structure_vs_fold_coverage.pdf",
    )
    plot_coverage_vs_fold_coverage(
        pdbench,
        "accessibility_coverage",
        "Accessibility Coverage (%)",
        "Accessibility",
        "accessibility_vs_fold_coverage.pdf",
    )
    plot_coverage_vs_fold_coverage(
        pdbench,
        "hbond_within_fragments",
        "Hbond Within Fragments Coverage (%)",
        "Hbond Within Fragments",
        "hbond_within_fragments_vs_fold_coverage.pdf",
    )

    plot_coverage_vs_fold_coverage(
        pdbench,
        "hbond_between_fragments",
        "Hbond Between Fragments Coverage (%)",
        "Hbond Between Fragments",
        "hbond_between_fragments_vs_fold_coverage.pdf",
    )


    plot_ratio_coverage(
        pdbench,
        "charge_coverage",
        "charge_coverage_ratio_by_fold.pdf",
        "Charge Coverage Ratio by Fold",
    )

    plot_ratio_coverage(
        pdbench,
        "polarity_coverage",
        "polarity_coverage_ratio_by_fold.pdf",
        "Polarity Coverage Ratio by Fold",
    )
    plot_ratio_coverage(
        pdbench,
        "dssp_coverage",
        "dssp_coverage_ratio_by_fold.pdf",
        "Secondary Structure Coverage Ratio by Fold",
    )
    plot_ratio_coverage(
        pdbench,
        "accessibility_coverage",
        "accessibility_coverage_ratio_by_fold.pdf",
        "Accessibility Coverage Ratio by Fold",
    )
    plot_ratio_coverage(
        pdbench,
        "hbond_within_fragments",
        "hbond_within_fragments_coverage_ratio_by_fold.pdf",
        "Hbond Within Fragments Coverage Ratio by Fold",
    )
    plot_ratio_coverage(
        pdbench,
        "hbond_between_fragments",
        "hbond_between_fragments_coverage_ratio_by_fold.pdf",
        "Hbond Between Fragments Coverage Ratio by Fold",
    )
    # Save the updated PDBench file
    pdbench.to_csv("pdbench_coverage.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--pdbench_path", type=Path, help="Path to input file")
    parser.add_argument(
        "--pdb_path", type=Path, help="Path to PDB files with chains extracted"
    )
    parser.add_argument(
        "--fragment_path", type=Path, required=True, help="Path to fragment classifier"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=9,
        help="Number of workers to use for parallel processing",
    )
    params = parser.parse_args()
    main(params)
