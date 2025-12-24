import argparse
import logging
import os
from collections import Counter
from itertools import combinations
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Dict, List, Tuple, Callable
from Bio.Align import substitution_matrices, PairwiseAligner
import ampal
import gmatch4py as gm
import numpy as np
import pandas as pd
import plotly.express as px
import umap
from adjustText import adjust_text
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from tqdm import tqdm

from tessera.difference_fn.shape_difference import RmsdBiopythonStrategy
from tessera.fragments.classification_config import selected_pdbs
from tessera.fragments.fragments_graph import StructureFragmentGraph
import math

from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score


def evaluate_dimensionality_reduction_quality(
    distance_matrix: pd.DataFrame, embedding: np.ndarray, n_clusters: int
) -> Dict[str, float]:
    """
    Evaluate how well the dimensionality reduction preserves the original data structure.
    Computes:
    - Trustworthiness (local neighborhood preservation)
    - Spearman correlation between original distances and distances in reduced space
    """
    # Compute trustworthiness
    n = embedding.shape[0]
    k = n_clusters
    distance_matrix.values[:] = np.clip(distance_matrix.values, 0, None)
    trust = trustworthiness(
        distance_matrix.values, embedding, n_neighbors=k, metric="precomputed"
    )

    # Compute distances in embedding space
    emb_dist = np.sqrt(
        ((embedding[:, None, :] - embedding[None, :, :]) ** 2).sum(axis=2)
    )
    # Flatten upper triangle of distance matrices for correlation
    idx = np.triu_indices(n, k=1)
    original_distances = distance_matrix.values[idx]
    reduced_distances = emb_dist[idx]
    # Compute Spearman correlation
    corr, pvalue = spearmanr(original_distances, reduced_distances)

    return {"Trustworthiness": trust, "Spearman_Distance_Correlation": (corr, pvalue)}


def plot_single_pair_triplet_violin(
    embedding: np.ndarray, categories: List[str], method_name: str, output_file: Path
) -> None:
    """
    Plot all single, pair, and triplet categories in one figure with two subplots (Dim0 and Dim1).
    Each category's points are plotted along x-axis, grouped by single/pair/triplet category type.
    Uses different colors for single, pair, and triplet groups.
    Rotates x-axis labels by 90 degrees.
    """
    # Determine group type
    def get_group_type(cat: str) -> str:
        plus_count = cat.count("+")
        if plus_count == 0:
            return "Single"
        elif plus_count == 1:
            return "Pair"
        else:
            return "Triplet"

    df = pd.DataFrame(
        {"Category": categories, "Dim0": embedding[:, 0], "Dim1": embedding[:, 1]}
    )
    df["Group"] = df["Category"].apply(get_group_type)

    # Sort categories within each group
    single_cats = sorted(df.loc[df["Group"] == "Single", "Category"].unique())
    pair_cats = sorted(df.loc[df["Group"] == "Pair", "Category"].unique())
    triplet_cats = sorted(df.loc[df["Group"] == "Triplet", "Category"].unique())

    # Create a categorical order that puts all singles, then pairs, then triplets
    cat_order = single_cats + pair_cats + triplet_cats

    # Map groups to colors
    group_colors = {"Single": "skyblue", "Pair": "lightgreen", "Triplet": "salmon"}

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

    # Dim0 violin plot
    sns.violinplot(
        x="Category",
        y="Dim0",
        data=df,
        order=cat_order,
        ax=axes[0],
        cut=0,
        inner="box",
        palette=[group_colors[get_group_type(c)] for c in cat_order],
    )
    axes[0].set_title(f"{method_name} Dimension 0")

    # Dim1 violin plot
    sns.violinplot(
        x="Category",
        y="Dim1",
        data=df,
        order=cat_order,
        ax=axes[1],
        cut=0,
        inner="box",
        palette=[group_colors[get_group_type(c)] for c in cat_order],
    )
    axes[1].set_title(f"{method_name} Dimension 1")

    # Rotate x labels
    for ax in axes:
        ax.tick_params(axis="x", rotation=90)

    # Add legend manually (one entry for each group)
    legend_patches = [
        Patch(color=color, label=grp) for grp, color in group_colors.items()
    ]
    plt.legend(handles=legend_patches, title="Group Type", loc="upper right")

    plt.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def plot_dim_reduction_violin(
    embedding: np.ndarray, categories: List[str], method_name: str, output_file: Path
) -> None:
    """
    Create a figure with 2 violin subplots for dimension 0 and dimension 1.
    X-axis: category, Y-axis: dimension value.
    """
    df = pd.DataFrame(
        {"Category": categories, "Dim0": embedding[:, 0], "Dim1": embedding[:, 1]}
    )

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 8), sharex=True)
    sns.violinplot(x="Category", y="Dim0", data=df, ax=axes[0], cut=0, inner="box")
    sns.violinplot(x="Category", y="Dim1", data=df, ax=axes[1], cut=0, inner="box")
    axes[0].set_title(f"{method_name} Dimension 0")
    axes[1].set_title(f"{method_name} Dimension 1")
    # Rotate x-axis labels for better readability
    for ax in axes:
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=90, horizontalalignment="center"
        )
    plt.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def find_fg_file(input_graph: Path, pdb_code: str, category: str) -> Path:
    # Extract the first two letters of the PDB code for the subdirectory
    subdir = pdb_code[1:3].upper()
    fg_file = input_graph / subdir / f"{pdb_code}_{category}.fg"
    return fg_file


def parse_pdb_file_tree(
    input_pdb: Path, input_graph: Path
) -> Dict[str, Dict[str, Path]]:
    pdb_data = {}
    # Walk through the directory tree
    for category in input_pdb.iterdir():
        if category.is_dir():
            for pdb_file in category.glob("*.pdb"):
                pdb_code = pdb_file.stem[:4]
                if pdb_code not in selected_pdbs:
                    continue
                category_name = category.name
                fg_path = find_fg_file(input_graph, pdb_code, category_name)
                assert fg_path.exists(), f"FG file {fg_path} does not exist"
                pdb_data[pdb_code] = {
                    "category": category_name,
                    "pdb_path": pdb_file,
                    "fg_path": fg_path,
                }

    return pdb_data


def append_to_csv(file_path: str, rows: list):
    """Append rows to a CSV file."""
    header = not os.path.exists(file_path)  # Write header only if file doesn't exist
    pd.DataFrame(rows, columns=["PDB1", "PDB2", "RMSD"]).to_csv(
        file_path, mode="a", index=False, header=header
    )


def filter_remaining_pairs(pairs, checkpoint_file):
    """Filter out pairs already present in the checkpoint file."""
    if os.path.exists(checkpoint_file):
        existing_results = pd.read_csv(checkpoint_file)
        existing_pairs = set(zip(existing_results["PDB1"], existing_results["PDB2"]))
        return [pair for pair in pairs if pair not in existing_pairs]
    return pairs


def get_sorted_pdb_keys(pdb_data: Dict[str, Dict[str, Path]]) -> List[str]:
    """
    Returns sorted PDB keys based on category and PDB name.
    Handles missing or inconsistent categories gracefully.
    """
    return sorted(
        pdb_data.keys(),
        key=lambda x: (pdb_data[x].get("category", "unknown").lower(), x),
    )


def prepare_pairs_for_computation(
    pdb_data: Dict[str, Dict[str, Path]], checkpoint_file: str
):
    """Prepare pairs for RMSD computation, filtering out already computed pairs."""
    pdb_keys = get_sorted_pdb_keys(pdb_data)
    pairs = list(combinations(pdb_keys, 2))
    return filter_remaining_pairs(pairs, checkpoint_file)


def compute_rmsd(pair: tuple, preloaded_pdbs: Dict[str, ampal.Assembly]) -> tuple:
    """Compute RMSD using preloaded PDB assemblies."""
    pdb1, pdb2 = pair
    print(pdb1, pdb2)
    try:
        rmsd = RmsdBiopythonStrategy.biopython_calculate_rmsd(
            preloaded_pdbs[pdb1], preloaded_pdbs[pdb2]
        )
        print("Success")
        return pdb1, pdb2, rmsd
    except Exception as e:
        logging.error(f"Failed to compute RMSD for pair {pdb1}, {pdb2}: {e}")
        return pdb1, pdb2, np.inf


def compute_rmsd_wrapper(args):
    """Wrapper for RMSD computation to use in multiprocessing."""
    pair, preloaded_pdbs = args
    return compute_rmsd(pair, preloaded_pdbs)


# Setup logging
logging.basicConfig(
    filename="timeout_log.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def preload_pdb_files(
    pdb_data: Dict[str, Dict[str, Path]]
) -> Dict[str, ampal.Assembly]:
    """Preload all PDB files into memory."""
    preloaded_pdbs = {}
    for pdb_code, data in pdb_data.items():
        try:
            preloaded_pdbs[pdb_code] = ampal.load_pdb(data["pdb_path"])
            logging.info(f"Preloaded {pdb_code}")
        except Exception as e:
            logging.error(f"Failed to preload {pdb_code}: {e}")
    return preloaded_pdbs


def create_RMSD_distance_matrix_parallel(
    pdb_data: Dict[str, Dict[str, Path]],
    checkpoint_file: str = "rmsd_checkpoint.csv",
    num_workers: int = 2,
) -> pd.DataFrame:
    """Create RMSD distance matrix with checkpointing and multiprocessing."""
    # Prepare pairs, filtering out already computed ones
    pairs = prepare_pairs_for_computation(pdb_data, checkpoint_file)

    # Preload PDB files using a multiprocessing manager for shared memory
    manager = Manager()
    preloaded_pdbs = manager.dict(preload_pdb_files(pdb_data))

    # Prepare arguments for multiprocessing
    args_list = [(pair, preloaded_pdbs) for pair in pairs]

    # Multiprocessing pool
    with Pool(processes=num_workers) as pool:
        results = []
        for result in tqdm(
            pool.imap_unordered(compute_rmsd_wrapper, args_list), total=len(args_list)
        ):
            if result:
                results.append(result)

            # Save results incrementally
            if len(results) >= 500:  # Save every 500 results
                append_to_csv(checkpoint_file, results)
                results = []  # Reset the buffer

    # Save any remaining results
    if results:
        append_to_csv(checkpoint_file, results)

    # Combine all results into a matrix
    all_results = pd.read_csv(checkpoint_file)
    pdb_keys = sorted(pdb_data.keys())
    rmsd_matrix = pd.DataFrame(np.inf, index=pdb_keys, columns=pdb_keys)
    for _, row in all_results.iterrows():
        pdb1, pdb2, rmsd = row["PDB1"], row["PDB2"], row["RMSD"]
        rmsd_matrix.loc[pdb1, pdb2] = rmsd
        rmsd_matrix.loc[pdb2, pdb1] = rmsd

    return rmsd_matrix


def create_graph_distance_matrices(
    pdb_data: Dict[str, Dict[str, Path]], metrics_list: List[object]
) -> Dict[str, pd.DataFrame]:
    # Sort PDB Keys by function and then by PDB code
    pdb_keys = get_sorted_pdb_keys(pdb_data)
    # Extract all gragments to list
    nx_graphs = []
    for pdb in pdb_keys:
        fragment_graph = StructureFragmentGraph.load(pdb_data[pdb]["fg_path"]).graph
        # Iterate through nodes and convert fragment_class to str
        for node in fragment_graph.nodes(data=True):
            node[1]["fragment_class"] = str(node[1]["fragment_class"])
        # Iterate through edges and convert peptide_bond to str
        for edge in fragment_graph.edges(data=True):
            edge[2]["peptide_bond"] = str(edge[2]["peptide_bond"])
        nx_graphs.append(fragment_graph)
    distance_matrix_dict = {}
    for class_ in metrics_list:
        # Initialize comparator with appropriate parameters
        if class_ in (gm.GraphEditDistance, gm.GreedyEditDistance):
            comparator = class_(1, 1, 1, 1)  # All edit costs are equal to 1
        elif class_ == gm.WeisfeleirLehmanKernel:
            comparator = class_(h=2)
        else:
            comparator = class_()
        # Set attributes used in comparison
        comparator.set_attr_graph_used(
            node_attr_key="fragment_class", edge_attr_key="peptide_bond"
        )
        # Compute distance or similarity matrix
        result = comparator.compare(nx_graphs, None)
        distance_matrix = comparator.distance(result)
        distance_matrix = np.array(distance_matrix)

        # Convert to pandas DataFrame with keys as index and columns
        distance_matrix = pd.DataFrame(
            distance_matrix, index=pdb_keys, columns=pdb_keys
        )
        # Convert class to string
        class_name = class_.__name__
        distance_matrix_dict[class_name] = distance_matrix

    return distance_matrix_dict


def plot_distance_matrix_with_categories(
    distance_matrix: pd.DataFrame,
    pdb_data: Dict[str, Dict[str, Path]],
    output_file: str,
):
    """
    Plots a large distance matrix with fold category labels and highlights self-comparisons along the diagonal.
    Adds gridlines to separate blocks for each category.

    Args:
        distance_matrix (pd.DataFrame): The full distance matrix.
        pdb_data (dict): Dictionary containing PDB data with categories.
        output_file (str): Path to save the plot.
    """
    # Create a mapping from PDB codes to categories
    pdb_to_category = {pdb: data["category"] for pdb, data in pdb_data.items()}
    category_labels = [pdb_to_category[pdb] for pdb in distance_matrix.index]

    # Get unique category labels and their positions
    unique_categories = []
    category_positions = []
    for i, label in enumerate(category_labels):
        if label not in unique_categories:
            unique_categories.append(label)
            category_positions.append(i)

    # Create a mapping from categories to start/end indices
    category_ranges = {}
    for i, label in enumerate(category_labels):
        if label not in category_ranges:
            category_ranges[label] = [i, i]
        else:
            category_ranges[label][1] = i

    # Plot the matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(distance_matrix.values.astype(float), cmap="viridis_r", aspect="auto")
    plt.colorbar(label="Distance")
    plt.title("Distance Matrix with Fold Categories")

    # Add gridlines to separate blocks
    ax = plt.gca()
    for category, (start, end) in category_ranges.items():
        # Vertical and horizontal gridlines
        ax.axhline(start - 0.5, color="white", linewidth=1.5)
        ax.axvline(start - 0.5, color="white", linewidth=1.5)
        ax.axhline(end + 0.5, color="white", linewidth=1.5)
        ax.axvline(end + 0.5, color="white", linewidth=1.5)

    # Add unique category labels to the axes
    plt.xticks(category_positions, unique_categories, rotation=90, fontsize=8)
    plt.yticks(category_positions, unique_categories, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)  # High DPI for clarity
    plt.close()


def match_clusters_to_labels(true_labels, predicted_clusters):
    """
    Match cluster labels to truth labels using maximum likelihood.

    Args:
        true_labels (List[str]): Ground truth category labels.
        predicted_clusters (List[int]): Predicted cluster labels.

    Returns:
        np.ndarray: Matched predicted labels.
    """
    # Sort true labels and map to integers
    unique_labels = np.sort(np.unique(true_labels))  # Ensure sorted order
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    true_labels_int = np.array([label_to_int[label] for label in true_labels])

    # Sort cluster labels
    unique_clusters = np.sort(np.unique(predicted_clusters))

    # Initialize array for matched labels
    matched_labels = np.empty_like(predicted_clusters)

    # Match clusters to truth labels
    for cluster in unique_clusters:
        mask = predicted_clusters == cluster
        true_counts = np.bincount(true_labels_int[mask])
        matched_labels[mask] = np.argmax(true_counts)

    return matched_labels


def perform_hierarchical_clustering(
    distance_matrix: pd.DataFrame,
    pdb_data: Dict[str, Dict[str, Path]],
    output_file: str,
):
    """
    Perform hierarchical clustering and plot a horizontally oriented dendrogram with proteins
    labeled and color-coded by functionality.

    Args:
        distance_matrix (pd.DataFrame): The distance matrix to cluster.
        pdb_data (dict): PDB metadata with categories.
        output_file (str): Path to save the dendrogram plot.
    """
    # Ensure the distance matrix is symmetric
    distance_matrix.values[:] = (distance_matrix.values + distance_matrix.values.T) / 2

    # Ensure diagonal is exactly zero
    np.fill_diagonal(distance_matrix.values, 0)

    # Clip any small negative values caused by floating-point errors
    distance_matrix.values[:] = np.clip(distance_matrix.values, 0, None)

    # Convert the distance matrix to condensed form
    condensed_distance_matrix = squareform(distance_matrix.values)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance_matrix, method="average")

    # Extract protein categories and append to labels
    categories = [pdb_data[pdb]["category"] for pdb in distance_matrix.index]
    labeled_proteins = [
        f"{pdb} ({category})"
        for pdb, category in zip(distance_matrix.index, categories)
    ]

    # Create a color palette for functionality
    unique_categories = sorted(set(categories))
    palette = sns.color_palette("hsv", len(unique_categories))
    category_colors = {cat: palette[i] for i, cat in enumerate(unique_categories)}

    # Map colors to each label
    label_colors = [category_colors[category] for category in categories]

    # Dynamically adjust figure size based on the number of proteins
    figure_height = max(10, len(labeled_proteins) * 0.2)

    # Plot horizontally-oriented dendrogram
    plt.figure(figsize=(12, figure_height))  # Increase height for readability
    dendrogram(
        linkage_matrix,
        labels=labeled_proteins,
        leaf_rotation=0,
        leaf_font_size=8,
        orientation="left",
    )
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Distance")
    plt.ylabel("Proteins")

    # Add color labels to the y-axis
    ax = plt.gca()
    y_labels = ax.get_ymajorticklabels()
    for label, color in zip(y_labels, label_colors):
        label.set_color(color)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def nearest_neighbor_clustering(
    distance_matrix: pd.DataFrame, categories: List[str]
) -> float:
    """
    Perform nearest neighbor clustering and benchmark using adjusted RAND index.

    Args:
        distance_matrix (pd.DataFrame): Distance matrix for clustering.
        categories (List[str]): Ground truth category labels.

    Returns:
        float: Adjusted RAND index score.
    """
    n_clusters = len(set(categories))
    clustering_model = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    )
    predicted_clusters = clustering_model.fit_predict(distance_matrix.values)
    ari_score = adjusted_rand_score(categories, predicted_clusters)
    return ari_score


from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_cluster_confusion_matrix(true_labels, predicted_clusters, output_file):
    """
    Plot a confusion matrix-like representation of the clustering results as percentages.

    Args:
        true_labels (List[str]): Ground truth category labels.
        predicted_clusters (List[int]): Predicted cluster labels.
        output_file (str): File path to save the confusion matrix plot.
    """
    # Ensure consistent label types
    unique_labels = np.unique(true_labels)  # Ensure labels are sorted
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    true_labels_int = np.array([label_to_int[label] for label in true_labels])

    # Match clusters to true labels
    matched_labels = match_clusters_to_labels(true_labels_int, predicted_clusters)
    # Convert matched labels to int
    matched_labels = np.array([int(label) for label in matched_labels])
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true_labels_int, matched_labels)

    # Convert counts to percentages
    conf_matrix_percentage = (
        conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    ) * 100

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))  # Larger figure for better readability
    sns.heatmap(
        conf_matrix_percentage,
        annot=True,
        fmt=".1f",  # Show percentages with one decimal
        xticklabels=unique_labels,
        yticklabels=unique_labels,
        cmap="viridis",  # Use viridis colormap
        cbar_kws={"label": "Percentage (%)"},
        vmin=0,  # Minimum value for color scale
        vmax=100,  # Explicitly set max to 100
        annot_kws={"size": 8},  # Larger font for annotations
    )
    plt.xlabel("Predicted Cluster", fontsize=14)  # Larger axis label font
    plt.ylabel("True Label", fontsize=14)  # Larger axis label font
    plt.title("Cluster Confusion Matrix (Percentage)", fontsize=16)  # Larger title font
    plt.xticks(fontsize=12)  # Larger tick font
    plt.yticks(fontsize=12)  # Larger tick font
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    # Calculate F1 score
    f1 = f1_score(true_labels_int, matched_labels, average="macro")
    print(f"F1-score: {f1} - {output_file}")


def perform_mds(distance_matrix: pd.DataFrame, categories: List[str], output_file: str):
    """
    Perform MDS and plot proteins as points.

    Args:
        distance_matrix (pd.DataFrame): Distance matrix to reduce.
        categories (List[str]): Ground truth category labels.
        output_file (str): Path to save the plot.
    """
    mds_model = MDS(
        n_components=10,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress=False,
    )
    coords = mds_model.fit_transform(distance_matrix.values)

    plt.figure(figsize=(10, 8))
    for category in set(categories):
        idx = [i for i, c in enumerate(categories) if c == category]
        plt.scatter(coords[idx, 0], coords[idx, 1], label=category)

    plt.title("MDS Visualization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def clustering_error_matrix_hierarchical(
    distance_matrix: pd.DataFrame, categories: List[str], n_categories: int
) -> pd.DataFrame:
    """
    Compute a clustering error matrix based on hierarchical clustering predictions.

    Args:
        distance_matrix (pd.DataFrame): Distance matrix for clustering.
        categories (List[str]): Ground truth category labels.
        n_categories (int): Number of categories for clustering.

    Returns:
        pd.DataFrame: Error matrix where entry (i, j) represents the clustering error
                      for the pair of proteins i and j.
    """
    # Ensure the distance matrix is symmetric
    distance_matrix.values[:] = (distance_matrix.values + distance_matrix.values.T) / 2

    # Ensure diagonal is exactly zero
    np.fill_diagonal(distance_matrix.values, 0)

    # Clip any small negative values caused by floating-point errors
    distance_matrix.values[:] = np.clip(distance_matrix.values, 0, None)

    # Convert the distance matrix to condensed form
    condensed_distance_matrix = squareform(distance_matrix.values)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance_matrix, method="average")

    # Cut the dendrogram to form n_categories clusters
    predicted_clusters = fcluster(linkage_matrix, n_categories, criterion="maxclust")

    # Map categories and clusters to each protein
    protein_ids = distance_matrix.index
    category_mapping = {
        protein: category for protein, category in zip(protein_ids, categories)
    }
    cluster_mapping = {
        protein: cluster for protein, cluster in zip(protein_ids, predicted_clusters)
    }

    # Create the error matrix
    error_matrix = pd.DataFrame(0, index=protein_ids, columns=protein_ids, dtype=float)
    for i, protein1 in enumerate(protein_ids):
        for j, protein2 in enumerate(protein_ids):
            if i == j:
                continue
            # Compare true category and predicted cluster
            same_category = category_mapping[protein1] == category_mapping[protein2]
            same_cluster = cluster_mapping[protein1] == cluster_mapping[protein2]
            # Error is 1 if category and cluster mismatch, 0 otherwise
            error_matrix.iloc[i, j] = 0 if same_category == same_cluster else 1

    return error_matrix


def plot_mean_matrix(
    distance_matrix: pd.DataFrame,
    categories: List[str],
    unique_sorted_categories: List[str],
    output_file: str,
    normalize: bool = True,
):
    """
    Processes a distance matrix to calculate mean distances between categories,
    creates a confusion matrix-like heatmap, and optionally applies min-max normalization.

    Args:
        distance_matrix (pd.DataFrame): Symmetric matrix of distances indexed by PDB codes.
        categories (List[str]): List of category labels corresponding to each PDB.
        unique_sorted_categories (List[str]): Sorted list of unique categories.
        output_file (str): File path to save the heatmap plot.
        normalize (bool): Whether to apply min-max normalization to the mean matrix.
    """
    # Initialize the mean distance matrix
    mean_matrix = pd.DataFrame(
        np.nan, index=unique_sorted_categories, columns=unique_sorted_categories
    )

    # Iterate over each pair of categories to compute mean distances
    for cat1 in unique_sorted_categories:
        for cat2 in unique_sorted_categories:
            # Find indices for the current pair of categories
            idx1 = [i for i, c in enumerate(categories) if c == cat1]
            idx2 = [i for i, c in enumerate(categories) if c == cat2]

            # Extract the block of distances
            block = distance_matrix.iloc[idx1, idx2].values

            # If comparing the same category, exclude the diagonal
            if cat1 == cat2:
                n = len(idx1)
                if n > 1:
                    # Extract the upper triangle without the diagonal
                    triu_indices = np.triu_indices(n, k=1)
                    block = distance_matrix.iloc[idx1, idx2].values[triu_indices]
                else:
                    # If only one PDB in the category, no pairs exist
                    block = np.array([])

            # Calculate the mean, ignoring NaNs
            mean_val = np.nanmean(block) if block.size > 0 else np.nan
            mean_matrix.loc[cat1, cat2] = mean_val

    # Replace NaNs with 0 for visualization purposes
    mean_matrix = mean_matrix.fillna(0)

    # Apply min-max normalization if required
    if normalize:
        min_val = mean_matrix.values.min()
        max_val = mean_matrix.values.max()
        if max_val > min_val:  # Avoid division by zero
            mean_matrix = (mean_matrix - min_val) / (max_val - min_val)

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        mean_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "Mean Distance"},
        xticklabels=unique_sorted_categories,
        yticklabels=unique_sorted_categories,
        vmin=0,  # Always 0 after normalization
        vmax=1 if normalize else mean_matrix.values.max(),  # 1 if normalized
    )
    plt.title("Mean Distance Between Categories")
    plt.xlabel("Category")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_pca_lineplots(
    coords_full: np.ndarray, categories: List[str], output_file: Path, metric_name: str
):
    """
    Create PCA value plots and line plots for specific functions.

    Args:
        coords_full (np.ndarray): PCoA coordinates (points-based).
        categories (List[str]): List of category labels for each point.
        output_file (Path): Path to save the plots.
        metric_name (str): Name of the distance metric used.
    """
    n_components = 10

    # Select function groups
    single_indices = [i for i, c in enumerate(categories) if c.count("+") == 0]
    paired_indices = [i for i, c in enumerate(categories) if c.count("+") == 1]
    triplet_indices = [i for i, c in enumerate(categories) if c.count("+") > 1]

    coords_single_function = coords_full[single_indices]
    coords_paired_function = coords_full[paired_indices]
    coords_triplet_function = coords_full[triplet_indices]

    single_functions = [categories[i] for i in single_indices]
    paired_functions = [categories[i] for i in paired_indices]
    triplet_functions = [categories[i] for i in triplet_indices]

    # Assign unique colors for single functions only
    unique_single_functions = sorted(set(single_functions))
    cmap = plt.cm.tab20  # Use a consistent colormap
    category_colors = {
        category: cmap(i % cmap.N) for i, category in enumerate(unique_single_functions)
    }

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot Single Functions (Scatter with colors)
    for i, coords in enumerate(coords_single_function):
        color = category_colors[single_functions[i]]
        axes[0].scatter(
            range(1, n_components + 1), coords[:n_components], color=color, alpha=0.2
        )
        axes[1].scatter(
            range(1, n_components + 1), coords[:n_components], color=color, alpha=0.2
        )

    # Plot Paired Functions (Line, Black)
    for coords in coords_paired_function[:5]:
        axes[0].plot(
            range(1, n_components + 1), coords[:n_components], color="black", alpha=0.2
        )

    # Plot Triplet Functions (Line, Black)
    for coords in coords_triplet_function[:5]:
        axes[1].plot(
            range(1, n_components + 1), coords[:n_components], color="black", alpha=0.2
        )

    # Customize axes
    axes[0].set_title(f"{metric_name} Single and Paired Functions")
    axes[0].set_xlabel("Number of Components")
    axes[0].set_ylabel("PCoA Value")
    axes[0].grid()

    axes[1].set_title(f"{metric_name} Single and Triplet Functions")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("PCoA Value")
    axes[1].grid()

    # Create custom legend for single functions only
    legend_elements = [
        Patch(facecolor=category_colors[cat], label=cat)
        for cat in unique_single_functions
    ]

    # Adjust layout to provide space for the legend
    plt.subplots_adjust(
        bottom=0.2
    )  # Increase bottom margin to make space for the legend

    # Add legend at the bottom
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.05)
        # Adjusted to be within the figure
    )

    # Save the figure with proper bounding box
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def marker_for_category(category):
    plus_count = category.count("+")
    if plus_count == 0:
        return "s"  # Square for single functions
    elif plus_count == 1:
        return "^"  # Triangle for paired functions
    else:
        return "P"  # Plus for triplet functions


def perform_pcoa_and_plot(
    distance_matrix: pd.DataFrame,
    categories: List[str],
    metric_name: str,
    output_file: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform PCoA using numpy and scipy, and create 2D scatter plots.

    Args:
        distance_matrix (pd.DataFrame): Square distance matrix.
        categories (List[str]): List of category labels for each point.
        metric_name (str): Name of the distance metric used (e.g., "RMSD").
        output_file (Path): Path to save the scatter plots and elbow plot.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Coordinates (PCoA) and cumulative variance.
    """
    # Convert distance_matrix to a NumPy array
    distance_matrix_np = distance_matrix.values
    pdb_names = distance_matrix.index
    # Ensure the matrix is symmetric
    n = distance_matrix_np.shape[0]
    assert (
        distance_matrix_np.shape[0] == distance_matrix_np.shape[1]
    ), "Matrix must be square."

    # Double centering to create the B matrix
    row_means = np.mean(distance_matrix_np, axis=1)
    total_mean = np.mean(row_means)
    B = -0.5 * (
        distance_matrix_np - row_means[:, None] - row_means[None, :] + total_mean
    )

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Retain only positive eigenvalues
    positive = eigvals > 0
    eigvals = eigvals[positive]
    eigvecs = eigvecs[:, positive]

    # Compute coordinates
    coords = eigvecs * np.sqrt(eigvals)

    # Elbow Plot: Proportion of variance explained
    explained_variance = eigvals / np.sum(eigvals)
    cumulative_variance = np.cumsum(explained_variance)

    elbow_plot_file = output_file.with_name(f"{metric_name}_elbow_plot.pdf")
    plt.figure(figsize=(5, 5))
    plt.plot(
        range(1, len(explained_variance) + 1),
        cumulative_variance,
        marker="o",
        linestyle="--",
    )
    # plt.title(f"{metric_name} Elbow Plot: Cumulative Variance Explained", fontsize=14)
    plt.xlabel("Number of Dimensions", fontsize=12)
    plt.ylabel("Cumulative Variance Explained", fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()

    # Save the elbow plot
    plt.savefig(elbow_plot_file, dpi=300)
    plt.close()

    # Scatter plot with matplotlib
    scatter_plot_file = output_file.with_name(f"{metric_name}_scatter_plot.pdf")
    x, y = coords[:, 0], coords[:, 1]
    plt.figure(figsize=(10, 8))
    unique_categories = sorted(set(categories))
    colormap = plt.cm.get_cmap("tab20", len(unique_categories))
    category_to_color = {cat: colormap(i) for i, cat in enumerate(unique_categories)}
    colors = [category_to_color[cat] for cat in categories]

    for category in unique_categories:
        idx = [i for i, c in enumerate(categories) if c == category]
        marker = marker_for_category(category)
        plt.scatter(x[idx], y[idx], label=category, alpha=0.8, s=100, marker=marker)

    plt.title(f"PCoA Scatter Plot ({metric_name})", fontsize=14)
    plt.xlabel("PCoA Axis 1", fontsize=12)
    plt.ylabel("PCoA Axis 2", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(scatter_plot_file, dpi=300)
    plt.close()

    # Interactive scatter plot with Plotly
    scatter_with_labels_plotly(
        x=x,
        y=y,
        labels=categories,
        title=f"PCoA Scatter Plot ({metric_name})",
        xlabel="PCoA Axis 1",
        ylabel="PCoA Axis 2",
        output_file=str(output_file.with_suffix(".html")),
        pdb_codes=pdb_names,
    )

    violin_output_file = output_file.with_name(f"{metric_name}_pcoa_violin.pdf")
    # plot_dim_reduction_violin(
    #     coords, categories, f"{metric_name} PCoA", violin_output_file
    # )
    grouped_violin_output_file = output_file.with_name(
        f"{metric_name}_pcoa_grouped_violin.pdf"
    )
    # plot_single_pair_triplet_violin(
    #     coords, categories, f"{metric_name} PCoA", grouped_violin_output_file
    # )
    unique_categories = sorted(set(categories))

    quality_scores = evaluate_dimensionality_reduction_quality(
        distance_matrix, coords, n_clusters=len(unique_categories)
    )
    print(f"PCoA Quality Scores ({metric_name}): {quality_scores}")

    # Perform K-Means clustering
    kmeans_output_file = output_file.with_name(f"{metric_name}_kmeans_pcoa")
    # Select the first two dimensions for clustering
    embedding = coords[:, :2]
    # perform_kmeans_on_embedding(embedding, categories, kmeans_output_file)

    # Return both coordinates and cumulative variance
    return coords, cumulative_variance


def marker_for_category(category: str) -> str:
    """
    Determines the marker style based on the category.
    """
    plus_count = category.count("+")
    if plus_count == 0:
        return "s"  # Square for single functions
    elif plus_count == 1:
        return "^"  # Triangle for paired functions
    else:
        return "P"  # Plus for triplet functions


def scatter_with_labels(
    x: np.ndarray,
    y: np.ndarray,
    categories: List[str],
    colors: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    output_file: str,
):
    """
    Create a scatter plot with categories as labels and adjust text for visibility.

    Args:
        x (np.ndarray): X coordinates for points.
        y (np.ndarray): Y coordinates for points.
        categories (List[str]): Labels/Categories for each point.
        colors (List[str]): Colors for each point.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        output_file (str): File path to save the plot.
    """

    def marker_for_category(category: str) -> str:
        """
        Determines the marker style based on the category.
        """
        plus_count = category.count("+")
        if plus_count == 0:
            return "s"  # Square for single functions
        elif plus_count == 1:
            return "^"  # Triangle for paired functions
        else:
            return "P"  # Plus for triplet functions

    plt.figure(figsize=(10, 8))
    unique_categories = sorted(set(categories))
    legend_elements = []

    # Add scatter points with markers and colors
    texts = []
    added_labels = set()  # Track already-added labels to avoid duplicates

    for xi, yi, category, color in zip(x, y, categories, colors):
        marker = marker_for_category(category)  # Determine marker style
        plt.scatter(
            xi,
            yi,
            color=color,
            alpha=0.8,
            s=100,
            marker=marker,
            label=category if category not in added_labels else None,
        )
        added_labels.add(category)

    #     # Add label only if it isn't too close to existing ones
    #     texts.append(plt.text(xi, yi, category, fontsize=8, color=color, alpha=0.9))
    #
    # adjust_text(
    #     texts,
    #     arrowprops=dict(arrowstyle="->", color="gray"),
    #     force_points=0.2,
    #     force_text=0.5,
    #     only_move={"points": "y", "text": "x"},
    # )

    # Add legend
    for category in unique_categories:
        color = colors[categories.index(category)]
        marker = marker_for_category(category)
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=color,
                markersize=10,
                label=category,
            )
        )

    plt.legend(
        handles=legend_elements,
        fontsize=10,
        loc="upper right",
        title="Categories",
    )

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def scatter_with_labels_plotly(
    x: np.ndarray,
    y: np.ndarray,
    labels: List[str],
    pdb_codes: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    output_file: str,
):
    """
    Create an interactive scatter plot with Plotly and save it as an HTML file.

    Args:
        x (np.ndarray): X coordinates for points.
        y (np.ndarray): Y coordinates for points.
        labels (List[str]): Labels for each point (used as both categories and hover text).
        pdb_codes (List[str]): PDB codes for hover information.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        output_file (str): File path to save the plot.
    """
    # Create a DataFrame for Plotly
    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "label": labels,
            "pdb_code": pdb_codes,
        }
    )

    # Assign unique colors dynamically for each label
    unique_labels = sorted(set(labels))
    colormap = px.colors.qualitative.Plotly
    label_to_color = {
        label: colormap[i % len(colormap)] for i, label in enumerate(unique_labels)
    }
    df["color"] = df["label"].map(label_to_color)

    # Create the Plotly scatter plot
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="label",
        text="pdb_code",  # Add PDB codes as hover text
        title=title,
        labels={"x": xlabel, "y": ylabel},
        color_discrete_map=label_to_color,
        hover_data={
            "pdb_code": True,
            "label": True,
        },  # Ensure PDB codes are visible in hover
    )

    # Customize layout for better visibility
    fig.update_traces(marker=dict(size=12, opacity=0.8), textposition="top center")
    fig.update_layout(
        title=dict(font=dict(size=20)),
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        legend=dict(title="Categories"),
    )

    # Save the interactive plot as an HTML file
    fig.write_html(output_file)
    print(f"Interactive scatter plot saved as {output_file}")


def perform_umap_and_plot(
    distance_matrix: pd.DataFrame,
    categories: List[str],
    metric_name: str,
    output_file: Path,
    clustering_output_file: Path,
):
    """
    Perform UMAP dimensionality reduction, cluster embeddings, and create a 2D scatter plot with points colored by categories.

    Args:
        distance_matrix (pd.DataFrame): Square distance matrix.
        categories (List[str]): Ground truth category labels.
        metric_name (str): Name of the distance metric used (e.g., "RMSD").
        output_file (Path): Path to save the scatter plot.
        clustering_output_file (Path): Path to save confusion matrix plot.
    """
    distance_matrix = distance_matrix.clip(lower=0)
    pdb_names = distance_matrix.index

    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(metric="precomputed", random_state=42, n_components=10)
    embedding = reducer.fit_transform(distance_matrix)

    # Prepare colors
    unique_categories = sorted(set(categories))
    colormap = plt.cm.get_cmap("tab20", len(unique_categories))
    category_to_color = {cat: colormap(i) for i, cat in enumerate(unique_categories)}
    colors = [category_to_color[cat] for cat in categories]

    # Plot UMAP scatter plot
    scatter_with_labels_plotly(
        x=embedding[:, 0],
        y=embedding[:, 1],
        labels=categories,
        title=f"UMAP Visualization ({metric_name})",
        xlabel="UMAP Axis 1",
        ylabel="UMAP Axis 2",
        output_file=output_file,
        pdb_codes=pdb_names,
    )

    scatter_with_labels(
        x=embedding[:, 0],
        y=embedding[:, 1],
        categories=categories,
        colors=colors,
        title=f"UMAP Visualization ({metric_name})",
        xlabel="UMAP Axis 1",
        ylabel="UMAP Axis 2",
        output_file=output_file.with_suffix(".pdf"),
    )

    n_clusters = len(set(categories))

    # Perform K-Means Clustering
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_clusters = kmeans_model.fit_predict(embedding)
    kmeans_scores = evaluate_clustering_scores(
        true_labels=categories,
        predicted_labels=kmeans_clusters,
        distance_matrix=distance_matrix.values,
    )
    print(f"K-Means Clustering on UMAP ({metric_name}): {kmeans_scores}")

    kmeans_confusion_file = clustering_output_file.with_name(
        f"{metric_name}_kmeans_confusion_matrix.pdf"
    )
    plot_cluster_confusion_matrix(
        true_labels=np.array(categories),
        predicted_clusters=kmeans_clusters,
        output_file=kmeans_confusion_file,
    )
    print(f"K-Means Clustering confusion matrix saved to {kmeans_confusion_file}")

    # Perform GMM Clustering
    gmm_model = GaussianMixture(
        n_components=n_clusters, covariance_type="full", random_state=42
    )
    gmm_clusters = gmm_model.fit(embedding).predict(embedding)
    gmm_scores = evaluate_clustering_scores(
        true_labels=categories,
        predicted_labels=gmm_clusters,
        distance_matrix=distance_matrix.values,
    )
    print(f"GMM Clustering on UMAP ({metric_name}): {gmm_scores}")

    gmm_confusion_file = clustering_output_file.with_name(
        f"{metric_name}_gmm_confusion_matrix.pdf"
    )
    plot_cluster_confusion_matrix(
        true_labels=np.array(categories),
        predicted_clusters=gmm_clusters,
        output_file=gmm_confusion_file,
    )
    print(f"GMM Clustering confusion matrix saved to {gmm_confusion_file}")

    # Evaluate dimensionality reduction quality
    quality_scores = evaluate_dimensionality_reduction_quality(
        distance_matrix, embedding, n_clusters=n_clusters
    )
    print(f"UMAP Quality Scores ({metric_name}): {quality_scores}")


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
from pathlib import Path


def perform_tsne_and_plot(
    distance_matrix: pd.DataFrame,
    categories: List[str],
    metric_name: str,
    output_file: Path,
):
    """
    Perform t-SNE dimensionality reduction and create a 2D scatter plot with points colored by categories.

    Args:
        distance_matrix (pd.DataFrame): Square distance matrix.
        categories (List[str]): List of category labels for each point.
        metric_name (str): Name of the distance metric used (e.g., "RMSD").
        output_file (Path): Path to save the scatter plot.
    """
    # Clip
    distance_matrix = distance_matrix.clip(lower=0)
    pdb_names = distance_matrix.index

    # Initialize t-SNE
    tsne = TSNE(
        n_components=10,
        metric="precomputed",
        random_state=42,
        perplexity=20,
        n_iter=1000,
        init="random",
        method="exact",
    )

    # Fit t-SNE on the precomputed distance matrix
    embedding = tsne.fit_transform(distance_matrix)

    # Prepare colors
    unique_categories = sorted(set(categories))
    colormap = plt.cm.get_cmap("tab20", len(unique_categories))
    category_to_color = {cat: colormap(i) for i, cat in enumerate(unique_categories)}
    colors = [category_to_color[cat] for cat in categories]

    # Plot t-SNE scatter plot
    scatter_with_labels_plotly(
        x=embedding[:, 0],
        y=embedding[:, 1],
        labels=categories,
        title=f"t-SNE Visualization ({metric_name})",
        xlabel="t-SNE Axis 1",
        ylabel="t-SNE Axis 2",
        output_file=output_file,
        pdb_codes=pdb_names,
    )
    scatter_with_labels(
        x=embedding[:, 0],
        y=embedding[:, 1],
        categories=categories,
        colors=colors,
        title=f"t-SNE Visualization ({metric_name})",
        xlabel="t-SNE Axis 1",
        ylabel="t-SNE Axis 2",
        output_file=output_file.with_suffix(".pdf"),
    )

    n_clusters = len(set(categories))

    # Perform K-Means Clustering
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_clusters = kmeans_model.fit_predict(embedding)
    kmeans_scores = evaluate_clustering_scores(
        true_labels=categories,
        predicted_labels=kmeans_clusters,
        distance_matrix=distance_matrix.values,
    )
    print(f"K-Means Clustering on t-SNE ({metric_name}): {kmeans_scores}")

    kmeans_confusion_file = output_file.with_name(
        f"{metric_name}_kmeans_confusion_matrix.pdf"
    )
    plot_cluster_confusion_matrix(
        true_labels=np.array(categories),
        predicted_clusters=kmeans_clusters,
        output_file=kmeans_confusion_file,
    )
    print(f"K-Means Clustering confusion matrix saved to {kmeans_confusion_file}")

    # Perform GMM Clustering
    gmm_model = GaussianMixture(
        n_components=n_clusters, covariance_type="full", random_state=42
    )
    gmm_clusters = gmm_model.fit(embedding).predict(embedding)
    gmm_scores = evaluate_clustering_scores(
        true_labels=categories,
        predicted_labels=gmm_clusters,
        distance_matrix=distance_matrix.values,
    )
    print(f"GMM Clustering on t-SNE ({metric_name}): {gmm_scores}")

    gmm_confusion_file = output_file.with_name(
        f"{metric_name}_gmm_confusion_matrix.pdf"
    )
    plot_cluster_confusion_matrix(
        true_labels=np.array(categories),
        predicted_clusters=gmm_clusters,
        output_file=gmm_confusion_file,
    )
    print(f"GMM Clustering confusion matrix saved to {gmm_confusion_file}")

    # Evaluate dimensionality reduction quality
    quality_scores = evaluate_dimensionality_reduction_quality(
        distance_matrix, embedding, n_clusters=n_clusters
    )
    print(f"t-SNE Quality Scores ({metric_name}): {quality_scores}")


def sort_categories(categories: List[str]) -> List[str]:
    """
    Sorts categories:
    - Categories without "+" come first.
    - Categories with one "+" come next.
    - Categories with two "+" come last.
    - Within each group, sorts alphabetically.
    """

    def category_sort_key(category):
        num_plus = category.count("+")
        return (num_plus, category.lower())

    return sorted(categories, key=category_sort_key)


def evaluate_clustering_scores(
    true_labels: List[str],
    predicted_labels: np.ndarray,
    distance_matrix: np.ndarray,
) -> Dict[str, float]:
    """
    Compute various clustering evaluation metrics.

    Args:
        true_labels (List[str]): Ground truth labels.
        predicted_labels (np.ndarray): Predicted cluster labels.
        distance_matrix (np.ndarray): Precomputed distance matrix.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    distance_matrix = np.clip(distance_matrix, 0, None)
    scores = {
        "ARI": adjusted_rand_score(true_labels, predicted_labels),
        "NMI": normalized_mutual_info_score(true_labels, predicted_labels),
        "Homogeneity": homogeneity_score(true_labels, predicted_labels),
        "Completeness": completeness_score(true_labels, predicted_labels),
        "V-Measure": v_measure_score(true_labels, predicted_labels),
        "Silhouette": silhouette_score(
            distance_matrix, predicted_labels, metric="precomputed"
        ),
    }
    return scores


def generate_combined_elbow_plot(
    all_distance_matrices: Dict[str, pd.DataFrame],
    pdb_data: Dict[str, Dict[str, Path]],
    output_path: Path,
):
    """
    Generates individual and combined elbow plots for all distance matrices.
    Performs clustering on 10D PCoA space and plots confusion matrices.

    Args:
        all_distance_matrices (Dict[str, pd.DataFrame]): Dictionary of distance matrices by metric.
        pdb_data (Dict[str, Dict[str, Path]]): Dictionary containing PDB metadata with categories.
        output_path (Path): Path to save the combined elbow plot and individual plots.
    """
    cumulative_variances = {}  # Store cumulative variances for all metrics

    # Define the fixed order of metrics
    metric_order = ["BagOfNodes", "GraphEditDistance", "BLOSUM", "RMSD"]

    # Ensure the distance matrices are processed in the defined order
    for metric in metric_order:
        if metric in all_distance_matrices:
            distance_matrix = all_distance_matrices[metric]
            pdb_categories = [
                pdb_data[pdb]["category"] for pdb in distance_matrix.index
            ]

            # Perform PCoA and extract explained variance
            output_pcoa_file = output_path / f"{metric}_pcoa"
            coords, cumulative_variance = perform_pcoa_and_plot(
                distance_matrix, pdb_categories, metric, output_pcoa_file
            )
            coords_10D = coords[:, :10]  # First 10 dimensions of PCoA
            cumulative_variances[metric] = cumulative_variance

    # Define colors using Seaborn colorblind palette and custom grays
    palette = sns.color_palette("colorblind", n_colors=2)
    color_mapping = {
        "BagOfNodes": palette[0],  # Seaborn colorblind color
        "GraphEditDistance": palette[1],  # Seaborn colorblind color
        "BLOSUM": "#7f7f7f",  # Light gray
        "RMSD": "#141414",  # Darker gray
    }

    # Combined Elbow Plot
    plt.figure(figsize=(4, 4))
    sns.set(style="whitegrid")  # Default Seaborn grid style

    ax = plt.gca()

    # Plot each metric in the specified order with its assigned color
    for metric in metric_order:
        if metric in cumulative_variances:
            variance = cumulative_variances[metric]
            plt.plot(
                range(1, len(variance) + 1),
                variance,
                marker="o",
                linestyle="--",
                label=metric,
                alpha=0.8,
                color=color_mapping[metric],
            )

    plt.xlabel("Number of Dimensions", fontsize=12)
    plt.ylabel("Cumulative Variance Explained", fontsize=12)
    plt.legend(fontsize=10, loc="best")

    # Black border on all four sides
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")

    # Remove all grid lines and add only horizontal dotted grey lines
    plt.grid(visible=False)  # Remove all grid lines first
    plt.grid(axis="y", linestyle="--", alpha=0.5, color="grey")  # Add only horizontal lines

    # plt.tight_layout()

    combined_elbow_plot_file = output_path / "combined_elbow_plot.pdf"
    print(f"Saving combined elbow plot to {combined_elbow_plot_file}")
    plt.savefig(combined_elbow_plot_file, format="pdf", bbox_inches="tight")
    plt.close()




def compute_blosum_distance(
    pair: Tuple[str, str],
    preloaded_pdbs: Dict[str, ampal.Assembly],
    aligner: PairwiseAligner,
) -> Tuple[str, str, float]:
    """Compute the BLOSUM-based distance for a pair of sequences using the snippet's logic."""
    pdb1, pdb2 = pair
    try:
        seq1 = preloaded_pdbs[pdb1][0].sequence
        seq2 = preloaded_pdbs[pdb2][0].sequence

        # Perform alignment and extract aligned sequences
        alignment = aligner.align(seq1, seq2)[0]  # Take the best alignment
        aln1, aln2 = str(alignment[0]), str(alignment[1])

        if len(aln1) != len(aln2):
            raise ValueError("Aligned sequences must have the same length.")

        # Replace gaps with '*'
        aln1 = aln1.replace("-", "*")
        aln2 = aln2.replace("-", "*")

        # Load BLOSUM matrix as a dictionary like the snippet
        blosum62 = substitution_matrices.load("BLOSUM62")
        blosum_matrix = {(aa1 + aa2): score for (aa1, aa2), score in blosum62.items()}

        # Convert scores to probabilities
        blosum_probs = {pair: math.exp(score) for pair, score in blosum_matrix.items()}

        # Calculate distances as per the snippet:
        # For each position, use 1 / probability. Default to 1.0 if not found.
        distances = [1 / blosum_probs.get(c1 + c2, 1.0) for c1, c2 in zip(aln1, aln2)]

        # Return the sum of these distances without normalization
        total_distance = sum(distances)
        return pdb1, pdb2, total_distance

    except Exception as e:
        print(f"Failed to compute similarity distance for pair {pdb1}, {pdb2}: {e}")
        # Return a fallback large distance if something goes wrong
        return pdb1, pdb2, 1e6  # Large number as a fallback


def compute_blosum_distance_wrapper(args):
    """Wrapper for multiprocessing."""
    pair, preloaded_pdbs, aligner = args
    return compute_blosum_distance(pair, preloaded_pdbs, aligner)


def create_blosum_distance_matrix_parallel(
    pdb_data: Dict[str, Dict[str, Path]],
    checkpoint_file: str = "blosum_checkpoint.csv",
    num_workers: int = 4,
) -> pd.DataFrame:
    """Create BLOSUM-based distance matrix in parallel."""
    # Prepare pairs
    pdb_keys = sorted(pdb_data.keys())
    pairs = list(combinations(pdb_keys, 2))

    # Preload PDB files
    manager = Manager()
    preloaded_pdbs = manager.dict(preload_pdb_files(pdb_data))

    # Initialize aligner
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

    # Prepare arguments for multiprocessing
    args_list = [(pair, preloaded_pdbs, aligner) for pair in pairs]

    # Compute distances in parallel
    with Pool(processes=num_workers) as pool:
        results = []
        for result in tqdm(
            pool.imap_unordered(compute_blosum_distance_wrapper, args_list),
            total=len(args_list),
        ):
            if result:
                results.append(result)

                # Save intermediate results
                if len(results) >= 500:  # Save every 500 results
                    append_to_csv(checkpoint_file, results)
                    results = []

    # Save remaining results
    if results:
        append_to_csv(checkpoint_file, results)

    # Construct the distance matrix
    all_results = pd.read_csv(checkpoint_file)
    blosum_matrix = pd.DataFrame(np.inf, index=pdb_keys, columns=pdb_keys)
    for _, row in all_results.iterrows():
        pdb1, pdb2, distance = row["PDB1"], row["PDB2"], row["RMSD"]
        blosum_matrix.loc[pdb1, pdb2] = distance
        blosum_matrix.loc[pdb2, pdb1] = distance

    return blosum_matrix


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd


def plot_data(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    categories: list,
    category_to_color: dict,
    output_file: Path,
):
    """
    Helper function to plot data with specified attributes, coloring points by their category.

    Args:
        x: X-axis data.
        y: Y-axis data.
        title: Title of the plot.
        xlabel: Label for the X-axis.
        ylabel: Label for the Y-axis.
        categories: Categories (PDB categories) for each data point.
        category_to_color: Mapping of category to color.
        output_file: Path to save the plot.
    """
    plt.figure(figsize=(6, 6))
    # Plot each point
    plotted_categories = set()
    for i in range(len(x)):
        cat = categories[i]
        plt.scatter(
            x[i],
            y[i],
            alpha=0.6,
            color=category_to_color[cat],
            label=cat if cat not in plotted_categories else None,
        )
        plotted_categories.add(cat)

    plt.xlim(0, 1)
    plt.ylim(0, 12)
    # plt.title(title)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=5 if len(plotted_categories) <= 5 else 3,
        title="Categories",
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_overlap_coefficient(dist1, dist2, bins=30):
    """
    Calculate the Overlap Coefficient between two distributions.

    Args:
        dist1: First distribution (array-like).
        dist2: Second distribution (array-like).
        bins: The number of bins to use for the histograms.

    Returns:
        The Overlap Coefficient between the two distributions.
    """
    # Calculate the histograms for the distributions
    bins = np.linspace(
        min(np.min(dist1), np.min(dist2)), max(np.max(dist1), np.max(dist2)), bins
    )
    hist1, bin_edges = np.histogram(dist1)
    hist2, bin_edges = np.histogram(dist2)

    # Ensure no zero values in the histograms
    hist1 = hist1.astype(np.float64)
    hist2 = hist2.astype(np.float64)
    hist1 += 1e-10
    hist2 += 1e-10

    # Calculate the overlap coefficient (area of overlap)
    overlap = np.sum(np.minimum(hist1, hist2))
    return overlap


def plot_all_histograms(
    rmsd_matrix: pd.DataFrame,
    ged_matrix: pd.DataFrame,
    bag_nodes_matrix: pd.DataFrame,
    blosum_matrix: pd.DataFrame,
    pdb_data: pd.DataFrame,
    output_dir: Path,
):
    """
    Plot histograms for RMSD, GED, BagOfNodes, and BLOSUM distances,
    with one curve for same-function comparisons and another for different-function comparisons.
    """
    distance_names = ["RMSD", "BLOSUM", "GED", "BagOfNodes"]
    distance_matrices = [rmsd_matrix, blosum_matrix, ged_matrix, bag_nodes_matrix]

    # Prepare lists to store distances for same vs different function pairs
    same_function_distances = {name: [] for name in distance_names}
    different_function_distances = {name: [] for name in distance_names}

    # Get categories from pdb_data
    pdb_categories = [pdb_data[pdb]["category"] for pdb in rmsd_matrix.index]

    # Iterate through the distance matrices and categorize the distances into same and different function pairs
    for i, pdb1 in enumerate(rmsd_matrix.index):
        for j, pdb2 in enumerate(rmsd_matrix.columns):
            if i >= j:  # Avoid double-counting pairs
                continue

            # Get the distances for the current pair
            rmsd = rmsd_matrix.iloc[i, j]
            ged = ged_matrix.iloc[i, j]
            bag_nodes = bag_nodes_matrix.iloc[i, j]
            blosum = blosum_matrix.iloc[i, j]

            # Get the categories of the two proteins
            cat1 = pdb_categories[i]
            cat2 = pdb_categories[j]

            # Check if the pair has the same function or different function
            if cat1 == cat2:  # Same function
                same_function_distances["RMSD"].append(rmsd)
                same_function_distances["GED"].append(ged)
                same_function_distances["BagOfNodes"].append(bag_nodes)
                same_function_distances["BLOSUM"].append(blosum)
            else:  # Different function
                different_function_distances["RMSD"].append(rmsd)
                different_function_distances["GED"].append(ged)
                different_function_distances["BagOfNodes"].append(bag_nodes)
                different_function_distances["BLOSUM"].append(blosum)

    # Check if the categorization and number of items in the lists make sense
    print(f"Same Function Distances (RMSD): {len(same_function_distances['RMSD'])}")
    print(
        f"Different Function Distances (RMSD): {len(different_function_distances['RMSD'])}"
    )

    # Plot histograms with separate curves for same vs different function comparisons
    plt.figure(figsize=(14, 10))  # Larger figure for multiple subplots

    # Store overlap coefficients for each distance type
    overlap_coefficients = {}

    for idx, distance_name in enumerate(distance_names):
        plt.subplot(2, 2, idx + 1)  # 2x2 grid of subplots

        # Get the distances for the current distance type
        same_dist = same_function_distances[distance_name]
        diff_dist = different_function_distances[distance_name]

        # Plot histograms for same-function and different-function comparisons
        sns.histplot(
            same_dist, kde=True, color="blue", label="Same Function", stat="count"
        )
        sns.histplot(
            diff_dist, kde=True, color="red", label="Different Function", stat="count"
        )
        # Check that the length of diff dist is higher than same dist

        # Calculate and log Overlap Coefficient
        overlap = calculate_overlap_coefficient(same_dist, diff_dist)
        overlap_coefficients[distance_name] = overlap

        # Customize the plot
        plt.title(f"{distance_name} - Same vs Different Function")
        plt.xlabel(f"{distance_name} Distance")
        plt.ylabel("Probability")
        plt.legend()

    plt.tight_layout()  # Adjust spacing to avoid overlap
    plt.savefig(output_dir / "same_vs_different_function_histograms.pdf", dpi=300)
    plt.close()

    # Print Overlap Coefficients for each distance type
    for distance_name, overlap in overlap_coefficients.items():
        print(
            f"Overlap Coefficient between Same and Different Function for {distance_name}: {overlap:.4f}"
        )


def plot_by_function_type(
    all_data: np.ndarray,
    function_labels: list,
    categories: list,
    x_index: int,
    y_index: int,
    title_prefix: str,
    xlabel: str,
    ylabel: str,
    output_prefix: str,
    output_dir: Path,
):
    """
    For each function type (Single, Double, Triple), plot the data colored by their actual category.
    """
    function_types = ["Single", "Double", "Triple"]
    for f_type in function_types:
        # Extract data for this function type
        indices = [i for i, flab in enumerate(function_labels) if flab == f_type]
        data_subset = all_data[indices]
        categories_subset = [categories[i] for i in indices]

        if len(data_subset) == 0:
            # No data for this function type
            continue

        # Unique categories for this function type subset
        unique_cats = list(set(categories_subset))
        # Assign colors from a colormap to these categories
        cmap = plt.cm.get_cmap("tab20", len(unique_cats))
        category_to_color = {cat: cmap(i) for i, cat in enumerate(unique_cats)}

        # Plot
        plot_data(
            x=data_subset[:, x_index],
            y=data_subset[:, y_index],
            title=f"{title_prefix} ({f_type} Function)",
            xlabel=xlabel,
            ylabel=ylabel,
            categories=categories_subset,
            category_to_color=category_to_color,
            output_file=output_dir / f"{output_prefix}_{f_type.lower()}_function.pdf",
        )


def plot_function_comparison(
    pdb_data: pd.DataFrame,
    rmsd_matrix: pd.DataFrame,
    ged_matrix: pd.DataFrame,
    bag_nodes_matrix: pd.DataFrame,
    blosum_matrix: pd.DataFrame,
    output_dir: Path,
):
    """
    Plot RMSD vs. GED, RMSD vs. BagOfNodes, BLOSUM vs. GED, and BLOSUM vs. BagOfNodes
    with single, double, and triple functions. For each function type (single, double, triple),
    points are colored by their actual PDB category. Additionally, pooled plots for all functions
    are included and colored by function type.
    """
    pdb_categories = [pdb_data[pdb]["category"] for pdb in rmsd_matrix.index]

    all_data = []
    function_labels = []
    categories = []

    for i, pdb1 in enumerate(rmsd_matrix.index):
        for j, pdb2 in enumerate(rmsd_matrix.columns):
            if i >= j:  # Avoid double-counting pairs
                continue
            rmsd = rmsd_matrix.iloc[i, j]
            ged = ged_matrix.iloc[i, j]
            bag_nodes = bag_nodes_matrix.iloc[i, j]
            blosum = blosum_matrix.iloc[i, j]
            is_same = pdb_categories[i] == pdb_categories[j]
            if is_same:  # Same function only
                cat = pdb_categories[i]
                f_type = (
                    "Single"
                    if "+" not in cat
                    else "Double"
                    if cat.count("+") == 1
                    else "Triple"
                )
                all_data.append((rmsd, blosum, ged, bag_nodes))
                function_labels.append(f_type)
                categories.append(cat)

    all_data = np.array(all_data)
    # Plot histograms for all distances (RMSD, GED, BagOfNodes, BLOSUM)
    plot_all_histograms(
        rmsd_matrix, ged_matrix, bag_nodes_matrix, blosum_matrix, pdb_data, output_dir
    )
    # Plot by function type (color by category)
    # RMSD vs. GED
    plot_by_function_type(
        all_data=all_data,
        function_labels=function_labels,
        categories=categories,
        x_index=1,
        y_index=0,
        title_prefix="RMSD vs. GED",
        xlabel="GED",
        ylabel="RMSD",
        output_prefix="rmsd_vs_ged",
        output_dir=output_dir,
    )

    # RMSD vs. BagOfNodes
    plot_by_function_type(
        all_data=all_data,
        function_labels=function_labels,
        categories=categories,
        x_index=2,
        y_index=0,
        title_prefix="RMSD vs. BagOfNodes",
        xlabel="BagOfNodes",
        ylabel="RMSD",
        output_prefix="rmsd_vs_bag_of_nodes",
        output_dir=output_dir,
    )

    # BLOSUM vs. GED
    plot_by_function_type(
        all_data=all_data,
        function_labels=function_labels,
        categories=categories,
        x_index=1,
        y_index=3,
        title_prefix="BLOSUM vs. GED",
        xlabel="GED",
        ylabel="BLOSUM",
        output_prefix="blosum_vs_ged",
        output_dir=output_dir,
    )

    # BLOSUM vs. BagOfNodes
    plot_by_function_type(
        all_data=all_data,
        function_labels=function_labels,
        categories=categories,
        x_index=2,
        y_index=3,
        title_prefix="BLOSUM vs. BagOfNodes",
        xlabel="BagOfNodes",
        ylabel="BLOSUM",
        output_prefix="blosum_vs_bag_of_nodes",
        output_dir=output_dir,
    )

    # Also produce pooled plots (all data) but colored by function type, if needed:
    # (User's last request was for the single, double, triple function plots by category,
    # but we keep the pooled example here if needed.)
    # If not needed, you can remove these pooled plots.
    function_colors = {"Single": "#377eb8", "Double": "#ff7f00", "Triple": "#4daf4a"}

    def plot_pooled(x_index, y_index, title, xlabel, ylabel, filename):
        plt.figure(figsize=(5, 5))
        for f_type, f_color in function_colors.items():
            idx = [i for i, flab in enumerate(function_labels) if flab == f_type]
            if len(idx) > 0:
                plt.scatter(
                    all_data[idx, x_index],
                    all_data[idx, y_index],
                    alpha=0.6,
                    color=f_color,
                    label=f_type,
                )
        plt.xlim(0, 1)
        plt.ylim(0, 12)
        # plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, title="Functions"
        )
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300)
        plt.close()

    # Pooled plots
    plot_pooled(
        1,
        0,
        "RMSD vs. GED (All Functions Pooled, Same Function Only)",
        "GED",
        "RMSD",
        "rmsd_vs_ged_all_functions_same_function_pooled.pdf",
    )
    plot_pooled(
        2,
        0,
        "RMSD vs. BagOfNodes (All Functions Pooled, Same Function Only)",
        "BagOfNodes",
        "RMSD",
        "rmsd_vs_bag_of_nodes_all_functions_same_function_pooled.pdf",
    )
    plot_pooled(
        1,
        3,
        "BLOSUM vs. GED (All Functions Pooled, Same Function Only)",
        "GED",
        "BLOSUM",
        "blosum_vs_ged_all_functions_same_function_pooled.pdf",
    )
    plot_pooled(
        2,
        3,
        "BLOSUM vs. BagOfNodes (All Functions Pooled, Same Function Only)",
        "BagOfNodes",
        "BLOSUM",
        "blosum_vs_bag_of_nodes_all_functions_same_function_pooled.pdf",
    )


from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def perform_kmeans_on_embedding(
    embedding: np.ndarray,
    categories: List[str],
    output_file: Path,
    n_clusters: int = None,
):
    """
    Apply K-Means clustering to a 2D embedding, evaluate clustering, and save results.

    Args:
        embedding (np.ndarray): 2D array of embedding coordinates (e.g., t-SNE, PCoA).
        categories (List[str]): Ground truth category labels.
        output_file (Path): Path to save the clustering metrics and confusion matrix.
        n_clusters (int): Number of clusters to form. Defaults to the number of unique categories.
    """
    # Default to number of unique categories if n_clusters is not provided
    if n_clusters is None:
        n_clusters = len(set(categories))

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    predicted_clusters = kmeans.fit_predict(embedding)

    # Match predicted cluster labels to ground truth labels
    matched_labels = match_clusters_to_labels(categories, predicted_clusters)

    # Convert both to strings for comparison consistency
    categories_str = list(map(str, categories))
    matched_labels_str = list(map(str, matched_labels))

    # Evaluate clustering
    scores = {
        "ARI": adjusted_rand_score(categories_str, matched_labels_str),
        "NMI": normalized_mutual_info_score(categories_str, matched_labels_str),
        "Homogeneity": homogeneity_score(categories_str, matched_labels_str),
        "Completeness": completeness_score(categories_str, matched_labels_str),
        "V-Measure": v_measure_score(categories_str, matched_labels_str),
        "Silhouette": silhouette_score(embedding, predicted_clusters),
        "Accuracy": accuracy_score(categories_str, matched_labels_str),
        "F1-Score": f1_score(categories_str, matched_labels_str, average="weighted"),
    }

    print(f"K-Means Clustering Metrics: {scores}")

    # Save metrics to a CSV
    metrics_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in scores.items()])
    metrics_file = output_file.with_suffix(".csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Clustering metrics saved to {metrics_file}")

    # Generate and save confusion matrix
    confusion_matrix_output = output_file.with_suffix(".pdf")
    plot_cluster_confusion_matrix(
        true_labels=np.array(categories_str),
        predicted_clusters=np.array(matched_labels_str),
        output_file=confusion_matrix_output,
    )
    print(f"Confusion matrix saved to {confusion_matrix_output}")


def plot_correlation_matrix(
    distance_matrices: Dict[str, pd.DataFrame], output_file: Path
):
    """
    Compute and plot a Spearman correlation matrix for all provided distance matrices.

    Args:
        distance_matrices (Dict[str, pd.DataFrame]): A dictionary with metric names as keys and distance matrices as values.
        output_file (Path): Path to save the correlation matrix plot.
    """
    # Reorder keys to put RMSD and BLOSUM first
    preferred_order = ["RMSD", "BLOSUM"]
    matrix_names = list(distance_matrices.keys())
    reordered_keys = [key for key in preferred_order if key in matrix_names] + [
        key for key in matrix_names if key not in preferred_order
    ]
    distance_matrices = {key: distance_matrices[key] for key in reordered_keys}

    # Compute Spearman correlations between all distance matrices
    matrix_names = list(distance_matrices.keys())
    correlations = pd.DataFrame(index=matrix_names, columns=matrix_names, dtype=float)

    for name1, matrix1 in distance_matrices.items():
        for name2, matrix2 in distance_matrices.items():
            # Flatten upper triangle of each matrix for pairwise comparison
            idx = np.triu_indices_from(matrix1.values, k=1)
            flat1 = matrix1.values[idx]
            flat2 = matrix2.values[idx]
            # Compute Spearman correlation
            corr, _ = spearmanr(flat1, flat2)
            correlations.loc[name1, name2] = corr

    # Plot the correlation matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        correlations.astype(float),
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "Spearman Correlation"},
        vmax=1,
        vmin=-1,
    )
    # plt.title("Correlation Matrix of Distance Matrices")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


from typing import Dict, Any


def create_ratio_csv(
    pdb_data: Dict[str, Any],
    rmsd_matrix: pd.DataFrame,
    ged_matrix: pd.DataFrame,
    bag_nodes_matrix: pd.DataFrame,
    output_csv: Path,
) -> None:
    rows = []
    pdb_keys = rmsd_matrix.index
    for i, pdb1 in enumerate(pdb_keys):
        for j, pdb2 in enumerate(pdb_keys):
            if i >= j:
                continue
            rmsd = rmsd_matrix.iloc[i, j]
            ged = ged_matrix.iloc[i, j]
            bag_nodes = bag_nodes_matrix.iloc[i, j]
            func1 = pdb_data[pdb1]["category"]
            func2 = pdb_data[pdb2]["category"]
            rmsd_ged_ratio = rmsd / ged if ged != 0 else np.nan
            rmsd_bag_ratio = rmsd / bag_nodes if bag_nodes != 0 else np.nan
            rows.append(
                {
                    "PDB1": pdb1,
                    "PDB2": pdb2,
                    "Function1": func1,
                    "Function2": func2,
                    "RMSD": rmsd,
                    "GED": ged,
                    "BagOfNodes": bag_nodes,
                    "RMSD/GED": rmsd_ged_ratio,
                    "RMSD/BagOfNodes": rmsd_bag_ratio,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)


# def plot_spider_charts_per_metric(
#     distance_matrix: pd.DataFrame,
#     categories: List[str],
#     metric_name: str,
#     output_dir: Path,
#     opacity: float = 0.2,
#     line_width: float = 2,
# ):
#     """
#     Plot spider charts for each metric and multi-function category with one line per structure,
#     ensuring unique axes for single-function categories and an inverted radial axis.
#
#     Args:
#         distance_matrix (pd.DataFrame): Distance matrix (n x n) for the metric.
#         categories (List[str]): List of categories corresponding to the rows/columns of the matrix.
#         metric_name (str): Name of the metric (e.g., "RMSD").
#         output_dir (Path): Directory to save the spider charts.
#         opacity (float): Opacity for individual data lines.
#         line_width (float): Thickness of the lines in the spider chart.
#     """
#     # Get unique single-function categories
#     single_categories = sorted(set(cat for cat in categories if "+" not in cat))
#     multi_function_categories = [cat for cat in categories if "+" in cat]
#
#     # Prepare angles for spider plot (one per unique single-function category)
#     num_axes = len(single_categories)
#     angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
#     angles += angles[:1]  # Complete the circle
#
#     for multi_cat in multi_function_categories:
#         # Get indices for structures belonging to this multi-function category
#         multi_cat_indices = [
#             idx for idx, cat in enumerate(categories) if cat == multi_cat
#         ]
#
#         fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
#         ax.set_theta_offset(np.pi / 2)
#         ax.set_theta_direction(-1)
#
#         # Set unique axis labels (single-function categories)
#         plt.xticks(angles[:-1], single_categories, fontsize=10)
#
#         for multi_idx in multi_cat_indices:
#             # Compute median distance for each unique single-function category
#             distances = []
#             for single_cat in single_categories:
#                 # Find indices of structures belonging to the single-function category
#                 single_cat_indices = [
#                     idx for idx, cat in enumerate(categories) if cat == single_cat
#                 ]
#                 # Extract distances and compute median
#                 single_cat_distances = [
#                     distance_matrix.iloc[multi_idx, idx] for idx in single_cat_indices
#                 ]
#                 median_distance = np.median(single_cat_distances)
#                 distances.append(median_distance)
#             distances += distances[:1]  # Close the circle
#
#             # Plot the line for this structure
#             ax.plot(
#                 angles,
#                 distances,
#                 color="blue",
#                 alpha=opacity,
#                 linewidth=line_width,  # Apply custom line width
#             )
#
#         # Invert the radial axis
#         ax.set_ylim(ax.get_ylim()[::-1])  # Invert the axis (smallest at the top)
#
#         # Title and layout
#         plt.title(f"{metric_name} Spider Chart for {multi_cat}", size=14, y=1.1)
#         plt.tight_layout()
#
#         # Save plot
#         output_file = output_dir / f"{metric_name}_{multi_cat}_spider_chart.pdf"
#         plt.savefig(output_file, dpi=300)
#         plt.close()


def plot_median_spider_charts(
    distance_matrix: pd.DataFrame,
    categories: List[str],
    metric_name: str,
    output_dir: Path,
    opacity: float = 0.5,
    line_width: float = 2,
):
    """
    Plot a min spider chart for each multi-function category with stars marking the
    lowest distance categories for dual/triple functions, accounting for inverted axes.

    Args:
        distance_matrix (pd.DataFrame): Distance matrix (n x n) for the metric.
        categories (List[str]): List of categories corresponding to the rows/columns of the matrix.
        metric_name (str): Name of the metric (e.g., "RMSD").
        output_dir (Path): Directory to save the spider charts.
        opacity (float): Opacity for the median line.
        line_width (float): Thickness of the median line in the spider chart.
    """
    # Get unique single-function categories
    single_categories = sorted(set(cat for cat in categories if "+" not in cat))
    multi_function_categories = [cat for cat in categories if "+" in cat]

    # Prepare angles for spider plot (one per unique single-function category)
    num_axes = len(single_categories)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for multi_cat in multi_function_categories:
        # Get indices for structures belonging to this multi-function category
        multi_cat_indices = [
            idx for idx, cat in enumerate(categories) if cat == multi_cat
        ]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Set unique axis labels (single-function categories)
        plt.xticks(angles[:-1], single_categories, fontsize=10)

        # Compute minimum distance for each unique single-function category
        min_distances = []
        for single_cat in single_categories:
            # Find indices of structures belonging to the single-function category
            single_cat_indices = [
                idx for idx, cat in enumerate(categories) if cat == single_cat
            ]
            # Extract distances and compute minimum
            single_cat_distances = [
                np.min(
                    [distance_matrix.iloc[multi_idx, idx] for idx in single_cat_indices]
                )
                for multi_idx in multi_cat_indices
            ]
            min_distance = np.min(single_cat_distances)
            min_distances.append(min_distance)
        min_distances += min_distances[:1]  # Close the circle

        # Plot the minimum line
        ax.plot(
            angles,
            min_distances,
            color="blue",
            alpha=opacity,
            linewidth=line_width,
        )

        # Determine the lowest distances
        sorted_indices = np.argsort(
            min_distances[:-1]
        )  # Exclude the repeated first/last point
        lowest_indices = sorted_indices[
            : len(multi_cat.split("+"))
        ]  # 2 for dual, 3 for triple

        # Invert the radial axis
        max_value = max(min_distances)  # Farthest radial value
        ax.set_ylim(max_value, min(min_distances))  # Invert axis: smallest at top

        # Mark the lowest categories with stars at the **top of the axis**
        for idx in lowest_indices:
            ax.scatter(
                angles[idx],
                min_distances[idx],  # Plot at the actual minimum value
                color="red",
                s=50,
                marker="*",
                label="Lowest Distance",
            )

        # Title and layout
        plt.title(f"{metric_name} Min Spider Chart for {multi_cat}", size=14, y=1.1)
        plt.tight_layout()

        # Save plot
        output_file = output_dir / f"{metric_name}_{multi_cat}_min_spider_chart.pdf"
        plt.savefig(output_file, dpi=300)
        plt.close()


def calculate_retrieval_success(
    distance_matrix: pd.DataFrame,
    categories: List[str],
    aggregation_method: Callable = np.mean,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate retrieval success rates and other evaluation metrics for each function in the distance matrix.

    Args:
        distance_matrix (pd.DataFrame): The distance matrix with proteins as columns.
        categories (List[str]): A list of categories corresponding to each protein.
        aggregation_method (Callable): A numpy function for aggregating column-wise success rates
                                       (e.g., np.mean, np.median, np.max).

    Returns:
        Dict[str, Dict[str, float]]: A dictionary with evaluation metrics for each single function.
    """
    # Identify single functions (exclude multi-function strings containing '+')
    single_functions = [func for func in set(categories) if "+" not in func]

    # Count occurrences of each function
    function_counter = Counter(categories)
    single_fn_count = function_counter
    # single_fn_count = {sf: 0 for sf in single_functions}
    # for func, count in function_counter.items():
    #     if "+" in func:
    #         pass
    #     #     for subf in func.split("+"):
    #     #         single_fn_count[subf] += count
    #     else:
    #         single_fn_count[func] += count

    retrieval_success = {}

    # For storing column-wise results to later aggregate them
    for func in set(categories):
        # Lists to accumulate column-wise results
        precision_20_list: List[float] = []
        precision_n_list: List[float] = []
        dcg_list: List[float] = []
        ndcg_list: List[float] = []
        auroc_list: List[float] = []

        # Boolean mask to select columns belonging to 'func'
        func_idxs = np.array(categories) == func
        func_distance_matrix = distance_matrix.loc[:, func_idxs]

        # Go through each column corresponding to this function
        for col in func_distance_matrix.columns:
            curr_col = func_distance_matrix[col]
            sorted_idx = np.argsort(curr_col)  # ascending order by distance
            sorted_categories = np.array(categories)[sorted_idx]

            # -------------------------------
            # 1) PRECISION@20
            # -------------------------------
            top_20 = sorted_categories[:20]
            correct_in_top_20 = sum(1 for cat in top_20 if func == cat)
            precision_at_20 = correct_in_top_20 / 20
            precision_20_list.append(precision_at_20)

            # -------------------------------
            # 2) PRECISION@N (N = single_fn_count[func])
            # -------------------------------
            n = single_fn_count[func]
            top_n = sorted_categories[:n]
            correct_in_top_n = sum(1 for cat in top_n if func == cat)
            precision_at_n = correct_in_top_n / n if n else 0
            precision_n_list.append(precision_at_n)

            # -------------------------------
            # 3) DCG and NDCG
            # -------------------------------
            # Build relevance scores
            relevance_scores = []
            for cat in sorted_categories:
                # exact single func
                if cat == func:
                    relevance_scores.append(1)
                # if cat has a '+' (multi func) but includes 'func'
                elif "+" in cat and func in cat:
                    # you suggested 1/2 for single or double plus
                    relevance_scores.append(0.5)
                else:
                    relevance_scores.append(0)

            # DCG
            dcg = 0.0
            for idx, rel in enumerate(relevance_scores):
                dcg += rel / np.log2(idx + 2)  # +2 because idx starts at 0

            # IDCG (ideal ranking) -> sorted relevance
            ideal_relevance = sorted(relevance_scores, reverse=True)
            idcg = 0.0
            for idx, rel in enumerate(ideal_relevance):
                idcg += rel / np.log2(idx + 2)

            ndcg = dcg / idcg if idcg != 0 else 0

            dcg_list.append(dcg)
            ndcg_list.append(ndcg)
            # -------------------------------
            # AUROC CALCULATION
            # -------------------------------
            # Define binary relevance (1 for relevant, 0 for irrelevant)
            binary_relevance = np.array([1 if cat == func else 0 for cat in categories])

            try:
                auroc = roc_auc_score(
                    binary_relevance, -curr_col
                )  # Negate distances (higher relevance -> smaller distance)
                auroc_list.append(auroc)
            except ValueError:
                # Handle cases where there's only one class in binary_relevance
                auroc_list.append(float("nan"))
        # -------------------------------
        # AGGREGATE METRICS PER FUNCTION
        # -------------------------------
        p20_agg = aggregation_method(precision_20_list) if precision_20_list else 0
        p20_std = float(np.std(precision_20_list)) if len(precision_20_list) > 1 else 0

        pn_agg = aggregation_method(precision_n_list) if precision_n_list else 0
        pn_std = float(np.std(precision_n_list)) if len(precision_n_list) > 1 else 0

        dcg_agg = aggregation_method(dcg_list) if dcg_list else 0
        dcg_std = float(np.std(dcg_list)) if len(dcg_list) > 1 else 0

        ndcg_agg = aggregation_method(ndcg_list) if ndcg_list else 0
        ndcg_std = float(np.std(ndcg_list)) if len(ndcg_list) > 1 else 0

        auroc_agg = (
            aggregation_method([val for val in auroc_list if not np.isnan(val)])
            if auroc_list
            else 0
        )
        auroc_std = float(np.nanstd(auroc_list)) if len(auroc_list) > 1 else 0
        # Store the aggregated metrics in retrieval_success
        retrieval_success[func] = {
            "AUROC": auroc_agg,
            "std_AUROC": auroc_std,
            "precision@20": p20_agg,
            "std_precision@20": p20_std,
            "precision@N": pn_agg,
            "std_precision@N": pn_std,
            "DCG": dcg_agg,
            "std_DCG": dcg_std,
            "NDCG": ndcg_agg,
            "std_NDCG": ndcg_std,
        }

    return retrieval_success


def plot_results_from_search(results_search: List[Dict[str, float]], output_dir: str = "/Users/leo/Desktop"):
    """
    Plot the results from the results_search list with grouped bars for each metric.

    Args:
        results_search (List[Dict[str, float]]): List of dictionaries containing retrieval success data.
        output_dir (str): Directory to save the plots.
    """
    # Convert results_search to pandas DataFrame
    df = pd.DataFrame(results_search)
    a = Path(output_dir) / "prova.csv"
    df.to_csv(a)
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract unique aggregation methods
    agg_methods = df["agg_method"].unique()

    # Define specific colors
    palette = sns.color_palette("colorblind", n_colors=2)
    color_map = {"BagOfNodes": palette[0],  # Seaborn colorblind color
        "GraphEditDistance": palette[1],  # Seaborn colorblind color
        "BLOSUM": "#7f7f7f",  # Light gray
        "RMSD": "#141414",  # Darker gray
    }

    # Generate grayscale colors for other metrics dynamically if they are not in the map
    gray_shades = plt.cm.gray(np.linspace(0.3, 0.7, max(1, len(df["metric"].unique()) - len(color_map))))

    # Sort metrics by specific order
    metric_order = ["BagOfNodes", "GraphEditDistance", "BLOSUM", "RMSD"]
    metrics = df["metric"].unique()
    # Sort: If metric is in the list, use its index; otherwise put it at the end
    metrics = sorted(metrics, key=lambda x: metric_order.index(x) if x in metric_order else len(metric_order))

    # Identify functions by looking for columns that end with known score types
    functions_set = {col.split("_")[0] for col in df.columns if
                     any(score_key in col for score_key in ["precision@", "DCG", "NDCG", "AUROC"])}

    # Sort x-axis based on target_categories
    target_categories = ["METAL", "ATP", "GTP", "DNA", "RNA", "ATP+GTP", "ATP+METAL", "DNA+ATP", "DNA+GTP", "DNA+METAL",
                         "DNA+RNA", "GTP+METAL", "RNA+ATP", "RNA+GTP", "RNA+METAL", "ATP+GTP+METAL", "DNA+ATP+GTP",
                         "DNA+ATP+METAL", "DNA+RNA+ATP", "DNA+RNA+METAL", "RNA+ATP+GTP", "RNA+ATP+METAL", "RNA+GTP+METAL"]

    # Sort functions: present in target_categories first, then any others alphabetically
    functions = [f for f in target_categories if f in functions_set]
    remaining_functions = sorted(list(functions_set - set(functions)))
    functions.extend(remaining_functions)

    # Identify possible score types
    score_types = sorted(
        set(col.replace(func + "_", "") for col in df.columns for func in functions if func + "_" in col and "std" not in col))

    # For each aggregation method, create a figure for each score type
    for agg_method in agg_methods:
        # Filter data for the current aggregation method
        df_agg = df[df["agg_method"] == agg_method]

        for score_type in score_types:
            # --- Modification: Set style for whitegrid before plotting ---
            sns.set(style="whitegrid")

            plt.figure(figsize=(8, 4))
            ax = plt.gca()

            # Grouped bar arrangement
            bar_width = 0.8 / len(metrics)
            x_indexes = np.arange(len(functions))

            for i, metric in enumerate(metrics):
                # Build column names for mean and std
                col_mean = [f"{func}_{score_type}" for func in functions]
                col_std = [f"{func}_std_{score_type}" for func in functions]

                means = [df_agg.loc[df_agg["metric"] == metric, m].values[0] if m in df_agg.columns else 0 for m in col_mean]
                errors = [df_agg.loc[df_agg["metric"] == metric, s].values[0] if s in df_agg.columns else 0 for s in col_std]

                # Assign color for the current metric
                if metric in color_map:
                    color = color_map[metric]
                else:
                    # Fallback to gray shades if metric not in specific map
                    color_idx = min(i, len(gray_shades) - 1)
                    color = gray_shades[color_idx]

                plt.bar(x_indexes + i * bar_width, means, yerr=errors, capsize=3,  # Smaller caps
                        width=bar_width, label=metric, alpha=0.8, color=color, error_kw={"elinewidth": 0.8, "capthick": 0.8}
                        # Thinner lines
                        )

            # --- Modification: Font sizes and Labels ---
            plt.xlabel("Functions", fontsize=12)

            if "AUROC" in score_type:
                plt.ylabel("AUROC Score", fontsize=12)
            else:
                plt.ylabel("Values", fontsize=12)

            # plt.title(f"{score_type} Comparison for {agg_method.capitalize()} Aggregation", fontsize=14)

            plt.xticks(x_indexes + (bar_width * (len(metrics) - 1) / 2), functions, rotation=90, fontsize=10)
            plt.yticks(fontsize=10)

            # --- Modification: Border and Grid Styling ---
            # Black border on all four sides
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")

            # Remove all grid lines and add only horizontal dotted grey lines
            plt.grid(visible=False)  # Remove all grid lines first
            plt.grid(axis="y", linestyle="--", alpha=0.5, color="grey")  # Add only horizontal lines

            plt.tight_layout()

            # Save the plot
            plt.savefig(os.path.join(output_dir, f"{score_type}_{agg_method}.pdf"), bbox_inches="tight")
            plt.close()

    print(f"Plots saved to {output_dir}")

def main(args):
    # Set Seed
    np.random.seed(42)
    assert (
        args.input_pdb.exists()
    ), f"Input PDB directory {args.input_pdb} does not exist"
    assert args.input_graph.exists(), f"Fragment path {args.input_graph} does not exist"
    # Parse the PDB file tree
    pdb_data = parse_pdb_file_tree(args.input_pdb, args.input_graph)

    # Compute or load RMSD distance matrix
    rmsd_path = args.output_path / "rmsd_distance_matrix.csv"
    rmsd_distance = (
        pd.read_csv(rmsd_path, index_col=0)
        if rmsd_path.exists()
        else create_RMSD_distance_matrix_parallel(pdb_data)
    )
    if not rmsd_path.exists():
        rmsd_distance.to_csv(rmsd_path)

    # Compute or load BLOSUM distance matrix
    blosum_path = args.output_path / "blosum_distance_matrix.csv"
    blosum_distance = (
        pd.read_csv(blosum_path, index_col=0)
        if blosum_path.exists()
        else create_blosum_distance_matrix_parallel(pdb_data)
    )
    if not blosum_path.exists():
        blosum_distance.to_csv(blosum_path)

    # List of metrics to use
    metrics_list = [
        gm.BagOfNodes,
        # gm.WeisfeleirLehmanKernel,
        gm.GraphEditDistance,
    ]
    # Check if the first distance matrix for graphs exists
    graph_distance_path = (
        args.output_path / f"{metrics_list[0].__name__}_distance_matrix.csv"
    )
    if graph_distance_path.exists():
        graph_distance_dict = {}
        for metric in metrics_list:
            graph_distance_path = (
                args.output_path / f"{metric.__name__}_distance_matrix.csv"
            )
            graph_distance = pd.read_csv(graph_distance_path, index_col=0)
            graph_distance_dict[metric.__name__] = graph_distance
    else:
        graph_distance_dict = create_graph_distance_matrices(pdb_data, metrics_list)
        # Save the distance matrices to csv
        for metric, distance_matrix in graph_distance_dict.items():
            distance_matrix.to_csv(f"{metric}_distance_matrix.csv")

    # for metric, distance_matrix in graph_distance_dict.items():
    #     plot_distance_matrix_with_categories(
    #         distance_matrix,
    #         pdb_data,
    #         output_file=args.output_path / f"{metric}_distance_matrix_plot.pdf",
    #     )

    # Ensure RMSD distance is sorted by category
    rmsd_distance = rmsd_distance.reindex(
        index=get_sorted_pdb_keys(pdb_data), columns=get_sorted_pdb_keys(pdb_data)
    )
    blosum_distance = blosum_distance.reindex(
        index=get_sorted_pdb_keys(pdb_data), columns=get_sorted_pdb_keys(pdb_data)
    )
    # Replace diagonal zeroes with small positive values (if required)
    np.fill_diagonal(rmsd_distance.values, 0)
    np.fill_diagonal(blosum_distance.values, 0)
    # For RMSD:
    # plot_distance_matrix_with_categories(
    #     rmsd_distance,
    #     pdb_data,
    #     output_file=args.output_path / "RMSD_distance_matrix_plot.pdf",
    # )
    # # For BLOSUM:
    # plot_distance_matrix_with_categories(
    #     blosum_distance,
    #     pdb_data,
    #     output_file=args.output_path / "BLOSUM_distance_matrix_plot.pdf",
    # )

    # Combine RMSD and other distance matrices into one loop
    all_distance_matrices = {
        "RMSD": rmsd_distance,
        **graph_distance_dict,
        "BLOSUM": blosum_distance,
    }
    # Generate combined elbow plots
    generate_combined_elbow_plot(
        all_distance_matrices=all_distance_matrices,
        pdb_data=pdb_data,
        output_path=args.output_path,
    )
    # raise ValueError
    #
    # plot_function_comparison(
    #     pdb_data=pdb_data,
    #     rmsd_matrix=rmsd_distance,
    #     ged_matrix=graph_distance_dict["GraphEditDistance"],
    #     bag_nodes_matrix=graph_distance_dict["BagOfNodes"],
    #     blosum_matrix=blosum_distance,
    #     output_dir=args.output_path,
    # )
    create_ratio_csv(
        pdb_data=pdb_data,
        rmsd_matrix=rmsd_distance,
        ged_matrix=graph_distance_dict["GraphEditDistance"],
        bag_nodes_matrix=graph_distance_dict["BagOfNodes"],
        output_csv=args.output_path / "rmsd_ged_bag_ratios.csv",
    )

    # Plot correlation matrix
    # plot_correlation_matrix(
    #     all_distance_matrices, args.output_path / "correlation_matrix.pdf"
    # )

    results_search = []
    for metric, distance_matrix in all_distance_matrices.items():
        pdb_categories = [pdb_data[pdb]["category"] for pdb in distance_matrix.index]
        # # Perform UMAP, cluster, and plot confusion matrix
        # output_umap_file = args.output_path / f"{metric}_umap.html"
        # clustering_output_file = (
        #     args.output_path / f"{metric}_umap_confusion_matrix.pdf"
        # )
        # perform_umap_and_plot(
        #     distance_matrix=distance_matrix,
        #     categories=pdb_categories,
        #     metric_name=metric,
        #     output_file=output_umap_file,
        #     clustering_output_file=clustering_output_file,
        # )
        # # Do t-SNE and plot
        # output_tsne_file = args.output_path / f"{metric}_tsne.html"
        # perform_tsne_and_plot(distance_matrix, pdb_categories, metric, output_tsne_file)
        # # # Organize the distance matrix by category
        # n_categories = len(set(pdb_categories))  # Number of unique categories
        sorted_unique_categories = sort_categories(list(set(pdb_categories)))
        sorted_pdbs = sorted(
            distance_matrix.index,
            key=lambda pdb: (
                sorted_unique_categories.index(pdb_data[pdb]["category"]),
                pdb,
            ),
        )
        # Reorder the distance matrix
        distance_matrix_sorted = distance_matrix.loc[sorted_pdbs, sorted_pdbs]
        sorted_categories_sorted = [pdb_data[pdb]["category"] for pdb in sorted_pdbs]

        for agg_method in [
            np.mean,
        ]:  # np.max, np.median]:
            # Calculate retrieval success
            retrieval_success = calculate_retrieval_success(
                distance_matrix_sorted, sorted_categories_sorted, agg_method
            )
            row = {"agg_method": agg_method.__name__, "metric": metric}
            # Flatten and structure retrieval success for each function
            for func, values in retrieval_success.items():
                for key, value in values.items():
                    row[f"{func}_{key}"] = value
            results_search.append(row)
        # # Make directory for spider charts
        # spider_chart_dir = args.output_path / "spider_charts"
        # spider_chart_dir.mkdir(exist_ok=True)
        #
        # # plot_median_spider_charts(
        # #     distance_matrix=distance_matrix_sorted,
        # #     categories=sorted_categories_sorted,
        # #     metric_name=metric,
        # #     output_dir=spider_chart_dir,
        # #     opacity=0.2,
        # # )
        # # Generate the confusion-matrix-like heatmap using mean distances
        # plot_mean_matrix(
        #     distance_matrix_sorted,
        #     sorted_categories_sorted,
        #     unique_sorted_categories=sorted_unique_categories,
        #     output_file=args.output_path / f"{metric}_mean_distance_heatmap.pdf",
        #     normalize=True,
        # )
        #
        # # Compute clustering error matrix using hierarchical clustering
        # error_matrix = clustering_error_matrix_hierarchical(
        #     distance_matrix, pdb_categories, n_categories
        # )
        # error_matrix.to_csv(args.output_path / f"{metric}_hierarchical_clustering_error_matrix.csv")
        #
        # # Print summary of mean clustering error
        # mean_error = error_matrix.mean().mean()
        # print(f"Mean clustering error for {metric} (hierarchical): {mean_error:.4f}")
        #
        # # Predict clusters using hierarchical clustering
        # distance_matrix.values[:] = np.clip(
        #     distance_matrix.values, 0, None
        # )  # Clip negatives
        # linkage_matrix = linkage(squareform(distance_matrix.values), method="average")
        # predicted_clusters = fcluster(
        #     linkage_matrix, n_categories, criterion="maxclust"
        # )
        #
        # # Plot confusion matrix
        # # Match clusters to true labels and plot confusion matrix
        # plot_cluster_confusion_matrix(
        #     true_labels=np.array(pdb_categories),
        #     predicted_clusters=predicted_clusters,
        #     output_file=args.output_path / f"{metric}_cluster_confusion_matrix.pdf",
        # )
        # # Evaluate clustering
        # print(f"\nEvaluating clustering for {metric}:")
        #
        # # 1. Adjusted Rand Index (ARI)
        # ari_score = adjusted_rand_score(pdb_categories, predicted_clusters)
        # print(f"ARI: {ari_score:.4f}")
        #
        # # 2. Normalized Mutual Information (NMI)
        # nmi_score = normalized_mutual_info_score(pdb_categories, predicted_clusters)
        # print(f"NMI: {nmi_score:.4f}")
        #
        # # 3. Homogeneity, Completeness, and V-Measure
        # h_score = homogeneity_score(pdb_categories, predicted_clusters)
        # c_score = completeness_score(pdb_categories, predicted_clusters)
        # v_score = v_measure_score(pdb_categories, predicted_clusters)
        # print(
        #     f"Homogeneity: {h_score:.4f}, Completeness: {c_score:.4f}, V-Measure: {v_score:.4f}"
        # )
        #
        # # 4. Silhouette Score
        # silhouette_avg = silhouette_score(
        #     distance_matrix.values, predicted_clusters, metric="precomputed"
        # )
        # print(f"Silhouette Score: {silhouette_avg:.4f}")
    plot_results_from_search(results_search)
    print(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse PDB files and find corresponding FG files"
    )
    parser.add_argument(
        "--input_pdb", type=Path, help="Path to the PDB directory", required=True
    )
    parser.add_argument(
        "--input_graph",
        type=Path,
        help="Path to the FG fragment directory",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to the output directory",
        required=True,
    )

    args = parser.parse_args()
    main(args)
