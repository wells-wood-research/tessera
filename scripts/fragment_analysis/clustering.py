import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import ampal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
)

from src.difference_fn.angle_difference import AngleDifferenceStrategy
from src.difference_fn.difference_processing import select_first_ampal_assembly
from src.difference_fn.difference_selector import difference_function_selector
from src.difference_fn.ramachandran_difference import (
    RamachandranCircularKDE,
    RamachandranNormalKDE,
    RamachandranProjected3DKDE,
)
from src.fragments.reference_fragments import ReferenceFragmentCreator


def get_fragment_color(fragment_id, fragment_to_class):
    fragment_id = int(fragment_id)
    if fragment_id in fragment_to_class["alpha"]:
        color= 'blue'
    elif fragment_id in fragment_to_class["beta"]:
        color = 'orange'
    elif fragment_id in fragment_to_class["alpha_beta"]:
        color = 'green'
    else:
        raise ValueError(f"Fragment ID {fragment_id} not found in any class.")

    print(f"Fragment ID {fragment_id} has color {color}")
    return color

def plot_ramachandran(phi: np.ndarray, psi: np.ndarray, filename: Path):
    """
    Generate a Ramachandran plot for given phi and psi angles.

    Parameters:
        phi (np.ndarray): Array of phi angles in radians.
        psi (np.ndarray): Array of psi angles in radians.
        filename (Path): Path to save the plot as a PDF.
    """
    # Convert angles from radians to degrees
    phi_degrees = np.degrees(phi)
    psi_degrees = np.degrees(psi)

    # Ensure all values are within the range [-180, 180] degrees
    assert np.all(
        (phi_degrees >= 0) & (phi_degrees <= 360)
    ), "Phi values are out of range."
    assert np.all(
        (psi_degrees >= 0) & (psi_degrees <= 360)
    ), "Psi values are out of range."

    plt.figure(figsize=(8, 8))
    sns.kdeplot(x=phi_degrees, y=psi_degrees, cmap="viridis", fill=True, bw_adjust=0.5)
    plt.title("Ramachandran Plot")
    plt.xlabel("Phi (degrees)")
    plt.ylabel("Psi (degrees)")
    plt.xlim(0, 360)
    plt.ylim(0, 360)
    plt.grid(True)
    plt.savefig(filename, format="pdf")
    plt.close()


def calculate_distance(kde_calculator, frag_id1, frag_id2, fragment_to_pdb_paths):

    # Placeholder for actual distance calculation
    angles1 = fragment_to_pdb_paths[frag_id1]
    angles2 = fragment_to_pdb_paths[frag_id2]
    distance = kde_calculator.calculate_difference(angles1, angles2)
    return distance


def calculate_distance_matrix(kde_calculator, fragment_ids, fragment_to_ampal_obj):
    num_fragments = len(fragment_ids)
    distance_matrix = np.zeros((num_fragments, num_fragments))
    args = [
        (kde_calculator, fragment_ids[i], fragment_ids[j], fragment_to_ampal_obj)
        for i in range(num_fragments)
        for j in range(i + 1, num_fragments)
    ]

    with Pool(cpu_count() - 2) as pool:
        results = pool.starmap(calculate_distance, args)

    # For debugging, run the calculations sequentially
    # results = []
    # for arg in args:
    #     results.append(calculate_distance(*arg))

    index = 0
    for i in range(num_fragments):
        for j in range(i + 1, num_fragments):
            distance = results[index]
            distance_matrix[i, j] = distance_matrix[j, i] = distance
            index += 1

    return distance_matrix


def main(args):
    """
    Main function to process fragments and perform hierarchical clustering.

    Parameters:
        args: Command line arguments.
    """
    fragment_path = Path(args.fragment_path)
    if not fragment_path.exists():
        raise FileNotFoundError(f"Input file {args.fragment_path} does not exist")

    # Load all the fragments
    difference_fn = difference_function_selector("logpr", "angle")
    id_to_fragment_dict = ReferenceFragmentCreator(
        folder_path=fragment_path, difference_fn=difference_fn
    ).create_all_fragments()
    fragment_to_class = {
        "alpha": [
            1, 2, 4, 5, 17, 18, 21, 22, 24, 27, 28, 29, 30, 31, 38, 39
        ],
        "beta": [10, 11, 12, 13, 15, 20, 25, 26],
        "alpha_beta": [3, 6, 7, 8, 9, 14, 16, 19, 20, 23, 32, 33, 34, 35, 36, 37, 40]
    }

    fragment_to_ampal_obj = {}
    for fragment_id, fragment in id_to_fragment_dict.items():
        ampal_fragment = ampal.load_pdb(str(fragment.paths[0]))
        ampal_fragment = select_first_ampal_assembly(ampal_fragment)
        phi_psi_angles = AngleDifferenceStrategy.get_ampal_data(ampal_fragment)
        phi_psi_angles = AngleDifferenceStrategy.normalise_angles(phi_psi_angles)
        phi_psi_angles = np.radians(phi_psi_angles)
        # plot_ramachandran(
        #     phi_psi_angles[:, 0],
        #     phi_psi_angles[:, 1],
        #     fragment_path / f"{fragment_id}_ramachandran.pdf",
        # )
        fragment_to_ampal_obj[fragment_id] = ampal_fragment

    # Initialize a dictionary to store cluster labels for each method
    cluster_labels_dict = {}

    kde_calculators = {
        "circular": RamachandranCircularKDE(),
        "projected_3d": RamachandranProjected3DKDE(),
        "normal": RamachandranNormalKDE(),
    }

    fragment_ids = list(id_to_fragment_dict.keys())
    fragment_ids.sort()
    for kde_name, kde_calculator in kde_calculators.items():
        distance_matrix = calculate_distance_matrix(kde_calculator, fragment_ids, fragment_to_ampal_obj)
        linkage_matrix = linkage(squareform(distance_matrix), method="ward")

        # Get the colors for the labels
        label_colors = [get_fragment_color(frag_id, fragment_to_class) for frag_id in fragment_ids]

        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix, labels=fragment_ids)

        # Get the current figure's axis
        ax = plt.gca()

        # Get the actual labels as they appear on the dendrogram
        xlbls = ax.get_xmajorticklabels()
        # Apply the correct colors based on the actual label text
        for lbl in xlbls:
            lbl.set_color(get_fragment_color(lbl.get_text(), fragment_to_class))

        plt.title(f"Hierarchical Clustering Dendrogram ({kde_name})", fontsize=18)
        # Fix x ticks size
        plt.xticks(fontsize=15, rotation=90)
        plt.xlabel("Fragment ID", fontsize=18)
        plt.ylabel("Distance", fontsize=18)
        plt.savefig(fragment_path / f"{kde_name}_clustering_dendrogram.pdf", format="pdf")
        plt.close()

    # Preparing the matrix to store ARI values
    # methods = list(cluster_labels_dict.keys())
    # num_methods = len(methods)
    # ari_matrix = np.zeros((num_methods, num_methods))

    # # Fill the matrix with ARI scores
    # for i in range(num_methods):
    #     for j in range(i, num_methods):
    #         labels_i = cluster_labels_dict[methods[i]]
    #         labels_j = cluster_labels_dict[methods[j]]
    #
    #         ari = adjusted_rand_score(labels_i, labels_j)
    #         ari_matrix[i, j] = ari
    #         ari_matrix[j, i] = ari
    #
    # # Plotting the ARI matrix as a heatmap
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(ari_matrix, xticklabels=methods, yticklabels=methods, annot=True, fmt=".2f", cmap="viridis")
    # plt.title("Adjusted Rand Index (ARI) Similarity Matrix")
    # plt.xlabel("Clustering Method")
    # plt.ylabel("Clustering Method")
    # plt.savefig(fragment_path / "clustering_similarity_matrix_ari.pdf", format="pdf")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical Clustering of Ramachandran Fragments"
    )
    parser.add_argument(
        "--fragment_path",
        type=str,
        required=True,
        help="Path to input fragments directory",
    )
    args = parser.parse_args()
    main(args)
