import pickle
import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from tessera.difference_fn.probability_processing import FragmentDetail
from tessera.fragments.classification_config import (
    categorical_edge_attrs,
    categorical_node_attrs,
    max_fragment_length,
)


def compute_distance_matrix(backbone_coords: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Euclidean distance matrix for an array of Ca xyz coordinates.

    Args:
        backbone_coords (np.ndarray): Array of shape (n, 3) representing xyz coordinates.

    Returns:
        np.ndarray: A (n, n) distance matrix.
    """
    # Compute pairwise squared differences
    diff = backbone_coords[:, np.newaxis, :] - backbone_coords[np.newaxis, :, :]

    # Compute squared Euclidean distances and take square root
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    return distance_matrix


class StructureFragmentGraph:
    def __init__(
        self,
        structure_fragment: "StructureFragment",
        graph: nx.Graph,
        distance_matrix: np.ndarray,
    ) -> None:
        self.structure_fragment = structure_fragment
        self.graph = graph
        self.distance_matrix = distance_matrix

    def __repr__(self):
        return f"{self.__class__.__name__}(structure_path={self.structure_fragment.structure_path}, graph_dataset={self.graph}, distance_matrix={self.distance_matrix.shape})"

    @staticmethod
    def from_structure_fragment(
        structure_fragment: "StructureFragment",
        edge_distance_threshold: float = 10,
        extra_graph_attributes: t.Dict[str, t.Any] = {},
    ):
        """
        Create a graph_dataset from a StructureFragment object

        Parameters
        ----------
        structure_fragment: StructureFragment
            The structure fragment to create a graph_dataset from
        edge_distance_threshold: float
            The threshold determining two nodes are connected by an edge based on Euclidean distance - In Angstroms.
        extra_graph_attributes: Dict[str, Any]
            Extra attributes to add to the graph_dataset. NB: These are graph_dataset level attributes, not node or edge attributes
        """
        ca_distance_matrix = compute_distance_matrix(structure_fragment.backbone_coords)
        # Aggregate matrix based on fragments
        (
            fragment_euclidean_distance_matrix,
            fragment_resn_distance_matrix,
        ) = create_fragment_distance_matrix(
            ca_distance_matrix, structure_fragment.classification_map
        )

        # Create a graph_dataset from the fragment distance matrix
        graph = create_graph_from_fragment_matrix(
            fragment_euclidean_distance_matrix,
            fragment_resn_distance_matrix,
            edge_distance_threshold,
            structure_fragment.classification_map,
            structure_fragment.probability_distance_data,
            structure_fragment.angles,
        )
        # Add PDB name as name for the graph_dataset
        graph.name = structure_fragment.structure_path.stem
        # Add extra attributes to the nx graph_dataset
        if extra_graph_attributes:
            for key, value in extra_graph_attributes.items():
                graph.graph[key] = value

        return StructureFragmentGraph(
            structure_fragment=structure_fragment,
            graph=graph,
            distance_matrix=fragment_euclidean_distance_matrix,
        )

    @staticmethod
    def flatten_attributes(attr: t.Any) -> t.List:
        """Flatten attributes if they are lists or numpy arrays."""
        if isinstance(attr, list) or isinstance(attr, np.ndarray):
            return list(attr)
        return [attr]

    def save(self, file_path: Path, verbose: bool = True) -> None:
        StructureFragmentGraphIO.save(self, file_path, verbose)

    @staticmethod
    def load(file_path: Path) -> "StructureFragmentGraph":
        return StructureFragmentGraphIO.load(file_path)

    def to_pyg(
        self, node_attributes: t.Iterable[str], edge_attributes: t.Iterable[str]
    ) -> Data:
        return StructureFragmentGraphIO.to_pyg(self, node_attributes, edge_attributes)

    def save_graph(self, file_path: Path) -> None:
        StructureFragmentGraphIO.save_graph(self.graph, file_path)

    def update_graph_attributes(self, attributes: t.Dict[str, t.Any]) -> None:
        for key, value in attributes.items():
            if isinstance(value, (list, np.ndarray)):
                value = self.flatten_attributes(value)
            self.graph.graph[key] = value


class StructureFragmentGraphIO:
    @staticmethod
    def save(
        obj: StructureFragmentGraph, file_path: Path, verbose: bool = True
    ) -> None:
        """
        Save the entire StructureFragmentGraph object to a file.

        Parameters
        ----------
        obj: StructureFragmentGraph
            The object to save.
        file_path: Path
            The path to the file where the object will be saved.
        """
        if file_path.suffix != ".fg":
            file_path = file_path.with_suffix(".fg")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        if verbose:
            print(f"Saved StructureFragmentGraph to {file_path}")

    @staticmethod
    def load(file_path: Path) -> StructureFragmentGraph:
        """
        Load a StructureFragmentGraph object from a file.

        Parameters
        ----------
        file_path: Path
            The path to the file from which the object will be loaded.

        Returns
        -------
        StructureFragmentGraph
            The loaded StructureFragmentGraph object.
        """
        if file_path.suffix != ".fg":
            raise ValueError("Invalid file extension. Expected '.fg'")
        with open(file_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def to_pyg(
        obj: StructureFragmentGraph,
        node_attributes: t.Iterable[str],
        edge_attributes: t.Iterable[str],
    ) -> Data:
        edge_index = []
        node_categorical_attrs = []
        node_continuous_attrs = []
        edge_categorical_attrs = []
        edge_continuous_attrs = []

        for node, data in obj.graph.nodes(data=True):
            node_cat_attrs = []
            node_cont_attrs = []
            for attr in node_attributes:
                flattened_attr = obj.flatten_attributes(data[attr])
                if attr in categorical_node_attrs:
                    node_cat_attrs.extend(flattened_attr)
                else:
                    node_cont_attrs.extend(flattened_attr)
            node_categorical_attrs.append(node_cat_attrs)
            node_continuous_attrs.append(node_cont_attrs)

        for u, v, data in obj.graph.edges(data=True):
            edge_index.append([u - 1, v - 1])  # Convert to 0-based index
            edge_cat_attrs = []
            edge_cont_attrs = []
            for attr in edge_attributes:
                flattened_attr = obj.flatten_attributes(data[attr])
                if attr in categorical_edge_attrs:
                    edge_cat_attrs.extend(flattened_attr)
                else:
                    edge_cont_attrs.extend(flattened_attr)
            edge_categorical_attrs.append(edge_cat_attrs)
            edge_continuous_attrs.append(edge_cont_attrs)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        node_categorical_attrs = torch.tensor(node_categorical_attrs, dtype=torch.long)
        node_continuous_attrs = torch.tensor(node_continuous_attrs, dtype=torch.float)
        edge_categorical_attrs = torch.tensor(edge_categorical_attrs, dtype=torch.long)
        edge_continuous_attrs = torch.tensor(edge_continuous_attrs, dtype=torch.float)

        node_features = torch.cat(
            (node_categorical_attrs, node_continuous_attrs), dim=-1
        )
        edge_features = torch.cat(
            (edge_categorical_attrs, edge_continuous_attrs), dim=-1
        )

        feature_type_map = {
            "node_categorical": (0, node_categorical_attrs.shape[1]),
            "node_continuous": (
                node_categorical_attrs.shape[1],
                node_features.shape[1],
            ),
            "edge_categorical": (0, edge_categorical_attrs.shape[1]),
            "edge_continuous": (
                edge_categorical_attrs.shape[1],
                edge_features.shape[1],
            ),
        }

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            feature_type_map=feature_type_map,
            name=obj.graph.name,
        )

        return data

    @staticmethod
    def save_graph(graph: nx.Graph, file_path: Path) -> None:
        nx.write_gpickle(graph, file_path)


def create_fragment_distance_matrix(
    ca_distance_matrix: np.ndarray, classification_map: t.List[FragmentDetail]
) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Converts a distance matrix of amino acids into a matrix of fragment distances.
    """
    num_fragments = len(classification_map)
    fragment_euclidean_distance_matrix = np.zeros((num_fragments, num_fragments))
    fragment_resn_distance_matrix = np.zeros((num_fragments, num_fragments))

    # Fill out the matrix, optimizing by only calculating each pair once
    for i, frag1 in enumerate(classification_map):
        for j in range(i, num_fragments):
            frag2 = classification_map[j]
            distance_submatrix = ca_distance_matrix[
                frag1.start_idx : frag1.end_idx + 1, frag2.start_idx : frag2.end_idx + 1
            ]
            min_euclidean_distance = np.min(distance_submatrix)

            # Compute sequence distance
            if i == j:
                min_resn_distance = 0
            else:
                min_resn_distance = frag2.start_idx - frag1.end_idx

            assert (
                min_resn_distance >= 0
            ), f"Residue distance should be positive, but got {min_resn_distance}"
            # Symmetrically assign values
            fragment_euclidean_distance_matrix[
                i, j
            ] = fragment_euclidean_distance_matrix[j, i] = min_euclidean_distance
            fragment_resn_distance_matrix[i, j] = fragment_resn_distance_matrix[
                j, i
            ] = min_resn_distance

    return fragment_euclidean_distance_matrix, fragment_resn_distance_matrix


def create_graph_from_fragment_matrix(
    fragment_distance_matrix: np.ndarray,
    fragment_resn_distance_matrix: np.ndarray,
    threshold: float,
    classification_map: t.List[FragmentDetail],
    probability_distance_data: np.ndarray,
    angles: np.ndarray,
) -> nx.Graph:
    """
    Creates a graph_dataset from a distance matrix by applying a threshold, with node labels derived from a classification map and probability distance data_paths.
    Nodes represent protein fragments, and edges are defined by distances that are within a specified threshold.
    Edges will also include Euclidean distance, residue sequence distance, and a flag indicating if they are connected by a peptide bond.

    Parameters:
    ----------
    fragment_distance_matrix : np.ndarray
        The distance matrix where each element represents the Euclidean distance between two fragments.
    fragment_resn_distance_matrix : np.ndarray
        The distance matrix where each element represents the sequence distance between two fragments.
    threshold : float
        The threshold for determining if an edge should exist between two nodes based on Euclidean distance.
    classification_map : List[FragmentDetail]
        A list containing details of each fragment, used to label the nodes in the graph_dataset.
    probability_distance_data : np.ndarray
        An array containing probability distances for each amino acid, used as node features.
    angles : np.ndarray
        An array containing the angles between each amino acid, used as node features.

    Returns:
    -------
    nx.Graph
        A NetworkX graph_dataset where nodes represent fragments and are labeled according to `classification_map`.
        Edges exist where distances are less than or equal to the threshold and include additional attributes.
    """
    # Create an empty graph_dataset
    graph = nx.Graph()

    # Label nodes based on classification map and add probability data_paths
    for idx, frag_detail in enumerate(classification_map):
        # Extract and pad the probability data_paths for each fragment
        prob_data = probability_distance_data[
            frag_detail.start_idx : frag_detail.end_idx + 1
        ]
        padded_prob_data = np.zeros(
            (max_fragment_length, 40)
        )  # 40, one for each fragment
        padded_prob_data[: prob_data.shape[0], : prob_data.shape[1]] = prob_data
        # Extract and pad the angles data_paths for each fragment
        angles_data = np.zeros((max_fragment_length, 2))  # 2 angles - phi and psi
        fragment_angles = angles[frag_detail.start_idx : frag_detail.end_idx + 1]
        angles_data[
            : fragment_angles.shape[0], : fragment_angles.shape[1]
        ] = fragment_angles

        # Add node with features
        graph.add_node(
            idx + 1,  # Use 1-based index
            label=f"({idx + 1}) \n Frag. {frag_detail.fragment_class}",
            # start_idx=frag_detail.start_idx, # TODO: Remove - idx is too variable and not easily encoded
            # end_idx=frag_detail.end_idx, # TODO: Remove -  idx is too variable and not easily encoded
            percentage_index=idx / len(classification_map),
            probability_data=padded_prob_data.flatten(),
            angles_data=angles_data.flatten(),
            fragment_class=frag_detail.fragment_class,
        )

    # Iterate over each pair of nodes to determine edges
    for i in range(len(classification_map)):
        for j in range(
            i + 1, len(classification_map)
        ):  # Only consider i < j to avoid duplicate edges
            if fragment_distance_matrix[i, j] <= threshold:
                # Determine if the bond is a peptide bond
                is_peptide_bond = (
                    classification_map[i].end_idx + 1 == classification_map[j].start_idx
                )
                # Add edge with attributes
                graph.add_edge(
                    i + 1,
                    j + 1,
                    euclidean_distance=fragment_distance_matrix[i, j],
                    # resn_distance=fragment_resn_distance_matrix[i, j], # TODO: Remove -  idx is too variable and not easily encoded or change to %
                    peptide_bond=is_peptide_bond,
                )

    return graph


def plot_graph(
    G: nx.Graph,
    layout="spring",
    node_color="lightblue",
    edge_color="gray",
    node_size=500,
    with_labels=True,
    label_font_size=12,
    title="",
    save_path=None,
):
    """
    Plots a NetworkX graph_dataset with customizable layout and style, highlighting peptide bonds in a rainbow spectrum with black borders and non-peptide bonds as dotted lines.
    """
    # Select the layout
    if layout == "spring":
        pos = nx.spring_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G)

    plt.title(title)

    # Identify nodes with "Frag. 0" and others
    frag_0_nodes = [
        node for node, data in G.nodes(data=True) if "Frag. 0" in data["label"]
    ]
    non_frag_0_nodes = [node for node in G.nodes if node not in frag_0_nodes]

    # Draw nodes for "Frag. 0" (smaller and grey)
    nx.draw_networkx_nodes(
        G, pos, nodelist=frag_0_nodes, node_color="grey", node_size=node_size * 0.5
    )

    # Draw nodes for others
    nx.draw_networkx_nodes(
        G, pos, nodelist=non_frag_0_nodes, node_color=node_color, node_size=node_size
    )

    # Draw edges
    peptide_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("peptide_bond", False)
    ]
    other_edges = [
        (u, v) for u, v, d in G.edges(data=True) if not d.get("peptide_bond", False)
    ]

    if peptide_edges:
        color_map = plt.get_cmap("rainbow_r")
        peptide_colors = [
            color_map(i / len(peptide_edges)) for i in range(len(peptide_edges))
        ]
    else:
        peptide_colors = []

    nx.draw_networkx_edges(
        G, pos, edgelist=other_edges, edge_color=edge_color, style="dotted"
    )

    for idx, edge in enumerate(peptide_edges):
        nx.draw_networkx_edges(
            G, pos, edgelist=[edge], edge_color="black", style="solid", width=4
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[edge],
            edge_color=peptide_colors[idx],
            style="solid",
            width=2,
        )

    # Draw labels, excluding "Frag. 0"
    if with_labels:
        labels = {
            node: data["label"]
            for node, data in G.nodes(data=True)
            if "Frag. 0" not in data["label"]
        }
        nx.draw_networkx_labels(
            G, pos, labels=labels, font_size=label_font_size, font_color="black"
        )

    # Hide axes
    plt.axis("off")
    ax = plt.gca()
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Save plot to PDF if a path is provided
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    # Show the plot
    plt.show()
