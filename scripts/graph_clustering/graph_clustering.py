import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import networkx as nx
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tessera.fragments.fragments_classifier import EnsembleFragmentClassifier


class GraphProcessor:
    def __init__(
        self,
        classifier: EnsembleFragmentClassifier,
        node_attributes: List[str],
        edge_attributes: List[str],
    ):
        self.classifier = classifier
        self.node_attributes = node_attributes
        self.edge_attributes = edge_attributes

    def classify_and_load_nx_graph(self, pdb_file_path: Path) -> nx.Graph:
        """Classify and load PDB file as NetworkX graph_dataset."""
        assert pdb_file_path.exists(), f"Structure path {pdb_file_path} does not exist"
        structure_fragment = self.classifier.classify_to_fragment(
            pdb_file_path, use_all_fragments=True
        )
        graph = structure_fragment.to_graph(edge_distance_threshold=10)
        return graph.graph

    @staticmethod
    def save_graph(graph: nx.Graph, graph_file_path: Path) -> None:
        """Save NetworkX graph_dataset to file."""
        with open(graph_file_path, "wb") as f:
            pickle.dump(graph, f)
        print(f"Saved graph_dataset to {graph_file_path}")

    @staticmethod
    def load_graph(graph_file_path: Path) -> nx.Graph:
        """Load NetworkX graph_dataset from file."""
        if graph_file_path.exists():
            with open(graph_file_path, "rb") as f:
                graph = pickle.load(f)
            print(f"Loaded graph_dataset from {graph_file_path}")
            return graph
        return None

    @staticmethod
    def download_pdb(pdb_code: str, download_path: Path) -> None:
        """Download PDB file from RCSB."""
        pdb_url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
        response = requests.get(pdb_url)
        if response.status_code == 200:
            with open(download_path, "wb") as file:
                file.write(response.content)
            print(f"Downloaded {pdb_code} to {download_path}")
        else:
            print(f"Failed to download {pdb_code}")

    def load_pdb_files(
        self, cath_df: pd.DataFrame, pdb_path: Path, graph_dir: Path
    ) -> List[Tuple[nx.Graph, str]]:
        """Load and classify PDB files, returning a list of graphs with their class labels."""
        selected_proteins = self._select_proteins(cath_df)
        graphs = []

        for class_name, pdb_codes in selected_proteins.items():
            print(f"Class {class_name}:")
            for pdb_code in pdb_codes:
                graph_file_dir = graph_dir / class_name
                graph_file_dir.mkdir(parents=True, exist_ok=True)
                graph_file_path = graph_file_dir / f"{pdb_code}.pkl"

                nx_graph = self.load_graph(graph_file_path)
                if not nx_graph:
                    pdb_file_path = self._get_pdb_file_path(pdb_code, pdb_path)
                    if not pdb_file_path.exists():
                        self.download_pdb(pdb_code, pdb_file_path)

                    nx_graph = self.classify_and_load_nx_graph(pdb_file_path)
                    if nx_graph:
                        self.save_graph(nx_graph, graph_file_path)
                        graphs.append((nx_graph, class_name))
                        print(f"Successfully processed {pdb_file_path}")
                    else:
                        raise ValueError(f"Failed to process {pdb_file_path}")
                else:
                    graphs.append((nx_graph, class_name))
            print()
        return graphs

    def _select_proteins(self, cath_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Select proteins from the CATH dataset."""
        selected_proteins = {}
        for class_column in cath_df.columns[1:]:
            class_proteins = cath_df[cath_df[class_column] == 1]
            selected_class_proteins = class_proteins.head(10)
            pdb_codes = selected_class_proteins["PDB-chain"].apply(
                lambda x: x.split("-")[0]
            )
            selected_proteins[class_column] = pdb_codes.tolist()
        return selected_proteins

    def _get_pdb_file_path(self, pdb_code: str, pdb_path: Path) -> Path:
        """Get the file path for the PDB file."""
        middle_letters = pdb_code[1:3]
        pdb_file_dir = pdb_path / middle_letters
        pdb_file_dir.mkdir(parents=True, exist_ok=True)
        return pdb_file_dir / f"{pdb_code}.pdb"

    @staticmethod
    def flatten_attributes(attr: Any) -> List:
        """Flatten attributes if they are lists or numpy arrays."""
        if isinstance(attr, list) or isinstance(attr, np.ndarray):
            return list(attr)
        return [attr]

    def compute_graph_features(self, graph: nx.Graph) -> List[float]:
        """Compute features for a given graph_dataset."""
        features = []

        # Graph-level features
        num_nodes = nx.number_of_nodes(graph)
        num_edges = nx.number_of_edges(graph)
        avg_degree = np.mean([d for n, d in graph.degree()]) if num_nodes > 0 else 0
        diameter = nx.diameter(graph) if nx.is_connected(graph) and num_nodes > 1 else 0
        avg_clustering = nx.average_clustering(graph) if num_nodes > 1 else 0

        features.extend([num_nodes, num_edges, avg_degree, diameter, avg_clustering])

        # Node features
        for attr in self.node_attributes:
            curr_node_attrs = []
            for _, data in graph.nodes(data=True):
                if attr in data:
                    curr_node_attrs.extend(self.flatten_attributes(data[attr]))
            if curr_node_attrs:
                curr_node_attrs = np.array(curr_node_attrs)
                features.extend(
                    [
                        np.mean(curr_node_attrs),
                        np.var(curr_node_attrs),
                        np.min(curr_node_attrs),
                        np.max(curr_node_attrs),
                    ]
                )
            else:
                features.extend([0, 0, 0, 0])

        # Edge features
        for attr in self.edge_attributes:
            curr_edge_attrs = []
            for _, _, data in graph.edges(data=True):
                if attr in data:
                    curr_edge_attrs.extend(self.flatten_attributes(data[attr]))
            if curr_edge_attrs:
                curr_edge_attrs = np.array(curr_edge_attrs)
                features.extend(
                    [
                        np.mean(curr_edge_attrs),
                        np.var(curr_edge_attrs),
                        np.min(curr_edge_attrs),
                        np.max(curr_edge_attrs),
                    ]
                )
            else:
                features.extend([0, 0, 0, 0])

        return features


def plot_pca(embeddings: np.ndarray, labels: List[str]) -> None:
    """Plot PCA results with sorted labels using the viridis colormap."""
    plt.figure(figsize=(10, 8))

    # Sort labels and embeddings accordingly
    sorted_indices = np.argsort(labels)
    sorted_embeddings = embeddings[sorted_indices]
    sorted_labels = np.array(labels)[sorted_indices]

    # Get unique labels and their colors from the viridis colormap
    unique_labels = sorted(set(sorted_labels))
    cmap = plt.cm.get_cmap("viridis", len(unique_labels))

    for i, label in enumerate(unique_labels):
        idx = [j for j, l in enumerate(sorted_labels) if l == label]
        plt.scatter(
            sorted_embeddings[idx, 0],
            sorted_embeddings[idx, 1],
            color=cmap(i),
            label=label,
            alpha=0.7,
        )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA of Graph Embeddings")
    plt.legend()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select and classify PDB structures from CATH dataset."
    )
    parser.add_argument(
        "--cath_file", type=str, required=True, help="Path to the CATH dataset file"
    )
    parser.add_argument(
        "--pdb_path", type=str, required=True, help="Base path to the PDB dataset"
    )
    parser.add_argument(
        "--fragment_path", type=str, required=True, help="Path to the fragment data_paths"
    )
    parser.add_argument(
        "--graph_dir", type=str, required=True, help="Directory to save and load graphs"
    )
    parser.add_argument(
        "--node_attributes",
        type=str,
        nargs="+",
        required=True,
        help="List of node attributes",
    )
    parser.add_argument(
        "--edge_attributes",
        type=str,
        nargs="+",
        required=True,
        help="List of edge attributes",
    )
    parser.add_argument(
        "--num_clusters", type=int, default=4, help="Number of clusters to create"
    )

    args = parser.parse_args()

    cath_df = pd.read_csv(args.cath_file)
    print(cath_df.head())

    classifier = EnsembleFragmentClassifier(
        Path(args.fragment_path),
        difference_types=["angle", "angle"],
        difference_names=["LogPr", "RamRmsd"],
        n_processes=10,
        step_size=1,
    )

    processor = GraphProcessor(classifier, args.node_attributes, args.edge_attributes)
    graphs = processor.load_pdb_files(
        cath_df, Path(args.pdb_path), Path(args.graph_dir)
    )

    feature_vectors = [processor.compute_graph_features(graph) for graph, _ in graphs]
    labels = [class_name for _, class_name in graphs]

    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_vectors)

    pca = PCA(n_components=2)
    graph_embeddings = pca.fit_transform(normalized_features)

    plot_pca(graph_embeddings, labels)


if __name__ == "__main__":
    main()
