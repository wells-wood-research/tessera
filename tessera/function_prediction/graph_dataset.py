import typing as t
from pathlib import Path

import torch
from torch_geometric.data import Dataset
from tqdm import tqdm

from tessera.fragments.fragments_classifier import EnsembleFragmentClassifier
from tessera.fragments.fragments_graph import StructureFragmentGraph


class GraphCreator:
    """
    Given a dictionary of UniProt Proteins, download the corresponding PDB files and create a graph.
    """

    def __init__(
        self,
        classifier: EnsembleFragmentClassifier,
        output_dir: Path,
        verbose: bool = True,
    ):
        """
        Initialize the GraphCreator class with the classifier, output directory, and verbosity.

        Parameters
        ----------
        classifier: EnsembleFragmentClassifier
            The classifier to use for creating the graphs.
        output_dir: Path
            Directory to save the generated graphs.
        verbose: bool
            Enable verbose mode to print more information.
        """
        self.output_dir = output_dir / "uniprot_graphs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.classifier = classifier
        self.error_list: t.Set[str] = set()
        self.error_file = self.output_dir / "error.txt"
        self.verbose = verbose

    def classify_and_save_graph(
        self,
        pdb_file_path: Path,
        edge_distance_threshold: float = 10.0,
        extra_graph_attributes: t.Dict[str, t.Any] = {},
    ) -> Path:
        """
        Classify a PDB file to a fragment and save the corresponding graph_dataset.

        Parameters
        ----------
        pdb_file_path: Path
            Path to the PDB file.
        edge_distance_threshold: float
            Distance threshold for edges.
        extra_graph_attributes: t.Dict[str, t.Any]
            Extra attributes to add to the graph_dataset.

        Returns
        -------
        fragment_graph_path: Path
            Path to the saved graph_dataset.
        """
        if not pdb_file_path.exists():
            raise FileNotFoundError(f"Structure path {pdb_file_path} does not exist")

        pdb_code = pdb_file_path.stem
        fragment_graph_path = self.output_dir / f"{pdb_code}.fg"

        if not fragment_graph_path.exists():
            try:
                structure_fragment = self.classifier.classify_to_fragment(
                    pdb_file_path, use_all_fragments=True
                )
                structure_fragment_graph = (
                    StructureFragmentGraph.from_structure_fragment(
                        structure_fragment,
                        edge_distance_threshold=edge_distance_threshold,
                        extra_graph_attributes=extra_graph_attributes,
                    )
                )
                structure_fragment_graph.save(fragment_graph_path, verbose=self.verbose)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to classify and save graph for {pdb_code}: {e}"
                )
            finally:
                del structure_fragment_graph

        return fragment_graph_path

    def process(self, uniprotid_to_path: t.Dict[str, Path]) -> t.Dict[str, Path]:
        """
        Process a dictionary of UniProt IDs and PDB file paths to generate graphs.

        Parameters
        ----------
        uniprotid_to_path: t.Dict[str, t.Dict[str, Path]
            Dictionary where keys are UniProt IDs and values are paths to PDB files.

        Returns
        -------
        uniprotid_to_graph_path: t.Dict[str, Path]
            Dictionary where keys are UniProt IDs and values are paths to the saved graphs.
        """
        if self.error_file.exists():
            with open(self.error_file, "r") as error_file:
                self.error_list.update(
                    protein.strip() for protein in error_file.readlines()
                )

        uniprotid_to_graph_path = {}
        with tqdm(
            total=len(uniprotid_to_path),
            desc="Converting proteins to graphs:",
            unit="protein",
        ) as pbar:
            for uniprot_id, pdb_path in uniprotid_to_path.items():
                if uniprot_id in self.error_list:
                    pbar.update(1)
                    continue
                try:
                    graph_path = self.classify_and_save_graph(pdb_path)
                    uniprotid_to_graph_path[uniprot_id] = graph_path
                except Exception as e:
                    self.error_list.add(uniprot_id)
                    if self.verbose:
                        print(f"\nError occurred for protein {uniprot_id}: {e}")
                pbar.update(1)

        if self.error_list:
            with open(self.error_file, "w") as error_file:
                for protein in self.error_list:
                    error_file.write(f"{protein}\n")
            if self.verbose:
                print(
                    f"\nErrors occurred for the following proteins {self.error_list} and were written to error.txt"
                )

        return uniprotid_to_graph_path


class GraphDataset(Dataset):
    def __init__(
        self,
        root: str,
        uniprot_to_graph_paths: t.Dict[str, Path],
        one_hot_labels: torch.Tensor,
        uniprot_id_labels: t.List[str],
        node_attributes: t.List[str],
        edge_attributes: t.List[str],
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.uniprot_to_graph_paths = uniprot_to_graph_paths
        self.one_hot_labels = one_hot_labels
        self.uniprot_id_labels = uniprot_id_labels

        self.node_attributes = node_attributes
        self.edge_attributes = edge_attributes
        # Initialize data storage lists
        self.data_paths = []
        self.labels = []
        # Process paths and labels before calling super().__init__
        self.process()
        # Initialize the PyTorch Dataset
        super().__init__(root, transform, pre_transform, pre_filter)

    def process(self):
        """Process the paths to graphs and corresponding labels."""
        data_paths = []
        labels = []

        uniprot_id_to_index = {
            uniprot_id: i for i, uniprot_id in enumerate(self.uniprot_id_labels)
        }

        for uniprot_id, path in self.uniprot_to_graph_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Path {path} does not exist")

            # Ensure the uniprot_id exists in the uniprot_id_to_index dictionary
            if uniprot_id in uniprot_id_to_index:
                index = uniprot_id_to_index[uniprot_id]
                data_paths.append(path)
                labels.append(self.one_hot_labels[index].to_dense())
            else:
                print(f"Uniprot ID {uniprot_id} not found in labels.")

        self.data_paths = data_paths
        self.labels = torch.stack(labels) 

    @property
    def processed_file_names(self):
        return self.data_paths

    def __len__(self):
        """Return the total number of samples."""
        return len(self.data_paths)

    def __getitem__(self, idx):
        """Retrieve a single item from the dataset."""
        structure_fragment_graph = StructureFragmentGraph.load(
            self.data_paths[idx]
        ).to_pyg(
            node_attributes=self.node_attributes, edge_attributes=self.edge_attributes
        )

        # Set the label for the graph with an additional batch dimension
        structure_fragment_graph.y = self.labels[idx].unsqueeze(0)

        return structure_fragment_graph


def create_graphs(
    graph_creator: GraphCreator,
    uniprot_to_path_dict: t.Dict,
    data_dict: t.Dict[str, t.Dict],
) -> t.Tuple[t.Dict, t.Dict[str, t.Dict]]:
    """
    Create graphs for the given Uniprot to path dictionary and update the labels in data_dicts.

    Parameters
    ----------
    graph_creator: GraphCreator
        Graph creator object.
    uniprot_to_path_dict: t.Dict
        Dictionary containing the Uniprot ID to the path of the downloaded file.
    data_dict: t.Dict[str, t.Dict]
        Dictionary containing the data dictionaries for each split (e.g., {"train": train_dict, "validation": valid_dict, "test": test_dict}).

    Returns
    -------
    uniprot_to_graphs: t.Dict
        Dictionary containing the Uniprot ID to the generated graph for each data split ["train", "validation", "test"].
    data_dicts: t.Dict[str, t.Dict]
        The updated data dictionaries after removing mismatched labels.
    """
    uniprot_to_graphs = {}

    for data_split, path in uniprot_to_path_dict.items():
        # Create graphs
        graphs = graph_creator.process(path)
        uniprot_to_graphs[data_split] = graphs

        # Update labels dict to keep only the graphs that have corresponding labels
        labels_dict = data_dict[data_split]["labels"]
        updated_labels_dict = {
            key: labels_dict[key] for key in graphs.keys() if key in labels_dict
        }

        # Update data_dict with the filtered labels
        data_dict[data_split]["labels"] = updated_labels_dict

    return uniprot_to_graphs, data_dict
