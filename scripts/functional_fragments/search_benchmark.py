from abc import ABC, abstractmethod
import argparse
import typing as t
from pathlib import Path
import random
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import ampal
import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
import gmatch4py as gm

from tessera.difference_fn.shape_difference import RmsdBiopythonStrategy
from tessera.fragments.classification_config import selected_pdbs
from tessera.fragments.fragments_classifier import EnsembleFragmentClassifier
from tessera.fragments.fragments_graph import StructureFragmentGraph


class SearchBase(ABC):
    def __init__(
        self,
        database: t.List[Path],
        output_dir: Path,
        fragment_path: Path,
        workers: int = 1,
    ):
        self.database = database
        self.output_dir = output_dir
        self.workers = workers
        self.fragment_path = fragment_path
        self.preloaded_data = self.preload()

    @abstractmethod
    def preload(self) -> t.Dict:
        """Initialize the search operation. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def search(self, query: Path) -> t.Dict:
        """Perform the search operation. Must be implemented by subclasses."""
        pass


class RMSDSearch(SearchBase):
    def preload(self) -> t.Dict[str, t.Dict[str, ampal.AmpalContainer]]:
        preloaded_pdbs = {}
        for pdb_file in self.database:
            preloaded_pdbs[str(pdb_file)] = ampal.load_pdb(str(pdb_file)).make_pdb()
        return {"data": preloaded_pdbs}

    def search(self, query: Path) -> None:
        query_pdb = self.preloaded_data["data"][str(query)]
        for pdb_code, pdb_structure in self.preloaded_data["data"].items():
            rmsd = RmsdBiopythonStrategy.biopython_calculate_rmsd_fast(
                reference_pdb=query_pdb, fragment_pdb=pdb_structure
            )
        # In theory this could be expanded, but this is just benchmarking


class GEDSearch(SearchBase):
    def preload(self) -> t.Dict[str, t.Dict[str, ampal.AmpalContainer]]:
        preloaded_pdbs = {}
        classifier = EnsembleFragmentClassifier(
            self.fragment_path,
            difference_names=["LogPr", "RamRmsd"],
            n_processes=1,
            step_size=1,
        )
        # Preload PDBs
        for pdb_file in self.database:
            # Load the PDB file
            structure_fragment = classifier.classify_to_fragment(
                pdb_file, use_all_fragments=True
            )
            fragment_graph = StructureFragmentGraph.from_structure_fragment(
                structure_fragment,
                edge_distance_threshold=10,
            ).graph
            # Iterate through nodes and convert fragment_class to str
            for node in fragment_graph.nodes(data=True):
                node[1]["fragment_class"] = str(node[1]["fragment_class"])
            # Iterate through edges and convert peptide_bond to str
            for edge in fragment_graph.edges(data=True):
                edge[2]["peptide_bond"] = str(edge[2]["peptide_bond"])
            preloaded_pdbs[str(pdb_file)] = fragment_graph
        comparator = gm.GraphEditDistance(1, 1, 1, 1)
        # Set attributes used in comparison
        comparator.set_attr_graph_used(
            node_attr_key="fragment_class", edge_attr_key="peptide_bond"
        )
        return {
            "data": preloaded_pdbs,
            "comparator": comparator,
            "classifier": classifier,
        }

    def search(self, query: Path) -> None:
        fragment_graph = self.preloaded_data["data"][str(query)]
        result = self.preloaded_data["comparator"].compare(
            list(self.preloaded_data["data"].values()) + [fragment_graph], None
        )


class BONSearch(SearchBase):
    def preload(self) -> t.Dict[str, t.Dict[str, ampal.AmpalContainer]]:
        preloaded_pdbs = {}
        classifier = EnsembleFragmentClassifier(
            self.fragment_path,
            difference_names=["LogPr", "RamRmsd"],
            n_processes=1,
            step_size=1,
        )
        # Preload PDBs
        for pdb_file in self.database:
            # Load the PDB file
            structure_fragment = classifier.classify_to_fragment(
                pdb_file, use_all_fragments=True
            )
            fragment_graph = StructureFragmentGraph.from_structure_fragment(
                structure_fragment,
                edge_distance_threshold=10,
            ).graph
            # Iterate through nodes and convert fragment_class to str
            for node in fragment_graph.nodes(data=True):
                node[1]["fragment_class"] = str(node[1]["fragment_class"])
            # Iterate through edges and convert peptide_bond to str
            for edge in fragment_graph.edges(data=True):
                edge[2]["peptide_bond"] = str(edge[2]["peptide_bond"])
            preloaded_pdbs[str(pdb_file)] = fragment_graph
        comparator = gm.BagOfNodes()
        # Set attributes used in comparison
        comparator.set_attr_graph_used(
            node_attr_key="fragment_class", edge_attr_key="peptide_bond"
        )
        return {
            "data": preloaded_pdbs,
            "comparator": comparator,
            "classifier": classifier,
        }

    def search(self, query: Path) -> t.Dict:
        fragment_graph = self.preloaded_data["data"][str(query)]
        result = self.preloaded_data["comparator"].compare(
            list(self.preloaded_data["data"].values()) + [fragment_graph], None
        )


class SequenceSearch(SearchBase):
    def preload(self) -> t.Dict[str, ampal.Polypeptide]:
        preloaded_sequences = {}
        for pdb_file in self.database:
            preloaded_sequences[str(pdb_file)] = ampal.load_pdb(str(pdb_file))[
                0
            ].sequence
        blosum_matrix = substitution_matrices.load("BLOSUM62")
        # Initialize aligner
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.open_gap_score = -10
        aligner.extend_gap_score = -0.5

        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

        return {
            "data": preloaded_sequences,
            "BLOSUM": blosum_matrix,
            "aligner": aligner,
        }

    def search(self, query: Path) -> t.Dict:
        query_sequence = self.preloaded_data["data"][str(query)]
        for pdb_code, pdb_sequence in self.preloaded_data["data"].items():
            alignment = self.preloaded_data["aligner"].align(
                query_sequence, pdb_sequence
            )[0]
            aln1, aln2 = str(alignment[0]).replace("-", "*"), str(alignment[1]).replace(
                "-", "*"
            )

            # Compute score based on the BLOSUM matrix
            score = sum(
                self.preloaded_data["BLOSUM"].get(aa1 + aa2, 0)
                for aa1, aa2 in zip(aln1, aln2)
            )


class SearchBenchmark:
    def __init__(
        self,
        pdb_files: t.List[Path],
        output_dir: Path,
        fragment_path: Path,
        random_seed: int = 42,
        workers: int = 1,
    ):
        self.pdb_files = pdb_files[:100]
        self.workers = workers
        self.output_dir = output_dir
        self.fragment_path = fragment_path
        # Set random seed
        random.seed(random_seed)
        # Randomly select from the list of PDBs for searching
        self.sel_pdb_1 = [random.choice(self.pdb_files)]
        self.sel_pdb_10 = random.sample(self.pdb_files, 10)
        self.sel_pdb_100 = random.sample(self.pdb_files, 100)

    def search(self, search_method: SearchBase) -> t.Dict:
        # Convert the search method to a string
        search_method_name = search_method.__name__
        # Initialize Search Method
        init_time_start = time.time()
        search_method = search_method(
            database=self.pdb_files,
            workers=self.workers,
            output_dir=self.output_dir,
            fragment_path=self.fragment_path,
        )
        init_time_end = time.time()
        init_time = init_time_end - init_time_start
        # Initialize the dictionary to store the results
        results = {"init_time": init_time}

        for query_list in [self.sel_pdb_1, self.sel_pdb_10, self.sel_pdb_100]:
            curr_query_number = len(query_list)
            start_time = time.time()
            for query in query_list:
                search_method.search(query=query)
            end_time = time.time()
            elapsed_time = end_time - start_time
            results[curr_query_number] = elapsed_time

        return {search_method_name: results}


def parse_pdb_file_tree(input_pdb: Path) -> t.List[Path]:
    pdb_files = []
    # Walk through the directory tree
    for category in input_pdb.iterdir():
        if category.is_dir():
            for pdb_file in category.glob("*.pdb"):
                pdb_code = pdb_file.stem[:4]
                if pdb_code not in selected_pdbs:
                    continue
                pdb_files.append(pdb_file)

    return pdb_files


def calculate_avg_datapoints(pdb_files: t.List[Path], fragment_path: Path) -> None:
    classifier = EnsembleFragmentClassifier(
        fragment_path, difference_names=["LogPr", "RamRmsd"], n_processes=1, step_size=1
    )
    fragment_list_counts = []
    fragment_set_counts = []
    sequence_counts = []
    atom_counts = []
    for pdb_file in pdb_files:
        structure_fragment = classifier.classify_to_fragment(
            pdb_file, use_all_fragments=True
        )
        # Extract classes from the fragment
        fragments_list = []
        for fragment_detail in structure_fragment.classification_map:
            fragments_list.append(fragment_detail.fragment_class)
        fragment_set = set(fragments_list)
        # Append the counts to the list
        fragment_list_counts.append(len(fragments_list))
        fragment_set_counts.append(len(fragment_set))
        sequence_count = len(structure_fragment.classification)
        sequence_counts.append(sequence_count)
        atom_counts.append(sequence_count * 6)

    # Print the averages and standard deviations
    print(
        f"Avg fragment list count: {np.mean(fragment_list_counts):.2f} ± {np.std(fragment_list_counts):.2f}"
    )
    print(
        f"Avg fragment set count: {np.mean(fragment_set_counts):.2f} ± {np.std(fragment_set_counts):.2f}"
    )
    print(
        f"Avg sequence count: {np.mean(sequence_counts):.2f} ± {np.std(sequence_counts):.2f}"
    )
    print(f"Avg atom count: {np.mean(atom_counts):.2f} ± {np.std(atom_counts):.2f}")

    # Plot the histograms
    fig, axes = plt.subplots(2, 2, figsize=(5, 5), sharey=True)
    axes = axes.flatten()

    # Data and titles
    data_list = [
        fragment_list_counts,
        fragment_set_counts,
        sequence_counts,
        atom_counts,
    ]
    titles = ["Frag. Graph", "Frag. Set", "Seq. Residues", "Atoms"]
    colors = [
        "#2ca02c",
        "#ff7f0e",
        "#d62728",
        "#1f77b4",
    ]  # Different colors for each plot

    for ax, data, title, color in zip(axes, data_list, titles, colors):
        # Plot histogram
        sns.histplot(data, ax=ax, bins=30, kde=False, color=color)

        # Set titles and labels
        ax.set_title(title, fontsize=14)
        ax.set_xlim(left=0)
        ax.set_xlabel("Count", fontsize=14)
        ax.set_ylabel("Frequency (%)", fontsize=14)

        # Convert y-axis to percentage
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=len(data)))

    # Improve readability
    plt.tight_layout()
    # output to file

    out_file = fragment_path / "output.pdf"
    plt.savefig(out_file, dpi=300)
    plt.close()
    raise ValueError


def main(args):
    assert args.input_pdb.exists(), f"Input file {args.input_pdb} does not exist"
    if not args.output.exists():
        args.output.mkdir(parents=True)
    # List all .pdb files in the pdb_path
    pdb_files = parse_pdb_file_tree(args.input_pdb)
    # calculate_avg_datapoints(pdb_files, args.fragment_path)
    # Initialize the benchmark
    benchmark = SearchBenchmark(
        pdb_files=pdb_files,
        output_dir=args.output,
        workers=1,
        fragment_path=args.fragment_path,
    )
    # Perform the benchmark
    results = benchmark.search(RMSDSearch)
    print(results)
    # Do the same for Sequence
    results = benchmark.search(SequenceSearch)
    print(results)
    results = benchmark.search(GEDSearch)
    print(results)
    results = benchmark.search(BONSearch)
    print(results)
    raise ValueError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_pdb", type=Path, help="Path to input PDB files")
    parser.add_argument("--output", type=Path, help="Path to output directory")
    parser.add_argument("--fragment_path", type=Path, help="Path to fragment directory")
    params = parser.parse_args()
    main(params)
