import argparse
import random
import json
import typing as t
from collections import Counter
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool

from src.fragments.classification_config import fragment_lengths, go_to_prosite
from src.fragments.fragments_graph import StructureFragmentGraphIO
from src.function_prediction.uniprot_processing import (
    PrositeToUniprot,
    UniprotDownloader,
)
from src.training.data_processing.dataset import GraphCreator
from src.visualization.fold_coverage import load_graph_creator
from tqdm import tqdm


def get_go_to_uniprot(
    go_to_prosite_uniprot: t.Dict[str, t.Dict[str, t.List[str]]],
    go_add: t.List[str],
    go_subtract: t.List[str],
    verbose: bool = False,
) -> t.Dict[str, t.Set[str]]:
    go_to_uniprot: t.Dict[str, t.Set[str]] = {}

    # Step 1: Collect UniProt IDs for each GO term
    for go_term, prosite_dict in go_to_prosite_uniprot.items():
        current_uniprot_ids = []
        for _, uniprot_ids in prosite_dict.items():
            current_uniprot_ids.extend(uniprot_ids)

        # Store the unique set of UniProt IDs for this GO term
        go_to_uniprot[go_term] = set(current_uniprot_ids)

    # Step 2: Create a new GO term with intersection of all terms in go_add
    if len(go_add) > 1:
        intersection_set = set(
            go_to_uniprot[go_add[0]]
        )  # Start with the first GO term's UniProt IDs
        for go_term in go_add[1:]:
            if go_term in go_to_uniprot:
                intersection_set &= go_to_uniprot[
                    go_term
                ]  # Perform intersection with next GO term's UniProt IDs

        # Create a new GO term with the intersection result
        new_go_term = "+".join(go_add)
        go_to_uniprot[new_go_term] = intersection_set

        if verbose:
            print(
                f"Created new GO term {new_go_term} with {len(go_to_uniprot[new_go_term])} shared UniProt IDs:"
            )

        assert (
            len(go_to_uniprot[new_go_term]) > 0
        ), f"GO term {new_go_term} has no UniProt IDs"

    # Step 3: Remove UniProt IDs for the terms in go_remove from all GO terms
    if go_subtract:
        removal_set = set()
        for go_term in go_subtract:
            if go_term in go_to_uniprot:
                removal_set.update(
                    go_to_uniprot[go_term]
                )  # Gather all UniProt IDs to be removed

        for go_term, uniprot_ids in go_to_uniprot.items():
            go_to_uniprot[go_term] = (
                uniprot_ids - removal_set
            )  # Remove the gathered UniProt IDs from all GO terms

    # Step 4: Remove GO terms we are not interested in
    all_go_codes = list(go_to_uniprot.keys())
    for go_term in all_go_codes:
        if go_term in go_add or "+" in go_term:
            continue
        else:
            go_to_uniprot.pop(go_term, None)

    # Step 5: Check that the go terms are not empty
    empty_go_terms = [
        go_term for go_term, uniprot_ids in go_to_uniprot.items() if not uniprot_ids
    ]
    assert (
        not empty_go_terms
    ), f"The following GO term(s) have no UniProt IDs: {', '.join(empty_go_terms)}"

    if verbose:
        print_overlap(go_to_uniprot)

    return go_to_uniprot


def print_overlap(go_to_uniprot: t.Dict[str, t.Set[str]]) -> None:
    overlap_results = {}

    go_terms = list(go_to_uniprot.keys())

    # Calculate overlap for pairs
    for go_term1, go_term2 in itertools.combinations(go_terms, 2):
        # Sort the pair to ensure uniqueness (AB and BA are considered the same)
        sorted_pair = tuple(sorted([go_term1, go_term2]))
        
        if sorted_pair not in overlap_results:
            overlap = go_to_uniprot[go_term1] & go_to_uniprot[go_term2]
            overlap_count = len(overlap)
            overlap_percent = (
                overlap_count
                / min(len(go_to_uniprot[go_term1]), len(go_to_uniprot[go_term2]))
            ) * 100
            overlap_results[sorted_pair] = {
                "count": overlap_count,
                "percent": overlap_percent,
                "uniprot_ids": list(overlap)
            }

    # Calculate overlap for triplets
    for go_term1, go_term2, go_term3 in itertools.combinations(go_terms, 3):
        # Sort the triplet to ensure uniqueness (ABC and BCA are considered the same)
        sorted_triplet = tuple(sorted([go_term1, go_term2, go_term3]))
        
        if sorted_triplet not in overlap_results:
            overlap = go_to_uniprot[go_term1] & go_to_uniprot[go_term2] & go_to_uniprot[go_term3]
            overlap_count = len(overlap)
            overlap_percent = (
                overlap_count
                / min(len(go_to_uniprot[go_term1]), len(go_to_uniprot[go_term2]), len(go_to_uniprot[go_term3]))
            ) * 100
            overlap_results[sorted_triplet] = {
                "count": overlap_count,
                "percent": overlap_percent,
                "uniprot_ids": list(overlap)
            }

    # Display the overlap results with descriptions
    for pair, results in overlap_results.items():
        go_terms = " and ".join(pair)
        print(
            f"Overlap between {go_terms}: {results['count']} UniProt IDs ({results['percent']:.2f}%)"
        )
        print(f"UniProt IDs: {', '.join(results['uniprot_ids'])}")

def plot_fragment_info(
    go_fragment_count: t.Dict[str, t.Dict[str, np.ndarray]],
    label_to_plot: str,
    ignore_unknown: bool = False,
    output_path: Path = None,
) -> None:
    x = np.arange(41) if not ignore_unknown else np.arange(1, 41)
    width = 0.35
    # ylim = 1 if ignore_unknown else 0.35
    ylim = 0.35

    n_go_terms = len(go_fragment_count)
    fig, axes = plt.subplots(nrows=n_go_terms, ncols=1, figsize=(14, 4 * n_go_terms))
    title = "Coverage (%)" if label_to_plot == "freq" else "Occurrence (%)"

    if n_go_terms == 1:
        axes = [axes]

    for idx, (go_term, data) in enumerate(go_fragment_count.items()):
        ax = axes[idx]

        active_site_mean = (
            data[f"active_site_{label_to_plot}_mean"][1:]
            if ignore_unknown
            else data[f"active_site_{label_to_plot}_mean"]
        )
        active_site_std = (
            data[f"active_site_{label_to_plot}_std"][1:]
            if ignore_unknown
            else data[f"active_site_{label_to_plot}_std"]
        )
        structure_mean = (
            data[f"structure_{label_to_plot}_mean"][1:]
            if ignore_unknown
            else data[f"structure_{label_to_plot}_mean"]
        )
        structure_std = (
            data[f"structure_{label_to_plot}_std"][1:]
            if ignore_unknown
            else data[f"structure_{label_to_plot}_std"]
        )

        ax.bar(
            x - width / 2,
            active_site_mean,
            width,
            yerr=active_site_std,
            label="Active Site",
            color="blue",
            capsize=5,
        )
        ax.bar(
            x + width / 2,
            structure_mean,
            width,
            yerr=structure_std,
            label="Structure",
            color="orange",
            capsize=5,
        )
        ax.set_ylim(0, ylim)
        ax.set_title(
            f"{go_term} - {go_to_prosite[go_term]['description'] if go_term in go_to_prosite else ''}", fontsize=18
        )
        ax.set_ylabel(title, fontsize=18)

        xticklabels = (
            ["UNK"] + [str(i) for i in range(1, 41)]
            if not ignore_unknown
            else [str(i) for i in range(1, 41)]
        )
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=12, rotation=45, ha="right")

        if idx == 0:
            ax.legend(fontsize=18)

    fig.text(0.5, 0.04, "Fragment", ha="center", fontsize=18)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path)
    plt.close()


def plot_hierarchical_clustering(
    go_fragment_count: t.Dict[str, t.Dict[str, np.ndarray]],
    label_to_plot: str,
    output_file: Path,
    ignore_unknown: bool = False,
) -> None:
    structure_means = []
    active_site_means = []
    go_terms = []

    for go_term, data in go_fragment_count.items():
        if ignore_unknown:
            structure_means.append(data[f"structure_{label_to_plot}_mean"][1:])
            active_site_means.append(data[f"active_site_{label_to_plot}_mean"][1:])
        else:
            structure_means.append(data[f"structure_{label_to_plot}_mean"])
            active_site_means.append(data[f"active_site_{label_to_plot}_mean"])

        go_terms.append(go_to_prosite[go_term]["description"] if go_term in go_to_prosite else go_term)

    # Convert lists to numpy arrays
    structure_means = np.array(structure_means)
    active_site_means = np.array(active_site_means)

    # Clustering on structure_mean
    linked_structure = linkage(pdist(structure_means), "ward")
    plt.figure(figsize=(10, 7))
    dendrogram(linked_structure, labels=go_terms)
    plt.title(
        f"Hierarchical Clustering of Structure Mean {label_to_plot.capitalize()} Frequencies"
    )
    plt.xlabel("GO Terms")
    plt.ylabel("Distance")
    plt.savefig(output_file)
    plt.close()

    # Clustering on active_site_mean
    output_file_active_site = output_file.with_name(
        output_file.stem.replace("struct", "active_site") + output_file.suffix
    )
    linked_active_site = linkage(pdist(active_site_means), "ward")
    plt.figure(figsize=(10, 7))
    dendrogram(linked_active_site, labels=go_terms)
    plt.title(
        f"Hierarchical Clustering of Active Site Mean {label_to_plot.capitalize()} Frequencies"
    )
    plt.xlabel("GO Terms")
    plt.ylabel("Distance")
    plt.savefig(output_file_active_site)
    plt.close()


def load_failed_uniprot_ids(output_path: Path) -> t.Tuple[t.Dict[str, t.Set[str]], Path]:
    failed_uniprot_ids_file = output_path / "failed_uniprot_ids.json"
    failed_uniprot_ids = dict()
    if failed_uniprot_ids_file.exists():
        failed_uniprot_ids = json.load(failed_uniprot_ids_file.open())
    return failed_uniprot_ids, failed_uniprot_ids_file


class UniprotFunctionAnalyser:
    def __init__(
        self,
        uniprot_to_prosite: t.Dict[str, t.Dict[str, t.Any]],
        go_to_prosite_uniprot: t.Dict[str, t.Dict[str, t.List[str]]],
        output_path: Path,
    ):
        self.uniprot_to_prosite = uniprot_to_prosite
        self.go_to_prosite_uniprot = go_to_prosite_uniprot
        self.output_path = output_path
        assert self.output_path.exists(), f"Path {self.output_path} does not exist"
        self.go_fragment_count = {}

    def analyse_fragments(self, uniprot_ids_list: t.List[str]) -> None:
        for uniprot_id in uniprot_ids_list:
            self.analyse_fragments_uniprot_id(uniprot_id)

        # Iterate through each GO term in the dictionary
        for go_term in self.go_fragment_count.keys():
            self._count_coverage(go_term)
            # Calculate mean and standard deviation of coverage after processing all proteins
            self._count_subgraphs(go_term)
            self.save_fragment_subgraph(go_term)

    def analyse_fragments_uniprot_id(self, uniprot_id: str) -> None:
        # Retrieve the GO term, Prosite information, and fragment path for the current UniProt ID
        go_term, prosite_info, fragment_path = (
            self.uniprot_to_prosite[uniprot_id]["go_term"],
            self.uniprot_to_prosite[uniprot_id]["prosite"],
            self.uniprot_to_prosite[uniprot_id]["fragment_path"],
        )
        # Load the fragment graph
        curr_fragment_graph = StructureFragmentGraphIO.load(fragment_path)
        fragment_classification = curr_fragment_graph.structure_fragment.classification
        # Create a mask to identify the active site fragments
        active_site_mask = np.zeros(fragment_classification.shape, dtype=bool)

        curr_subgraphs = []
        # Filter the prosite to only the function we are interested in
        for curr_prosite in prosite_info:
            pru_id = curr_prosite["prosite_rule_id"]
            if pru_id in self.go_to_prosite_uniprot[go_term].keys():
                start, end = (
                    curr_prosite["location_start"],
                    curr_prosite["location_end"],
                )
                # In the case of 1 residue active sites, we expand the range to 5 residues on each side
                if start == end:
                    start = max(0, start - 5)
                    end += 5
                active_site_mask[start + 1 : end + 1] = True
                # Extract the subgraph for the active site
                # Find indices where the difference is not zero
                change_indices = np.where(
                    np.diff(fragment_classification[start + 1 : end + 1]) != 0
                )[0]
                # Extract the values at the change points, include the first element of the sliced array
                flat_subgraph = np.concatenate(
                    (
                        [
                            fragment_classification[start + 1]
                        ],  # Use the first element of the sliced array
                        fragment_classification[
                            start + 1 + change_indices + 1
                        ],  # Adjust indices correctly
                    )
                )
                # Check if the array is only composed of [0]
                if not np.all(flat_subgraph == 0):
                    subgraph_strings = "-".join(map(str, flat_subgraph))
                    curr_subgraphs.append(subgraph_strings)

        # Count the coverage of each fragment in the active site and outside the active site
        fragments_with_pru = fragment_classification[active_site_mask]
        fragments_outside_pru = fragment_classification[~active_site_mask]
        fragments_active, counts_active = np.unique(
            fragments_with_pru, return_counts=True
        )
        fragments_structure, counts_structure = np.unique(
            fragments_outside_pru, return_counts=True
        )
        # Convert count to frequency
        freq_active = np.zeros(41)
        freq_structure = np.zeros(41)
        freq_active[fragments_active] = counts_active / len(fragments_with_pru)
        freq_structure[fragments_structure] = counts_structure / len(
            fragments_outside_pru
        )
        # Count the number of times each fragment appears in the active site and outside the active site
        active_fragments_counts = self.count_fragments_in_area(
            fragments_active, counts_active
        )
        structure_fragments_counts = self.count_fragments_in_area(
            fragments_structure, counts_structure
        )
        # Store the frequency arrays in the dictionary
        if go_term not in self.go_fragment_count:
            self.go_fragment_count[go_term] = {
                "active_site_freq": [],
                "structure_freq": [],
                "active_site_counts": [],
                "structure_counts": [],
                "subgraphs": [],
            }

        self.go_fragment_count[go_term]["active_site_freq"].append(freq_active)
        self.go_fragment_count[go_term]["structure_freq"].append(freq_structure)
        self.go_fragment_count[go_term]["active_site_counts"].append(
            active_fragments_counts
        )
        self.go_fragment_count[go_term]["structure_counts"].append(
            structure_fragments_counts
        )
        if curr_subgraphs:
            self.go_fragment_count[go_term]["subgraphs"].extend(curr_subgraphs)

    def _count_subgraphs(self, go_term: str) -> None:
        # Count the frequency of each unique subgraph string
        data = self.go_fragment_count[go_term]
        subgraph_counts = Counter(data["subgraphs"])
        # Save the frequencies to dict
        self.go_fragment_count[go_term]["subgraphs"] = subgraph_counts


    def _count_coverage(self, go_term: str):
        curr_dict_keys = list(self.go_fragment_count[go_term].keys())
        # Remove the subgraphs key
        curr_dict_keys.remove("subgraphs")

        for protein_area in curr_dict_keys:
            # Convert to numpy array
            self.go_fragment_count[go_term][protein_area] = np.array(
                self.go_fragment_count[go_term][protein_area]
            )
            # Calculate mean and standard deviation
            self.go_fragment_count[go_term][f"{protein_area}_mean"] = np.mean(
                self.go_fragment_count[go_term][protein_area], axis=0
            )
            self.go_fragment_count[go_term][f"{protein_area}_std"] = np.std(
                self.go_fragment_count[go_term][protein_area], axis=0
            )

    def save_fragment_subgraph(self, go_term: str) -> None:
        subgraph_counts = self.go_fragment_count[go_term]["subgraphs"]
        output_file = self.output_path / f"{go_term}_subgraphs.txt"
        # Write the frequencies to the file
        with output_file.open("w") as file:
            # Sort by count in descending order
            for subgraph, count in subgraph_counts.most_common():
                file.write(f"{subgraph}: {count}\n")

    @staticmethod
    def count_fragments_in_area(
        fragments: t.List[int], counts: t.List[int]
    ) -> np.ndarray:
        fragment_counts = np.zeros(41)
        for f, c in zip(fragments, counts):
            if f == 0:
                # Skip unknown fragments
                continue
            n_fragment = c / fragment_lengths[f]
            fragment_counts[f] += n_fragment

        # Divide by the total number of fragments
        sum_fragments = np.sum(fragment_counts)
        # IF sum is 0, return the counts to avoid returning nans
        if sum_fragments == 0:
            return fragment_counts
        else:
            return fragment_counts / sum_fragments


class PrositeFetcher:
    def __init__(
        self,
        go_to_uniprot: t.Dict[str, t.Set[str]],
        uniprot_downloader: UniprotDownloader,
        graph_creator: GraphCreator,
        n_proteins: int,
        output_path: Path,
        verbose: bool = False,
    ):
        self.go_to_uniprot = go_to_uniprot
        self.graph_creator = graph_creator
        self.uniprot_downloader = uniprot_downloader
        self.n_proteins = n_proteins
        self.output_path = output_path
        # Load failed_uniprot_ids from previous run
        self.failed_uniprot_ids, self.failed_uniprot_ids_file = load_failed_uniprot_ids(
            self.output_path
        )
        self.failed_tolerance = int(n_proteins * 0.25)
        self.verbose = verbose

    def fetch(self) -> t.Dict[str, t.Dict[str, t.Any]]:
        uniprot_to_prosite = {}

        for go_term, uniprot_ids in self.go_to_uniprot.items():
            # Ensure there's an entry for the GO term in failed_uniprot_ids and convert lists to sets
            if go_term not in self.failed_uniprot_ids:
                self.failed_uniprot_ids[go_term] = set()
                failed_count = 0
            else:
                self.failed_uniprot_ids[go_term] = set(self.failed_uniprot_ids[go_term])
                failed_count = len(self.failed_uniprot_ids[go_term])

            # Convert uniprot_ids to a set and remove failed ones
            uniprot_ids = set(uniprot_ids) - self.failed_uniprot_ids[go_term]
            # Sort to ensure reproducibility
            sorted_uniprot_ids = sorted(uniprot_ids)
            with tqdm(total=self.n_proteins, desc=f"Processing {go_term}") as pbar:
                count = 0
                while count < self.n_proteins and sorted_uniprot_ids:
                    # Randomly choose a UniProt ID
                    uniprot_id = random.choice(sorted_uniprot_ids)

                    # Check if uniprot_id is available
                    if not self._is_uniprot_available(uniprot_id, go_term):
                        self.failed_uniprot_ids[go_term].add(uniprot_id)
                        sorted_uniprot_ids.remove(uniprot_id)
                        continue
                    
                    # Attempt to download the PDB file for the UniProt ID
                    self.uniprot_downloader.process_protein(uniprot_id)
                    
                    pdb_file = self._is_uniprot_pdb_available(uniprot_id)
                    if not pdb_file:
                        self.failed_uniprot_ids[go_term].add(uniprot_id)
                        sorted_uniprot_ids.remove(uniprot_id)
                        continue
                    
                    active_site_info = self._is_active_site_info_available(uniprot_id)
                    if not active_site_info:
                        self.failed_uniprot_ids[go_term].add(uniprot_id)
                        sorted_uniprot_ids.remove(uniprot_id)
                        continue
                    
                    fragment_path = self._is_fragment_available(pdb_file)
                    if not fragment_path:
                        self.failed_uniprot_ids[go_term].add(uniprot_id)
                        sorted_uniprot_ids.remove(uniprot_id)
                        continue

                    # Add the information to the dictionary
                    uniprot_to_prosite[uniprot_id] = {
                        "go_term": go_term,
                        "prosite": active_site_info,
                        "fragment_path": fragment_path,
                    }

                    count += 1
                    pbar.update(1)
                    sorted_uniprot_ids.remove(uniprot_id)

                    if len(self.failed_uniprot_ids[go_term]) - failed_count >= self.failed_tolerance:
                        if self.verbose:
                            print(
                                f"More than {self.failed_tolerance} UniProt IDs failed for {go_term}"
                            )
                        break

                if not sorted_uniprot_ids:
                    print(f"Ran out of UniProt IDs for {go_term}")

        self.save_failed_uniprot_ids()
        return uniprot_to_prosite

    def _is_uniprot_available(self, uniprot_id: str, go_term: str):
        if uniprot_id in self.failed_uniprot_ids[go_term]:
            if self.verbose:
                print(f"Skipping Failed {uniprot_id}")
            return False
        return True

    def _is_uniprot_pdb_available(self, uniprot_id: str) -> t.Union[bool, Path]:
        pdb_file = self.uniprot_downloader.output_dir / f"{uniprot_id}.pdb"
        if pdb_file.exists():
            return pdb_file
        else:
            if self.verbose:
                print(f"No PDB file found for {uniprot_id}")
            return False

    def _is_active_site_info_available(
        self, uniprot_id: str
    ) -> t.Union[bool, t.List[t.Dict[str, t.Any]]]:
        active_site_info = self.uniprot_downloader.get_active_site_from_uniprot(
            uniprot_id
        )
        if not active_site_info:
            if self.verbose:
                print(f"No active site information found for {uniprot_id}")
            return False
        else:
            return active_site_info

    def _is_fragment_available(self, pdb_file: Path) -> t.Union[bool, Path]:
        # change extensiong to .fg
        fragment_file = pdb_file.with_suffix(".fg")
        if not fragment_file.exists():
            try:
                fragment_path = self.graph_creator.classify_and_save_graph(pdb_file)
                return fragment_path
            except Exception as e:
                if self.verbose:
                    print(f"Failed to create fragment for {pdb_file}")
                    print(e)
                return False
        return fragment_file

    def save_failed_uniprot_ids(self):
        # Convert sets to lists for JSON serialization
        serializable_failed_uniprot_ids = {k: list(v) for k, v in self.failed_uniprot_ids.items()}
        with open(self.failed_uniprot_ids_file, "w") as f:
            json.dump(serializable_failed_uniprot_ids, f)
        if self.verbose:
            print(f"Failed UniProt IDs: {serializable_failed_uniprot_ids}")


def main(args):
    # Set the random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Check if the output path exists
    args.output_path = Path(args.output_path)
    args.fragment_path = Path(args.fragment_path)
    assert args.output_path.exists(), f"Path {args.output_path} does not exist"
    assert args.fragment_path.exists(), f"Path {args.fragment_path} does not exist"
    # Check if the GO codes are valid
    go_codes_to_check = args.add_go_codes
    if args.subtract_go_codes:
        # Assert that there is no intersection between dd_go_codes and subtract_go_codes
        assert (
            len(set(args.add_go_codes) & set(args.subtract_go_codes)) == 0
        ), "GO codes cannot be in both add and subtract"
        go_codes_to_check += args.subtract_go_codes
    # Check if the GO codes are in the Prosite dictionary using sets
    for go_code in go_codes_to_check:
        assert go_code in go_to_prosite, f"GO code {go_code} not found in Prosite"
    # Retrieve UniProt ids for each Prosite entry for the selected GO term
    prosite_to_uniprot = PrositeToUniprot(args.output_path)
    # Create a dictionary with GO terms as keys and a dictionary of Prosite entries and their corresponding UniProt as value
    go_to_prosite_uniprot = prosite_to_uniprot.process(go_to_prosite)
    # Create a dictionary with GO terms as keys and a set of UniProt IDs as value
    go_to_uniprot = get_go_to_uniprot(
        go_to_prosite_uniprot,
        go_add=args.add_go_codes,
        go_subtract=args.subtract_go_codes,
        verbose=args.verbose,
    )
    # Adjust prosite info for combined GO terms
    for go_code in go_to_uniprot.keys():
        if "+" in go_code:
            # Combine the prosite for each GO term
            list_go_codes = go_code.split("+")
            for g in list_go_codes:
                if go_code not in go_to_prosite_uniprot:
                    go_to_prosite_uniprot[go_code] = {}
                go_to_prosite_uniprot[go_code].update(go_to_prosite_uniprot[g])
    # Initialize UniprotDownloader
    uniprot_downloader = UniprotDownloader(output_dir=args.output_path)
    graph_creator = load_graph_creator(
        fragment_path=args.fragment_path,
        workers=args.workers,
        pdb_path=uniprot_downloader.output_dir,
        verbose=args.verbose,
    )
    # Download the PDB files for the UniProt IDs and extract the active site information
    prosite_fetcher = PrositeFetcher(
        go_to_uniprot=go_to_uniprot,
        n_proteins=args.n_proteins,
        uniprot_downloader=uniprot_downloader,
        graph_creator=graph_creator,
        output_path=args.output_path,
        verbose=args.verbose,
    )
    uniprot_to_prosite = prosite_fetcher.fetch()

    # Create output path for fragment coverage
    fragment_counts_output_path = args.output_path / f"fragment_coverage_n_proteins_{args.n_proteins}"
    fragment_counts_output_path.mkdir(exist_ok=True)
    # Initialize the dictionary to store the fragment counts for each GO term
    uniprot_function_analyser = UniprotFunctionAnalyser(
        uniprot_to_prosite, go_to_prosite_uniprot, output_path=fragment_counts_output_path
    )
    # For each uniprot_id, calculate the frequency of each fragment in the active site and outside the active site
    uniprot_function_analyser.analyse_fragments(list(uniprot_to_prosite.keys()))
    # Plot the fragment coverage for each GO term
    go_fragment_count = uniprot_function_analyser.go_fragment_count
    # For frequency
    all_go_terms = "_".join(go_fragment_count.keys())
    output_file = f"fragment_coverage_{args.n_proteins}_proteins_go_{all_go_terms}_freq.pdf"
    plot_fragment_info(
        go_fragment_count, "freq", args.ignore_unknown, args.output_path / output_file
    )
    output_file = (
        args.output_path
        / f"clustering_{args.n_proteins}_proteins_go_{all_go_terms}_freq.pdf"
    )
    if len(args.add_go_codes) > 1:
        plot_hierarchical_clustering(go_fragment_count, "freq", output_file)
    # For Counts
    output_file = f"fragment_coverage_{args.n_proteins}_proteins_go_{all_go_terms}_counts.pdf"
    plot_fragment_info(
        go_fragment_count, "counts", args.ignore_unknown, args.output_path / output_file
    )
    output_file = (
        args.output_path
        / f"clustering_{args.n_proteins}_proteins_go_{all_go_terms}_counts.pdf"
    )
    if len(args.add_go_codes) > 1:
        plot_hierarchical_clustering(go_fragment_count, "counts", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--output_path", type=str, help="Path to output file")
    parser.add_argument(
        "--add_go_codes", type=str, nargs="+", help="List of GO codes to intersect"
    )
    parser.add_argument(
        "--subtract_go_codes", type=str, nargs="+", help="List of GO codes to subtract"
    )
    parser.add_argument(
        "--n_proteins", type=int, help="Number of proteins to fetch", default=2
    )
    parser.add_argument("--verbose", action="store_true", help="Print details")
    parser.add_argument("--workers", type=int, help="Number of workers", default=10)
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility", default=42
    )
    parser.add_argument(
        "--fragment_path", type=str, required=True, help="Path to fragment classifier"
    )
    parser.add_argument(
        "--ignore_unknown",
        action="store_true",
        help="Whether to ignore unknown fragments in the analysis",
    )
    params = parser.parse_args()
    main(params)
