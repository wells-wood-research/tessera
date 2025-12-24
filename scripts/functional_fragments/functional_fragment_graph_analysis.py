import argparse
import logging
import os
import typing as t
from gmatch4py import mcs

from itertools import combinations
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Dict, List, Tuple
from Bio.Align import substitution_matrices, PairwiseAligner
import ampal
import gmatch4py as gm
import numpy as np
import pandas as pd
import plotly.express as px
import umap
from adjustText import adjust_text
from matplotlib.patches import Patch
from networkx.algorithms.similarity import graph_edit_distance
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
import networkx as nx

def extract_bag_of_nodes(graph) -> t.Set[str]:
    """
    Converts a graph into a set of node attributes (like a bag of nodes).
    For simplicity, we use fragment_class attribute of each node.
    """
    node_attrs = set()
    for _, data in graph.nodes(data=True):
        node_attrs.add(data["fragment_class"])
    return node_attrs


def find_frequent_node_sets(
    grouped_graphs: t.Dict[str, t.List], output_path: Path, frequency_threshold: int = 2
):
    """
    Identifies frequently occurring node attributes (or sets) across all graphs in each category.
    Writes the results to the output directory.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    for category, graphs in grouped_graphs.items():
        # Extract bag of nodes from each graph
        bags = [extract_bag_of_nodes(g) for g in graphs]

        # Count frequency of each node attribute across all graphs in this category
        freq_count: t.Dict[str, int] = {}
        for bag in bags:
            for node_attr in bag:
                freq_count[node_attr] = freq_count.get(node_attr, 0) + 1

        # Filter by frequency threshold
        frequent = {
            item: count
            for item, count in freq_count.items()
            if count >= frequency_threshold
        }

        # Write results
        cat_output = output_path / f"{category}_frequent_node_sets.txt"
        with cat_output.open("w") as f:
            for item, count in frequent.items():
                f.write(f"{item}: {count}\n")


def group_graphs_by_category(
    pdb_data: t.Dict[str, t.Dict[str, Path]], nx_graphs: t.List
) -> t.Dict[str, t.List]:
    """
    Groups graphs by their category.
    """
    category_groups: t.Dict[str, t.List] = {}
    pdb_keys = get_sorted_pdb_keys(pdb_data)
    for pdb, g in zip(pdb_keys, nx_graphs):
        cat = pdb_data[pdb]["category"]
        if cat not in category_groups:
            category_groups[cat] = []
        category_groups[cat].append(g)
    return category_groups


import matplotlib.pyplot as plt





def get_sorted_pdb_keys(pdb_data: Dict[str, Dict[str, Path]]) -> List[str]:
    """
    Returns sorted PDB keys based on category and PDB name.
    Handles missing or inconsistent categories gracefully.
    """
    return sorted(
        pdb_data.keys(),
        key=lambda x: (pdb_data[x].get("category", "unknown").lower(), x),
    )


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


def get_nx_graph(pdb_data: Dict[str, Dict[str, Path]]) -> List[nx.Graph]:
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
    return nx_graphs


def main(args):
    # Set Seed
    np.random.seed(42)
    assert (
        args.input_pdb.exists()
    ), f"Input PDB directory {args.input_pdb} does not exist"
    assert args.input_graph.exists(), f"Fragment path {args.input_graph} does not exist"
    # Parse the PDB file tree
    pdb_data = parse_pdb_file_tree(args.input_pdb, args.input_graph)
    # Create distance matrices
    nx_graphs = get_nx_graph(pdb_data)
    grouped_graphs = group_graphs_by_category(pdb_data, nx_graphs)
    find_frequent_node_sets(grouped_graphs, args.output_path, frequency_threshold=2)


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
