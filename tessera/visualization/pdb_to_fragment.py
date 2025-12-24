import argparse
from pathlib import Path

from tessera.fragments.fragments_classifier import EnsembleFragmentClassifier
from tessera.fragments.fragments_graph import StructureFragmentGraph


def main(args):
    # Check Paths exist
    assert args.structure_path.exists(), f"{args.structure_path} does not exist"
    assert args.fragment_path.exists(), f"{args.fragment_path} does not exist"

    assert Path(args.fragment_path).exists(), "Fragment all_pdb_paths does not exist"
    assert Path(args.structure_path).exists(), "Structure all_pdb_paths does not exist"
    classifier = EnsembleFragmentClassifier(
        Path(args.fragment_path),
        difference_names=["logpr", "RamRmsd"],
        n_processes=1
    )
    import time

    time_start = time.time()
    structure_fragment = classifier.classify_to_fragment(
        Path(args.structure_path), use_all_fragments=False
    )
    time_end = time.time()
    print(f"Time taken: {time_end - time_start}")
    print(structure_fragment.classification_map)

    time_start = time.time()
    # Convert to graph_dataset:
    structure_fragment_graph = StructureFragmentGraph.from_structure_fragment(
        structure_fragment,
        edge_distance_threshold=10,
    )
    time_end = time.time()
    print(f"Time taken: {time_end - time_start}")
    print(structure_fragment_graph)
    raise ValueError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--structure_path", type=Path, help="Path to input file")
    parser.add_argument("--fragment_path", type=Path, help="Path to input file")
    params = parser.parse_args()
    main(params)
