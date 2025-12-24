import ampal
import pandas as pd

from pathlib import Path

from tessera.difference_fn.difference_processing import (
    get_residue_ids,
    select_first_ampal_assembly,
)
from tessera.fragments.fragments_classifier import StructureToFragmentClassifier

if __name__ == "__main__":
    fragments_df = pd.read_csv("data/fragments.csv")
    fragment_path = Path("data/fragments/")
    structure_path = Path("data_paths/pdbs/")
    output_path = Path("data_paths/metrics/")
    output_path.mkdir(exist_ok=True)
    # Check all paths exist
    assert fragment_path.exists()
    assert structure_path.exists()

    differences_angles = ["logpr"]
    # differences_angles = []
    differences_sequences = []
    # differences_sequences = []
    differences_types = ["angle"] * len(differences_angles) + ["sequence"] * len(
        differences_sequences
    )
    differences = differences_angles + differences_sequences

    probabilistic_classification = [False] * len(differences)
    # For each of these differences create classifier and add to dictionary
    classifiers = {}
    for difference, p_c, diff_type in zip(
        differences, probabilistic_classification, differences_types
    ):
        classifier = StructureToFragmentClassifier(
            Path(fragment_path),
            difference_type=diff_type,
            difference_name=difference,
            n_processes=10,
            step_size=1,

        )
        classifiers[difference] = classifier

    for row in fragments_df.iterrows():
        pdb = row[1]["Identifier"]
        start_position = str(row[1]["Start Residue"])
        end_position = str(row[1]["End Residue"])
        fragment_n = row[1]["Fragment Number"]
        if fragment_n.startswith("B"):
            continue
        print(f"Processing {pdb}")
        try:
            pdb_path = structure_path / f"{pdb}.pdb1"
            assert pdb_path.exists(), f"{pdb_path} does not exist"
            structure = ampal.load_pdb(str(pdb_path))
            selected_structure = select_first_ampal_assembly(structure)
            if len(selected_structure.sequence) < 70:
                print(f"Skipping {pdb} as it is too short")
                continue
            print(selected_structure.sequence)
            assert Path(
                fragment_path
            ).exists(), f"Fragment {fragment_path} does not exist"
            assert Path(
                structure_path
            ).exists(), f"Structure {structure_path} does not exist"

            residue_ids = get_residue_ids(selected_structure)
            # Check where the start and end positions are in the residue ids
            start_position = residue_ids.index(start_position)
            end_position = residue_ids.index(end_position)

            for difference, classifier in classifiers.items():
                curr_fragmented_structure = classifier.classify_to_fragment(
                    Path(pdb_path), use_all_fragments=True,
                )
                curr_fragmented_structure.save_probability_to_csv(
                    csv_save_path=output_path,
                    prefix=f"{difference}_fragment_{fragment_n}_start_{start_position}_end_{end_position}",
                )
        except Exception as e:
            print(f"Error processing {pdb}: {e}")
            continue
