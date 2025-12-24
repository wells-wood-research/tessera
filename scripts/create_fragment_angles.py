import numpy as np
import argparse
from pathlib import Path
import ampal
import pandas as pd


def preprocess_sequence(sequence):
    # Identify positions of lowercase letters in the sequence
    lowercase_positions = [i for i, c in enumerate(sequence) if c.islower()]
    return lowercase_positions


def calculate_angles_for_structure(residues):
    angles = []
    for i, residue in enumerate(residues[1:-1], start=1):  # Skip first and last
        if (
            i - 1 < len(residues) - 2
        ):  # Ensure we don't go out of bounds after removing residues
            r_prev = residues[i - 1]
            r_curr = residue
            r_next = residues[i + 1]

            omega = ampal.geometry.dihedral(
                r_prev["CA"], r_prev["C"], r_curr["N"], r_curr["CA"]
            )
            phi = ampal.geometry.dihedral(
                r_prev["C"], r_curr["N"], r_curr["CA"], r_curr["C"]
            )
            psi = ampal.geometry.dihedral(
                r_curr["N"], r_curr["CA"], r_curr["C"], r_next["N"]
            )
            ca = ampal.geometry.angle_between_vectors(r_prev["CA"], r_curr["CA"])

            angles.append([omega, phi, psi, ca])
    return angles


def remove_unused_residues(residues, lowercase_positions):
    # Remove residues corresponding to lowercase positions
    if lowercase_positions:
        residues = [
            residue
            for i, residue in enumerate(residues)
            if i not in lowercase_positions
        ]
    return residues


def load_structures_and_calculate_angles(folder_path, fragments_df):
    all_angles = []

    for pdb_file in folder_path.glob("*.pdb1"):
        pdb_id = (
            pdb_file.stem.upper()
        )  # Assuming the PDB ID is the file stem and is uppercase
        fragment_info = fragments_df[fragments_df["Identifier"] == pdb_id]
        if not fragment_info.empty:
            # If there are multiple sequences, take the first one
            sequence = fragment_info.iloc[0]["Amino Acid Sequence"]
            lowercase_positions = preprocess_sequence(sequence)

            structure = ampal.load_pdb(str(pdb_file))[0]
            residues = remove_unused_residues(
                structure.get_monomers(ligands=False), lowercase_positions
            )
            angles = calculate_angles_for_structure(residues)
            all_angles.append(angles)

    # Check that all angles is not empty. if it is throw an error
    if not all_angles:
        raise ValueError(f"No angles found for {folder_path}")
    return all_angles


def main(input_folder, fragments_path):
    assert input_folder.exists(), f"The folder {input_folder} does not exist."
    assert any(input_folder.iterdir()), f"The folder {input_folder} is empty."

    if fragments_path.exists():
        fragments_df = pd.read_csv(fragments_path)
    else:
        raise FileNotFoundError(
            f"The file {fragments_path} does not exist. Run create_fragment_pdb.py first."
        )

    for folder in input_folder.iterdir():
        if folder.is_dir():
            angles = load_structures_and_calculate_angles(folder, fragments_df)
            angles_array = np.array(angles)
            if len(angles_array.shape) < 2:
                for i, angles in enumerate(angles_array):
                    if isinstance(angles, np.ndarray):
                        print(f"Element {i} shape:", angles.shape)
                    else:
                        print(f"Element {i} length:", len(angles))
                        # Optionally, explore the shape/length of inner elements if they are expected to be uniform
                        inner_shapes = [
                            len(inner) if isinstance(inner, list) else inner.shape
                            for inner in angles
                        ]
                        print(f"  Inner shapes/lengths: {inner_shapes}")
                    raise ValueError(
                        f"Invalid shape for angles array: "
                    )
            np.save(folder / "angles.npy", angles_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PDB structures and calculate angles."
    )
    parser.add_argument(
        "--input_folder",
        type=Path,
        help="Input folder containing subdirectories with PDB files",
    )
    # Create argument for csv all_pdb_paths
    parser.add_argument(
        "--fragments_csv",
        type=Path,
        help="Fragment CSV file all_pdb_paths",
    )

    args = parser.parse_args()
    main(args.input_folder, args.fragments_csv)
