import argparse
import shutil
from pathlib import Path
import pandas as pd
import requests
from ampal import load_pdb, AmpalContainer, Assembly, Polypeptide


def download_pdb(pdb_id, pdb_dir, recreate):
    """Downloads a PDB file given a PDB ID, optionally overwriting existing files."""
    pdb_file_path = pdb_dir / f"{pdb_id}.pdb1"
    if (
        not pdb_file_path.exists() or recreate
    ):  # Check if PDB file already exists or recreate is True
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb1"
        response = requests.get(url)
        if response.status_code == 200:
            with open(pdb_file_path, "w") as file:
                file.write(response.text)
        else:
            raise ValueError(
                f"Failed to download {pdb_id}, status code {response.status_code}"
            )


def create_fragments(pdb_id, pdb_dir, fragment_dir, row, recreate):
    """Processes a PDB file using AMPAL, selects specified chain and residues, optionally overwriting existing files."""
    chain_id, start, end, fragment_number = (
        row["Chain"],
        int(row["Start Residue"]),
        int(row["End Residue"]),
        row["Fragment Number"],
    )
    fragment_output_dir = fragment_dir / str(fragment_number)
    fragment_output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = fragment_output_dir / f"{pdb_id}.pdb1"
    if not output_file_path.exists() or recreate:
        print(f"Creating fragment {fragment_number} for {pdb_id}...")
        pdb_path = pdb_dir / f"{pdb_id}.pdb1"
        pdb_structure = load_pdb(str(pdb_path))
        # Select first state of container:
        if isinstance(pdb_structure, AmpalContainer):
            pdb_structure = pdb_structure[0]
        selected_chain = pdb_structure[chain_id]
        selected_region = selected_chain.get_slice_from_res_id(start, end).backbone

        # Verify amino acid sequence
        csv_sequence = row["Amino Acid Sequence"].replace("-", "").upper()
        ampal_sequence = selected_region.sequence

        if "X" in ampal_sequence:
            ampal_sequence = ampal_sequence.replace("X", "")
            x_indices = [
                i for i, char in enumerate(selected_region.sequence) if char == "X"
            ]
            # Remove characters at the indices of 'X' from the AMPAL sequence in the CSV sequence
            for index in sorted(
                x_indices, reverse=True
            ):  # Reverse sort to delete from end to avoid index shifting
                if index < len(csv_sequence):  # Ensure index is within bounds
                    csv_sequence = csv_sequence[:index] + csv_sequence[index + 1 :]

        if ampal_sequence != csv_sequence:
            raise ValueError(f"Sequence mismatch for {pdb_id} {chain_id} {start}-{end}")

        pdb_text = selected_region.make_pdb()
        with open(output_file_path, "w") as pdb_file:
            pdb_file.write(pdb_text)
    else:
        print(f"Fragment {fragment_number} for {pdb_id} already exists. Skipping.")


def main(input_filepath, output_filepath, recreate):
    """Main function to parse table and process PDB files, with an option to recreate the dataset."""
    if not input_filepath.exists():
        raise FileNotFoundError(f"The file {input_filepath} does not exist.")

    pdb_dir = output_filepath / "pdbs"
    fragment_dir = output_filepath / "fragments"

    pdb_dir.mkdir(parents=True, exist_ok=True)
    fragment_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_filepath)
    for index, row in df.iterrows():
        pdb_id = row["Identifier"][:4]
        download_pdb(pdb_id, pdb_dir, recreate)
        create_fragments(pdb_id, pdb_dir, fragment_dir, row, recreate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PDB files based on table data_paths."
    )
    parser.add_argument(
        "--input_filepath",
        type=Path,
        help="Path to the input CSV file.",
        default=Path("../../data/fragments.csv"),
    )
    parser.add_argument(
        "--output_filepath",
        type=Path,
        help="Path to the output directory.",
        default=Path("../../data/"),
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the dataset from scratch by overwriting existing files.",
    )

    args = parser.parse_args()
    main(args.input_filepath, args.output_filepath, args.recreate)
