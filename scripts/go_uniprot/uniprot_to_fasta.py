import argparse
from pathlib import Path
import dask.dataframe as dd
import dask
from dask.distributed import Client
import os
from tqdm import tqdm
from src.difference_fn.difference_processing import (
    StructureConvolutionOperator,
    select_first_ampal_assembly,
)
from src.training.data_processing.data_selectors import InterproDatasetSelector


def extract_unique_pdb_chains(df, output_file):
    """
    Extracts unique Uniprot IDs and saves them to a file.
    """
    unique_pdbs = df[df["CHAIN"] == "A"]["UNIPROT"].unique().compute()
    with open(output_file, "w") as f:
        for pdb in unique_pdbs:
            f.write(f"{pdb}\n")
    print(f"Unique PDB chains saved to {output_file}")

def find_matching_pdb_file(pdb_id, pdb_folder):
    """
    Finds a PDB file in the pdb_folder that matches the pdb_id.
    """
    pdb_file = None
    for root, _, files in os.walk(pdb_folder):
        for file in files:
            if pdb_id in file and file.endswith('.pdb'):
                pdb_file = os.path.join(root, file)
                break
        if pdb_file:
            break
    return pdb_file

def parse_sequence(pdb_id, pdb_folder):
    """
    Parses the sequence from the PDB file for the given PDB ID.
    """
    pdb_path = find_matching_pdb_file(pdb_id, pdb_folder)
    
    if pdb_path is None:
        return pdb_id, None
    
    try:
        ampal_container = StructureConvolutionOperator._load_structure(Path(pdb_path))
        structure = select_first_ampal_assembly(ampal_container)
        return pdb_id, structure.sequence
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return pdb_id, None

def extract_sequences(pdb_list_file, pdb_folder, output_fasta):
    """
    Extracts sequences from a list of PDB files and writes them to a FASTA file.
    """
    with open(pdb_list_file, "r") as f:
        pdb_list = f.read().splitlines()

    sequences = {}
    chunk_size = 1000  # Set chunk size for writing to the file
    print(len(pdb_list))
    with open(output_fasta, "w") as fasta_file:
        for i in tqdm(range(0, len(pdb_list), chunk_size), desc="Processing PDB files"):
            chunk = pdb_list[i:i + chunk_size]
            chunk_tasks = {pdb_id: dask.delayed(parse_sequence)(pdb_id, pdb_folder) for pdb_id in chunk}
            chunk_results = dask.compute(chunk_tasks)[0]
            sequences.update(chunk_results)

            for pdb_id, seq in chunk_results.items():
                if seq:
                    fasta_file.write(f">{pdb_id}\n{seq}\n")

    print(f"Sequences saved to {output_fasta}")

def main():
    parser = argparse.ArgumentParser(
        description="Process PDB files to extract sequences and save to a FASTA file"
    )
    parser.add_argument(
        "--input_csv", required=True, help="Input CSV file with Uniprot IDs and chains"
    )
    parser.add_argument(
        "--pdb_folder", required=True, help="Folder containing PDB files"
    )
    parser.add_argument(
        "--output_fasta",
        default="sequences.fasta",
        help="Output FASTA file for sequences",
    )

    args = parser.parse_args()

    # Start Dask client for parallel processing
    client = Client()
    print(f"Dask client started: {client}")

    # Load CSV using Dask
    df = dd.read_csv(args.input_csv)
    output_pdb_list = "unique_pdbs.txt"  # Specify the output file for unique PDBs
    extract_unique_pdb_chains(df, output_pdb_list)
    extract_sequences(output_pdb_list, args.pdb_folder, args.output_fasta)

if __name__ == "__main__":
    main()
