import argparse
from pathlib import Path
from multiprocessing import Pool
from tessera.difference_fn.difference_processing import (
    StructureConvolutionOperator,
    select_first_ampal_assembly,
)
from tessera.training.data_processing.data_selectors import InterproDatasetSelector


def extract_unique_pdb_chains(df, output_file):
    unique_pdbs = df[df["CHAIN"] == "A"]["PDB"].unique()
    with open(output_file, "w") as f:
        for pdb in unique_pdbs:
            f.write(f"{pdb}\n")
    print(f"Unique PDB chains saved to {output_file}")


def parse_sequence(pdb_id_pdb_folder):
    pdb_id, pdb_folder = pdb_id_pdb_folder
    pdb_path = f"{pdb_folder}/{pdb_id[1:3]}/{pdb_id}.pdb1.gz"
    try:
        ampal_container = StructureConvolutionOperator._load_structure(Path(pdb_path))
        structure = select_first_ampal_assembly(ampal_container)
        return pdb_id, structure.sequence
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return pdb_id, None


def extract_sequences(pdb_list_file, pdb_folder, output_fasta, num_processes):
    with open(pdb_list_file, "r") as f:
        pdb_list = f.read().splitlines()

    pdb_id_pdb_folder = [(pdb_id, pdb_folder) for pdb_id in pdb_list]

    with Pool(num_processes) as pool:
        sequences = dict(pool.map(parse_sequence, pdb_id_pdb_folder))

    with open(output_fasta, "w") as f:
        for pdb_id, seq in sequences.items():
            if seq:
                f.write(f">{pdb_id}\n{seq}\n")
    print(f"Sequences saved to {output_fasta}")


def main():
    parser = argparse.ArgumentParser(
        description="Process PDB files for MMseqs2 clustering"
    )
    parser.add_argument(
        "--input_csv", required=True, help="Input CSV file with PDB chains"
    )
    parser.add_argument(
        "--output_pdb_list",
        default="unique_pdbs.txt",
        help="Output file for unique PDB list",
    )
    parser.add_argument(
        "--pdb_folder", required=True, help="Folder containing PDB files"
    )
    parser.add_argument(
        "--output_fasta",
        default="sequences.fasta",
        help="Output FASTA file for sequences",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of processes to use for multiprocessing",
    )

    args = parser.parse_args()

    df = InterproDatasetSelector.load_interpro_to_df(args.input_csv)
    extract_unique_pdb_chains(df, args.output_pdb_list)
    extract_sequences(
        args.output_pdb_list, args.pdb_folder, args.output_fasta, args.num_processes
    )


if __name__ == "__main__":
    main()
