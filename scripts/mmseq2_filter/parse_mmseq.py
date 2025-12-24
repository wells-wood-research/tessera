import argparse
from datetime import datetime
import pandas as pd
from src.training.data_processing.data_selectors import InterproDatasetSelector


def create_representatives_csv(clusters_file, input_csv, output_csv):
    # Read clusters with sequence similarity information
    clusters = pd.read_csv(clusters_file, sep="\t", header=None, names=["representative", "member", "similarity"])

    # Load dataframe using InterproDatasetSelector
    df = InterproDatasetSelector.load_interpro_to_df(input_csv)
    
    # Merge the cluster information with the original dataframe
    representative_data = df.merge(clusters, left_on='PDB', right_on='member', how='inner')

    # Extract header from input CSV
    with open(input_csv, 'r') as file:
        header = file.readline().strip()

    # Generate date info
    date_info = datetime.now().strftime("# %Y/%m/%d - %H:%M")
    mmseqs_info = " | MMseqs2: Similarity Threshold 40%"
    header_info = f"{date_info} | {header}{mmseqs_info}\nPDB,CHAIN,INTERPRO_ID,CLUSTER_REPRESENTATIVE,SIMILARITY\n"

    # Write the header and representative data to the output CSV
    with open(output_csv, "w") as f:
        f.write(header_info)
        representative_data.to_csv(f, index=False, columns=["PDB", "CHAIN", "INTERPRO_ID", "representative", "similarity"], header=False)
    print(f"Representative PDBs saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Process MMseqs2 clustering results")
    parser.add_argument("--clusters_file", required=True, help="Cluster results file from MMseqs2")
    parser.add_argument("--input_csv", required=True, help="Input CSV file with original PDB data")
    parser.add_argument("--output_csv", default="representative_pdbs.csv", help="Output CSV file for representative PDBs")

    args = parser.parse_args()

    create_representatives_csv(args.clusters_file, args.input_csv, args.output_csv)

if __name__ == "__main__":
    main()
