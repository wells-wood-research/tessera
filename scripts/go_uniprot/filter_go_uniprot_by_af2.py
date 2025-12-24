import dask.dataframe as dd
import os
import argparse
from dask.distributed import Client

def get_existing_uniprot_ids(folder_path):
    files = os.listdir(folder_path)
    # Extract Uniprot IDs from the filenames
    uniprot_ids = {file.split('-')[1] for file in files if '-' in file}
    print(f"Extracted Uniprot IDs: {uniprot_ids}")  # Debugging line
    return uniprot_ids

def filter_dataframe(csv_file_path, folder_path):
    df = dd.read_csv(csv_file_path)
    existing_uniprot_ids = get_existing_uniprot_ids(folder_path)
    
    # Filtering DataFrame with Dask
    df_filtered = df[df['UNIPROT'].map_partitions(lambda x: x.isin(existing_uniprot_ids))]
    print(f"Filtered DataFrame before compute: {df_filtered}")  # Debugging line
    return df_filtered

def main():
    parser = argparse.ArgumentParser(description='Filter a CSV file based on the existence of files containing Uniprot IDs in a specified folder.')
    parser.add_argument('csv_file_path', type=str, help='Path to the CSV file.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing files to check.')
    parser.add_argument('output_file_path', type=str, help='Path to save the filtered CSV file.')

    args = parser.parse_args()

    client = Client()  # Start a Dask client to manage resources
    print("Dask client started:", client)

    filtered_df = filter_dataframe(args.csv_file_path, args.folder_path)
    # Compute to trigger actual computation and convert to pandas DataFrame
    filtered_df_pd = filtered_df.compute()
    print(f"Filtered DataFrame after compute: {filtered_df_pd}")  # Debugging line

    filtered_df_pd.to_csv(args.output_file_path, index=False)
    print(f"Filtered CSV saved to {args.output_file_path}")

if __name__ == '__main__':
    main()
