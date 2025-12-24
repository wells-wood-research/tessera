import os
import csv
import time
import requests
import argparse
import concurrent.futures
from tqdm import tqdm

def get_uniprot_entry(protein_id):
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {"query": protein_id, "format": "json", "fields": "accession,id"}

    response = requests.get(base_url, params=params)
    response.raise_for_status()

    results = response.json().get("results", [])
    if results:
        entry_id = results[0]["primaryAccession"]
        return entry_id
    else:
        raise Exception(f"No entry found for protein ID: {protein_id}")

def get_pdb_ids_from_uniprot(entry_id):
    base_url = f"https://rest.uniprot.org/uniprotkb/{entry_id}.json"
    response = requests.get(base_url)
    response.raise_for_status()

    results = response.json()
    pdb_ids = []
    for db_reference in results.get("uniProtKBCrossReferences", []):
        if db_reference["database"] == "PDB":
            pdb_ids.append(db_reference["id"])
    return pdb_ids

def get_alphafold_structure(entry_id):
    base_url = f"https://alphafold.ebi.ac.uk/api/prediction/{entry_id}"

    response = requests.get(base_url)
    response.raise_for_status()

    results = response.json()
    if results:
        pdb_url = results[0].get("pdbUrl")
        if pdb_url:
            return pdb_url
        else:
            raise Exception(f"No PDB URL found in AlphaFold results for entry ID: {entry_id}")
    else:
        raise Exception(f"No AlphaFold structure found for entry ID: {entry_id}")

def get_pdbe_structure(pdb_id):
    base_url = f"https://www.ebi.ac.uk/pdbe-srv/view/entry/{pdb_id}"
    response = requests.get(base_url)
    response.raise_for_status()

    if response.status_code == 200:
        return base_url
    else:
        raise Exception(f"No PDBe structure found for PDB ID: {pdb_id}")

def download_structure(download_url, output_dir, protein_name):
    response = requests.get(download_url)
    response.raise_for_status()

    filename = os.path.join(output_dir, f"{protein_name}.pdb")
    with open(filename, "wb") as file:
        file.write(response.content)

def process_protein(protein_id, output_dir, error_list):
    attempts = 0
    success = False
    entry_id = None

    while attempts < 3 and not success:
        try:
            entry_id = get_uniprot_entry(protein_id)
            if entry_id:
                try:
                    structure_url = get_alphafold_structure(entry_id)
                except Exception as e:
                    # print(f"AlphaFold structure not found for {entry_id}: {e}")
                    pdb_ids = get_pdb_ids_from_uniprot(entry_id)
                    if pdb_ids:
                        structure_url = get_pdbe_structure(pdb_ids[0])
                    else:
                        raise Exception(f"No PDB IDs found for entry ID: {entry_id}")
                download_structure(structure_url, output_dir, protein_id)
                success = True
            else:
                raise Exception(f"Entry ID not found for protein ID: {protein_id}")
        except Exception as e:
            attempts += 1
            # print(f"An error occurred for {protein_id} on attempt {attempts}: {e}")
            if attempts < 3:
                time.sleep(5)  # Wait for 5 seconds before retrying

    if not success:
        error_list.append(protein_id)

def main(input_file, output_dir, percentage):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        proteins = [row["proteins"] for row in reader]

    # Select the first x% of proteins based on the percentage
    num_proteins = len(proteins)
    num_to_download = int(num_proteins * percentage / 100)
    proteins_to_download = proteins[:num_to_download]

    error_list = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_protein, protein, output_dir, error_list)
            for protein in proteins_to_download
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing proteins"):
            future.result()  # Wait for all threads to complete

    if error_list:
        with open(os.path.join(output_dir, "error.txt"), "w") as error_file:
            for protein in error_list:
                error_file.write(f"{protein}\n")
        print(
            f"Errors occurred for the following proteins and were written to error.txt"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download AlphaFold structures for a list of proteins."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        type=str,
        help="Path to the input CSV file containing protein IDs.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory to save the downloaded structures.",
    )
    parser.add_argument(
        "--percentage",
        required=True,
        type=int,
        default=100,
        help="Percentage of proteins to download (default is 100%).",
    )

    args = parser.parse_args()

    main(args.input_file, args.output_dir, args.percentage)
