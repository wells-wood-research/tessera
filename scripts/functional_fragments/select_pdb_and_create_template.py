import json
import argparse
import random
from typing import Dict, List, Tuple
from pathlib import Path
import requests
import ampal
from src.fragments.classification_config import go_to_prosite


def download_pdb(pdb_id: str, output_dir: Path) -> bool:
    """
    Downloads a PDB file from the RCSB PDB database and saves it to the output directory.

    Parameters
    ----------
    pdb_id
    output_dir

    Returns
    -------

    """
    # Download the PDB file
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    pdb_path = output_dir / f"{pdb_id}.pdb"
    with pdb_path.open("wb") as file:
        response = requests.get(pdb_url)
        # If the request was successful, write the content to the file and return true
        if response.status_code == 200:
            file.write(response.content)
            return True
        # If the request was not successful, return false
        else:
            print(
                f"Failed to download PDB file {pdb_id} with status code {response.status_code}."
            )
            return False


def get_entity_sequence(pdb_path: Path, chain_id: str) -> str:
    # Get PDB name
    pdb_name = pdb_path.stem
    pdb_entity = f"{pdb_name}_{chain_id}"
    # Query RCSB for the sequence
    url = f"https://www.rcsb.org/fasta/entity/{pdb_entity}/download"
    response = requests.get(url)
    # Extract the sequence from the fasta response (second line)
    response_lines = response.text.split("\n")
    pdb_sequence = response_lines[1]
    return pdb_sequence


def save_pdb_chain(structure: ampal.Polypeptide, chain_id: str, pdb_path: Path) -> bool:
    """
    Saves a specific chain from a PDB file to a new PDB file.

    Parameters
    ----------
    structure
    chain_id
    pdb_path

    Returns
    -------

    """
    # Assert this is a Polypeptide
    if not isinstance(structure, ampal.Polypeptide):
        print(f"Chain {chain_id} is not a polypeptide.")
        return False
    entity_path = pdb_path.parent / f"{pdb_path.stem}_{chain_id}.pdb"
    # Save the chain to a new PDB file
    with entity_path.open("w") as file:
        file.write(str(structure.pdb))
        if not entity_path.exists():
            print(f"Failed to save chain {chain_id} to {pdb_path}.")
            return False
    return True


# def select_entity(pdb_path: Path, chain_id: str) -> bool:
#     """
#     Selects a specific chain from a PDB file and saves it to a new file.
#     Parameters
#     ----------
#     pdb_path
#     chain_id
#
#     Returns
#     -------
#
#     """
#     # Load the PDB file
#     structure = ampal.load_pdb(pdb_path)
#     # Get the chain sequence from the RCSB database
#     pdb_sequence = get_entity_sequence(pdb_path, chain_id)
#     # Check if the sequence matches the sequence in the PDB file
#     for curr_structure in structure:
#         if isinstance(curr_structure, ampal.Polypeptide):
#             curr_chain_sequence = curr_structure.sequence
#             # If the sequence matches, save the chain to a new PDB file
#             if curr_chain_sequence == pdb_sequence:
#                 return save_pdb_chain(curr_structure, chain_id, pdb_path)
#             # Sequence can match if it is a subset of the sequence - Because the PDB sucks
#             elif (pdb_sequence in curr_chain_sequence) or (
#                 curr_chain_sequence in pdb_sequence
#             ):
#                 return save_pdb_chain(curr_structure, chain_id, pdb_path)
#     print(f"Failed to find chain {chain_id} in {pdb_path}.")
#     return False
import requests
import gzip
from io import BytesIO
import xml.etree.ElementTree as ET
import typing as t


def get_entity_to_chain(pdb_id: str) -> t.Dict[int, t.List[str]]:
    url = f"https://files.rcsb.org/download/{pdb_id}.xml.gz"
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    # Decompress the gzipped content
    with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz:
        decompressed_content = gz.read()

    # Parse the XML content
    root = ET.fromstring(decompressed_content)
    namespace = {'PDBx': root.tag.split('}')[0].strip('{')}

    entity_chain_mapping = {}

    # Extracting the relevant data
    for struct_asym in root.findall('.//PDBx:struct_asym', namespaces=namespace):
        chain_id = struct_asym.get('id')
        entity_id = int(struct_asym.find('PDBx:entity_id', namespaces=namespace).text)

        # Append the chain_id to the list if entity_id exists; otherwise, initialize the list
        if entity_id in entity_chain_mapping:
            if chain_id not in entity_chain_mapping[entity_id]:  # Avoid duplicates
                entity_chain_mapping[entity_id].append(chain_id)
        else:
            entity_chain_mapping[entity_id] = [chain_id]

    return entity_chain_mapping


def select_entity(pdb_path: Path, entity_id: str) -> bool:
    """
    Selects a specific chain from a PDB file and saves it to a new file.
    Parameters
    ----------
    pdb_path
    entity_id

    Returns
    -------

    """
    # Load the PDB file
    structure = ampal.load_pdb(pdb_path)
    if isinstance(structure, ampal.AmpalContainer):
        structure = structure[0]
    # Get the chain sequence from the RCSB database
    entity_to_chain_dict = get_entity_to_chain(pdb_path.stem)
    # Check if the chain ID is valid
    if int(entity_id) not in entity_to_chain_dict:
        print(f"Entity {entity_id} not found in {pdb_path}.")
        return False
    # Check if chain is in the structure
    chain_id = entity_to_chain_dict[int(entity_id)][0]
    # Check if the chain is in the structure - Ampal does not handle this properly so we need to try and except
    # >>> structure['D']
    # Out[9]: <Polypeptide containing 203 Residues. Sequence: TVKLDTMIFGVI...>
    # >>> 'D' in structure
    # Out[10]: False
    try:
        structure[chain_id]
    except KeyError:
        print(f"Chain {chain_id} not found in {pdb_path}.")
        return False
    return save_pdb_chain(structure[chain_id], entity_id, pdb_path)


def parse_query_results(query_dir: Path) -> Dict[Tuple[str], List[str]]:
    """
    Parses JSON query result files, selecting 10 identifiers at random from the result set
    in each file and storing them in a dictionary.
    """
    go_to_pdb_ids = {}

    # Iterate over each JSON file in the query_results directory
    for file_path in query_dir.glob("*_results.json"):
        with file_path.open("r") as file:
            data = json.load(file)

        # Extract filename without extension and "_results" suffix
        filename_str = file_path.stem.replace("_results", "")
        # Extract GO codes from filename
        query_go = filename_str.split("+")
        # Convert to tuple for dictionary key
        query_id = tuple(sorted(query_go))
        # Extract query_id and result_set identifiers
        result_set = data.get("result_set", [])

        # Shuffle identifiers
        if result_set:
            identifiers = [entry["identifier"] for entry in result_set]
            go_to_pdb_ids[query_id] = identifiers
        else:
            raise ValueError(
                f"No result set found in {file_path}. Please check that this is the results JSON file."
            )

    return go_to_pdb_ids


def main(args: argparse.Namespace):
    # Set the random seed for reproducibility
    random.seed(args.seed)
    # Blacklist of PDBs that are not useful / used
    blacklist = ["1A0B", "1D1N", "1H99", "1ID0", "1IR3", "1JAD", "1NSQ", "1Q2L", "1Q31", "1Q57", "1UM2", "1V47", "1WQS", "1WXL", "2AFF", "2BCW", "2CJQ", "2CJR", "2CS1", "2DK1", "2DWQ", "2FH5", "2G2K", "2GX5", "2K85", "2KUE", "2LKC", "2LKZ", "2MFR", "2N51", "2NA2", "2NRR", "2O2V", "2OPU", "2PMY", "2Q2E", "2Q7U", "2QMH", "2QY2", "2WX4", "2Y9Y", "2YI9", "2YKH", "3A1G", "3AI4", "3BLE", "3BLQ", "3C5H", "3CXH", "3EPK", "3FZM", "3H63", "3HRT", "3IBP", "3KMP", "3KVT", "3MGZ", "3MP2", "3O47", "3RGH", "3RQI", "3T12", "3TIX", "3TWL", "3VKP", "3VPY", "4A69", "4ACB", "4ARZ", "4ASN", "4B47", "4BZQ", "4CBL", "4DW4", "4E6N", "4FWT", "4GEH", "4GP7", "4IAO", "4IJX", "4LAW", "4MN4", "4N3S", "4OGA", "4OXP", "4QS7", "4R4M", "4R71", "4RD6", "4U12", "4U4P", "4UUD", "4WBZ", "4XJ1", "4ZGQ", "4ZU9", "5C1T", "5CA9", "5CSA", "5DIS", "5DV7", "5EYA", "5FWH", "5HNO", "5HZH", "5I4Q", "5IRC", "5IRR", "5IZL", "5LBD", "5LM7", "5LOJ", "5LUT", "5MLC", "5O5S", "5UJE", "5X3S", "5YWW", "5ZZ7", "6BKG", "6DI7", "6DS6", "6E0M", "6EHR", "6F5D", "6G0Y", "6H2X", "6IY6", "6J72", "6KWR", "6LUR", "6MD3", "6O56", "6O8H", "6RIE", "6VKJ", "6WG6", "6X1M", "6XR4", "6ZHU", "6ZN8", "7B7Z", "7C7L", "7CRW", "7DPQ", "7EGY", "7EPD", "7FID", "7GNZ", "7JQQ", "7KSL", "7LJN", "7MSS", "7O1Q", "7O9G", "7OQC", "7RY1", "7V3W", "7VUF", "7W02", "7WU8", "7XC4", "7Y04", "7YZ8", "7Z6L", "7ZJ1", "7ZR1", "8ATD", "8B0J", "8BAH", "8CYL", "8D9S", "8FX4", "8H5V", "8JF7", "8PAG", "8PK5", "8U5H", "8W4J", "8YB5", "9ASP", "9BE2", "9F5B", "9FOF"]
    # Parse the query results and select identifiers
    go_to_pdb_ids = parse_query_results(args.query_dir)
    # Create output directory if it does not exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Iterate over each GO code, download the PDB files, and create a template
    go_to_selection = {}
    for go_code in go_to_pdb_ids:
        selected_pdb = []
        # Convert go code to text
        go_descr = "+".join(
            go_to_prosite[curr_go]["description"].split(" ")[0].upper()
            for curr_go in go_code
        )
        # Create a directory for each GO code
        go_dir = args.output_dir / go_descr
        go_dir.mkdir(parents=True, exist_ok=True)
        while (
            len(selected_pdb) < min(args.n_samples, len(go_to_pdb_ids[go_code]))
            or len(go_to_pdb_ids[go_code]) == 0
        ):
            # Select a random PDB entity
            pdb_entity = random.choice(go_to_pdb_ids[go_code])
            if pdb_entity in blacklist:
                go_to_pdb_ids[go_code].remove(pdb_entity)
                continue
            print(f"Selecting {pdb_entity} for {go_code}")
            # Extract PDB ID
            pdb_id, entity_id = pdb_entity.split("_")
            # Check if the PDB file already exists
            pdb_path = go_dir / f"{pdb_id}.pdb"
            if not pdb_path.exists():
                was_pdb_downloaded = download_pdb(pdb_id, go_dir)
                if not was_pdb_downloaded:
                    continue  # Potentially add errors

            # Select a specific chain from the PDB file
            pdb_chain_path = go_dir / f"{pdb_id}_{entity_id}.pdb"
            if not pdb_chain_path.exists():
                try:
                    was_chain_selected = select_entity(pdb_path, entity_id)
                    if not was_chain_selected:
                        # Delete the PDB file if the chain was not selected
                        pdb_path.unlink()
                        continue  # Potentially add errors
                except ValueError:  # Due to Malformed PDB files
                    # Delete the PDB file if the chain was not selected
                    pdb_path.unlink()
                    continue
            pdb_path.unlink()
            # Add go text to the selected pdb
            pdb_chain_path.rename(go_dir / f"{pdb_id}_{go_descr}.pdb")
            # Add the selected PDB to the list
            selected_pdb.append(pdb_chain_path)
            # Remove the selected PDB from the list of available PDBs
            go_to_pdb_ids[go_code].remove(pdb_entity)
        # Save the selected PDBs to the output directory
        go_to_selection[go_code] = selected_pdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse query results and select 10 random identifiers."
    )
    parser.add_argument(
        "--query_dir",
        type=Path,
        default=Path("query_results"),
        help="Directory containing JSON query result files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("selected_pdbs"),
        help="Directory to save the selected PDBs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Number of identifiers to select from each query result",
    )
    params = parser.parse_args()
    main(params)
