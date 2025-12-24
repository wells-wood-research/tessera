import os
import requests
import gzip
import shutil

# Mapping of categories to PDB codes
pdb_categories = {
    "DNA": ["10MH", "1A3Q"],
    "Metal": ["101M", "1A42"],
    "ATP": ["11AS", "1A44"],
    "GTP": ["1C80", "1BKD"],
    "RNA": ["1914", "1A9N"],
    "DNA+Metal": ["1A1F", "1EE8"],
    "RNA+Metal": ["1A5Y", "1F4E"],
    "DNA+ATP+Metal": ["1A0I", "1CR0"],
    "RNA+ATP+Metal": ["1A49", "1BX4"]
}

def download_and_unzip_pdb(pdb_code: str, pdb_type: str, save_dir: str = "./pdb_files"):
    """
    Downloads a PDB file from the RCSB PDB website in .pdb1.gz format, decompresses it,
    and saves it with the type in the file name.

    :param pdb_code: str, The PDB code of the structure to download.
    :param pdb_type: str, The category type (e.g., DNA, RNA, Metal, etc.).
    :param save_dir: str, The directory to save the downloaded PDB file.
    """
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Construct the download URL and the file save path
    url = f"https://files.rcsb.org/download/{pdb_code}.pdb1.gz"
    save_path_gz = os.path.join(save_dir, f"{pdb_code}_{pdb_type}.pdb1.gz")
    save_path_pdb = os.path.join(save_dir, f"{pdb_code}_{pdb_type}.pdb1")

    # Download the file
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path_gz, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {pdb_code} and saved as {save_path_gz}")

        # Unzip the file
        with gzip.open(save_path_gz, 'rb') as f_in:
            with open(save_path_pdb, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Unzipped {pdb_code} and saved as {save_path_pdb}")

        # Remove the .gz file after extraction
        os.remove(save_path_gz)
    else:
        print(f"Failed to download {pdb_code}: HTTP {response.status_code}")


def download_multiple_pdb_files(pdb_categories: dict, save_dir: str = "./pdb_files"):
    """
    Downloads multiple PDB files based on the categories provided.

    :param pdb_categories: dict, A dictionary mapping categories (e.g., DNA, RNA, etc.) to lists of PDB codes.
    :param save_dir: str, Directory to save the downloaded PDB files.
    """
    for category, pdb_codes in pdb_categories.items():
        for pdb_code in pdb_codes:
            download_and_unzip_pdb(pdb_code, category, save_dir)


# Call the function to download all the PDB files
download_multiple_pdb_files(pdb_categories)
