import concurrent.futures
import json
import math
import multiprocessing as mp
import pickle
import time
import typing as t
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Set

import ampal
import pandas as pd
import requests
import scipy.sparse as ssp
import torch
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.difference_fn.difference_processing import select_first_ampal_assembly
from src.fragments.classification_config import (FUNC_DICT, NAMESPACES_REVERT, UniprotResults, go_to_prosite, )


class BeProfLoader:
    """
    Load BeProf dataset from the given path
    """

    def __init__(
        self, beprof_path: Path, go_category: str = "all", debug_mode: bool = False
    ):
        """
        Initialize the BeProfLoader class with the given path, category, and debug mode.

        Parameters
        ----------
        beprof_path: Path
            Path to the BeProf dataset.
        go_category: str
            Category of data to load (cc, mf, bp, all).
        debug_mode: bool
            Enable debug mode to load only first 10 entries of each dataset.
        """
        assert beprof_path.exists(), f"Input file {beprof_path} does not exist"
        self.beprof_path = beprof_path
        self.go_category = go_category
        self.debug_mode = debug_mode
        self.data_dict: t.Dict[str, t.Dict[str, t.Any]] = {
            split: {} for split in ["train", "valid", "test"]
        }

    def load_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Load a CSV file and return its content as a DataFrame.

        Parameters
        ----------
        file_path: Path
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the CSV file content. In debug mode, it contains only the first 10 entries.
        """
        df = pd.read_csv(file_path, index_col=0)
        if self.debug_mode:
            df = df.head(10)
            assert len(df) <= 10, "Debug mode: More than 10 entries loaded from CSV"
        return df

    def load_pkl(self, file_path: Path) -> t.Dict[str, t.Any]:
        """
        Load a pickle file and return its content.

        Parameters
        ----------
        file_path: Path
            Path to the pickle file.

        Returns
        -------
        t.Any
            Content of the pickle file. In debug mode, it contains only the first 10 entries.
        """
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            if self.debug_mode:
                if isinstance(data, list):
                    data = data[:10]
                    assert (
                        len(data) <= 10
                    ), "Debug mode: More than 10 entries loaded from PKL list"
                elif isinstance(data, dict):
                    limited_data = {
                        k: v for i, (k, v) in enumerate(data.items()) if i < 10
                    }
                    data = limited_data
                    assert (
                        len(data) <= 10
                    ), "Debug mode: More than 10 entries loaded from PKL dict"
            return data

    def load_split_data(self, split: str) -> None:
        """
        Load data for a specific split (train, valid, test) and store it in the data dictionary.

        Parameters
        ----------
        split: str
            The split to load data for ("train", "valid", "test").
        """
        if self.go_category == "all":
            self.data_dict[split]["labels"] = self.load_pkl(
                self.beprof_path / f"{split}_data" / f"{split}_data_labels.pkl"
            )
            self.data_dict[split]["proteins"] = self.load_csv(
                self.beprof_path / f"{split}_data" / f"{split}_data_proteins.csv"
            )
            self.data_dict[split]["sequences"] = self.load_pkl(
                self.beprof_path / f"{split}_data" / f"{split}_data_sequences.pkl"
            )
        else:
            prefix = f"{split}_data_separate_{self.go_category}"
            self.data_dict[split][f"{self.go_category}_labels"] = self.load_pkl(
                self.beprof_path / f"{split}_data" / f"{prefix}_labels.pkl"
            )
            self.data_dict[split][f"{self.go_category}_proteins"] = self.load_csv(
                self.beprof_path / f"{split}_data" / f"{prefix}_proteins.csv"
            )
            self.data_dict[split][f"{self.go_category}_sequences"] = self.load_pkl(
                self.beprof_path / f"{split}_data" / f"{prefix}_sequences.pkl"
            )

    def load_data(self) -> t.Dict[str, t.Dict[str, t.Any]]:
        """
        Load all data splits and return the data dictionary.

        Returns
        -------
        t.Dict[str, t.Dict[str, t.Any]]
            Dictionary containing the loaded data for all splits.
        """
        for split in self.data_dict.keys():
            self.load_split_data(split)
            if self.debug_mode:
                for key, value in self.data_dict[split].items():
                    if isinstance(value, pd.DataFrame):
                        assert (
                            len(value) <= 10
                        ), f"Debug mode: More than 10 entries in DataFrame for {key} in split {split}"
                    elif isinstance(value, list):
                        assert (
                            len(value) <= 10
                        ), f"Debug mode: More than 10 entries in list for {key} in split {split}"
                    elif isinstance(value, dict):
                        assert (
                            len(value) <= 10
                        ), f"Debug mode: More than 10 entries in dictionary for {key} in split {split}"

        return self.data_dict


class UniprotDownloader:
    """
    Given a list of UniProt Proteins, download the corresponding AlphaFold PDB files.

    If they are not available, download the corresponding UniProt PDB files.

    TODO: If that is not available, we use Facebook's ESM model to predict the protein structure.

    And if that fails, well, we are out of luck.
    """

    def __init__(self, output_dir: Path, verbose: bool = False):
        """
        Initialize the UniprotDownloader class with input data dictionary and output directory.

        Parameters
        ----------
        output_dir : Path
            Directory to save the downloaded structures.
        """
        self.output_dir = output_dir / "uniprot_pdb"
        self.verbose = verbose
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_list: t.Set[str] = set()
        self.error_file = self.output_dir / "error.txt"
        self.uniprot_to_path_dict: t.Dict[str, Path] = {}
        self.uniprot_to_accession_dict: t.Dict[str, str] = {}
        self.dict_file = self.output_dir / "uniprot_to_accession.json"
        self._load_uniprot_to_accession_dict()

    def _reset_class(self):
        """
        Reset the class attributes.
        """
        self.error_list = set()
        self.uniprot_to_path_dict = {}
        self._load_uniprot_to_accession_dict()

    def _load_uniprot_to_accession_dict(self):
        """
        Load the uniprot_to_accession_dict from a JSON file if it exists.
        """
        if self.dict_file.exists():
            try:
                with open(self.dict_file, "r") as file:
                    self.uniprot_to_accession_dict = json.load(file)
            # If the file is empty:
            except json.JSONDecodeError:
                self.uniprot_to_accession_dict = {}
        else:
            self.uniprot_to_accession_dict = {}

    def _save_uniprot_to_accession_dict(self):
        """
        Save the uniprot_to_accession_dict to a JSON file.
        """
        # Load existing data if it exists
        if self.dict_file.exists():
            try:
                with open(self.dict_file, "r") as file:
                    existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = {}
        else:
            existing_data = {}

        # Update existing data with new entries
        existing_data.update(self.uniprot_to_accession_dict)

        # Save the combined data back to the JSON file
        with open(self.dict_file, "w") as file:
            json.dump(existing_data, file)

    @staticmethod
    def get_uniprot_entry(protein_id: str) -> UniprotResults:
        """
        Retrieve the UniProt entry ID and the length of the protein sequence for a given protein ID.

        Parameters
        ----------
        protein_id : str
            The protein ID to query.

        Returns
        -------
        UniprotResults
            A named tuple containing the UniProt entry ID, sequence length, and GO codes.
        """
        base_url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": protein_id,
            "format": "json",
            # "fields": "accession,id,uniProtKBCrossReferences,sequence",
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()

        results = response.json().get("results", [])
        if results:
            # Check if entryType equals to Inactive. If so raise an exception
            if results[0]["entryType"] == "Inactive":
                raise Exception(f"Entry {protein_id} is inactive")
            uniprot_id = results[0]["primaryAccession"]
            sequence_length = results[0]["sequence"]["length"]
            cross_references = results[0].get("uniProtKBCrossReferences", [])
            go_codes = []
            for ref in cross_references:
                if ref["database"] == "GO":
                    go_codes.append(ref["id"])

            return UniprotResults(uniprot_id, sequence_length, go_codes)
        else:
            raise Exception(f"No entry found for protein ID: {protein_id}")

    def get_active_site_from_uniprot(
        self, protein_id: str
    ) -> t.List[t.Dict[str, t.Any]]:
        # Check if the protein pdb already exists:
        output_file = self.output_dir / f"{protein_id}.json"
        if output_file.exists():
            full_entry = json.load(output_file.open("r"))
        else:
            # Retrieve the full entry to extract binding site information
            full_entry_url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
            full_entry_response = requests.get(full_entry_url)
            full_entry_response.raise_for_status()
            full_entry = full_entry_response.json()

            # Save the full entry to a JSON file
            with open(output_file, "w") as file:
                json.dump(full_entry, file)

        relevant_sites = []
        features = full_entry.get("features", [])

        for feature in features:
            feature_type = feature.get("type", "").lower()
            prosite_rule_id = None
            # Check if it is a PROSITE-ProRule feature
            if any(
                evidence.get("source", "").lower() == "prosite-prorule"
                for evidence in feature.get("evidences", [])
            ):
                prosite_rule_id = feature.get("evidences", [{}])[0].get("id", "")

            # Else, check if it is DNA or RNA binding
            elif feature_type == "dna binding":
                prosite_rule_id = "PRU00655"
            elif feature_type == "rna binding":
                prosite_rule_id = "PRU01085"

            # Else, check if it is a LIGAND feature (ATP, GTP, or metal ion)
            else:
                ligand_name = feature.get("ligand", {}).get("name", "").lower()
                if "atp" in ligand_name:
                    prosite_rule_id = "PRU00434"
                elif "gtp" in ligand_name:
                    prosite_rule_id = "PRU01057"
                elif "+" in ligand_name:
                    prosite_rule_id = "PRU01192"

            # If any relevant rule was found, add the feature to the list
            if prosite_rule_id:
                binding_site_info = {
                    "description": feature.get("description", ""),
                    "type": feature.get("type", ""),
                    "location_start": feature.get("location", {})
                    .get("start", {})
                    .get("value"),
                    "location_end": feature.get("location", {})
                    .get("end", {})
                    .get("value"),
                    "evidence_code": feature.get("evidences", [{}])[0].get(
                        "evidenceCode", ""
                    ),
                    "prosite_rule_id": prosite_rule_id,
                }
                relevant_sites.append(binding_site_info)

        return relevant_sites

    @staticmethod
    def get_pdb_ids_from_uniprot(entry_id: str) -> t.List[str]:
        """
        Retrieve PDB IDs associated with a given UniProt entry ID.

        Parameters
        ----------
        entry_id : str
            The UniProt entry ID.

        Returns
        -------
        List[str]
            List of PDB IDs.
        """
        base_url = f"https://rest.uniprot.org/uniprotkb/{entry_id}.json"
        response = requests.get(base_url)
        response.raise_for_status()

        results = response.json()
        pdb_ids = [
            db_reference["id"]
            for db_reference in results.get("uniProtKBCrossReferences", [])
            if db_reference["database"] == "PDB"
        ]
        return pdb_ids

    @staticmethod
    def get_alphafold_structure(entry_id: str) -> str:
        """
        Retrieve the AlphaFold structure URL for a given UniProt entry ID.

        Parameters
        ----------
        entry_id : str
            The UniProt entry ID.

        Returns
        -------
        str
            The AlphaFold structure URL.
        """
        base_url = f"https://alphafold.ebi.ac.uk/api/prediction/{entry_id}"

        response = requests.get(base_url)
        response.raise_for_status()

        results = response.json()
        if results:
            pdb_url = results[0].get("pdbUrl")
            if pdb_url:
                return pdb_url
            else:
                raise Exception(
                    f"No PDB URL found in AlphaFold results for entry ID: {entry_id}"
                )
        else:
            raise Exception(f"No AlphaFold structure found for entry ID: {entry_id}")

    @staticmethod
    def get_pdbe_structure(pdb_id: str) -> str:
        """
        Retrieve the PDBe structure URL for a given PDB ID.

        Parameters
        ----------
        pdb_id : str
            The PDB ID.

        Returns
        -------
        str
            The PDBe structure URL.
        """
        base_url = f"https://www.ebi.ac.uk/pdbe-srv/view/entry/{pdb_id}"
        response = requests.get(base_url)
        response.raise_for_status()

        if response.status_code == 200:
            return base_url
        else:
            raise Exception(f"No PDBe structure found for PDB ID: {pdb_id}")

    def download_structure(self, download_url: str, protein_name: str) -> Path:
        """
        Download the structure file from the given URL and save it to the output directory.

        Parameters
        ----------
        download_url : str
            URL to download the structure file from.
        protein_name : str
            The name of the protein.

        Returns
        -------
        Path
            The path to the downloaded structure file.
        """
        response = requests.get(download_url)
        response.raise_for_status()

        filename = self.output_dir / f"{protein_name}.pdb"
        with open(filename, "wb") as file:
            file.write(response.content)
        return filename

    def _are_sequence_lengths_equal(self, file_path: Path, seq_length) -> bool:
        """
        Check if the sequence length of the protein ID matches the length of the structure.

        Parameters
        ----------
        file_path : Path
            The path to the structure file.
        seq_length : int
            The length of the protein sequence.

        Returns
        -------
        bool
            True if the sequence lengths match, False otherwise.
        """
        # Call uniprot API

        structure = ampal.load_pdb(str(file_path))
        assembly = select_first_ampal_assembly(structure)
        # These conditions are necessary because uniprot is a mess :)
        try:
            return len(assembly.sequence) == seq_length
        except AttributeError:
            return False

    def process_protein(
        self, protein_id: str, check_len_during_reload: bool = False
    ) -> None:
        """
        Process a single protein ID to download its structure.

        Parameters
        ----------
        protein_id : str
            The protein ID to process.
        """
        # Check if the protein pdb already exists:
        file_path = self.output_dir / f"{protein_id}.pdb"
        if file_path.exists():
            if check_len_during_reload:
                uniprot_results = self.get_uniprot_entry(protein_id)
                # Check that the length of the sequence is equal to the length of the structure
                # Necessary as some uniprot structures are not the same length as their sequence. Great Success AlphaFold!
                if not self._are_sequence_lengths_equal(file_path, uniprot_results.sequence_length):
                    # Delete the file if the sequence length does not match
                    file_path.unlink()
                    self.error_list.add(protein_id)
            return
        
        # If json path exists but no PDB, no need to call uniprot - just fail
        json_path = self.output_dir / f"{protein_id}.json"
        if json_path.exists():
            return
        
        attempts = 0
        success = False
        while attempts < 1 and not success:
            try:
                # Check if the protein ID is already in the uniprot_to_accession_dict
                if protein_id in self.uniprot_to_accession_dict:
                    entry_id = self.uniprot_to_accession_dict[protein_id]
                    length = None
                else:
                    uniprot_results = self.get_uniprot_entry(protein_id)
                    entry_id = uniprot_results.uniprot_id
                    length = uniprot_results.sequence_length
                # Get the structure from the alphafold server using the entry ID
                try:
                    structure_url = self.get_alphafold_structure(entry_id)
                # If the structure is not available, get the structure from the PDBe server
                except Exception:
                    pdb_ids = self.get_pdb_ids_from_uniprot(entry_id)
                    if pdb_ids:
                        structure_url = self.get_pdbe_structure(pdb_ids[0])
                    else:
                        if self.verbose:
                            print(f"No PDB IDs found for entry ID: {entry_id}")
                        break

                file_path = self.download_structure(structure_url, protein_id)
                if length:
                    if not self._are_sequence_lengths_equal(file_path, length):
                        # Delete the file if the sequence length does not match
                        file_path.unlink()
                        if self.verbose:
                            print(
                                f"Sequence length mismatch for protein ID: {protein_id}"
                            )
                        break
                # Update the dictionaries
                self.uniprot_to_path_dict[protein_id] = file_path
                self.uniprot_to_accession_dict[protein_id] = entry_id
                success = True
            # Retry if an exception occurs up to 3 times
            except Exception as e:
                #attempts += 1
                break
            
        if not success:
            self.error_list.add(protein_id)

    def process(
        self, data_dict: t.Dict[str, str]
    ) -> t.Tuple[t.Dict[str, Path], t.Dict[str, str]]:
        """
        Execute the download process for the specified proteins.

        Parameters
        ----------
        data_dict : Dict[str, str]
            A dictionary containing protein IDs.

        Returns
        -------
        Tuple[Dict[str, Path], Dict[str, str]]
            Dictionary mapping protein IDs to the paths of the downloaded structure files.
            Dictionary mapping protein IDs to the UniProt accession numbers.
        """
        # Reset the dictionaries
        self._reset_class()
        # Check whether protein exists in the output directory
        proteins_to_download = set()
        for protein in data_dict.keys():
            path_to_pdb = self.output_dir / f"{protein}.pdb"
            # If the protein exists AND its entry is in the json file, we don't need to download it
            if path_to_pdb.exists():
                if protein in self.uniprot_to_accession_dict:
                    self.uniprot_to_path_dict[protein] = path_to_pdb
            else:
                proteins_to_download.add(protein)

        # Filter out error proteins
        if self.error_file.exists():
            with open(self.error_file, "r") as error_file:
                self.error_list.update(
                    protein.strip() for protein in error_file.readlines()
                )

        # Check that proteins_to_download are not in the error list
        proteins_to_download = proteins_to_download - self.error_list

        # Process proteins in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.process_protein, protein)
                for protein in proteins_to_download
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Uniprot Downloader:",
            ):
                future.result()  # Wait for all threads to complete

        # Write errors to error.txt file
        if self.error_list:
            with open(self.error_file, "w") as error_file:
                for protein in self.error_list:
                    error_file.write(f"{protein}\n")
        
            if self.verbose:
                print(
                    f"Errors occurred for the following proteins {self.error_list} and were written to error.txt"
                )

        # Check that the dictionary is not empty
        assert self.uniprot_to_path_dict, "No structures were downloaded"
        assert self.uniprot_to_accession_dict, "No uniprot accession IDs were saved"

        # Save the uniprot_to_accession_dict to a JSON file
        self._save_uniprot_to_accession_dict()

        return self.uniprot_to_path_dict, self.uniprot_to_accession_dict


class Ontology:
    """
    A class to represent Gene Ontology and perform various operations related to it.

    Attributes:
    -----------
    ont : Dict[str, Dict]
        A dictionary containing ontology terms and their properties.
    ancestors : Dict[str, Set[str]]
        A dictionary containing the ancestors of terms.
    """

    def __init__(
        self,
        obo_filepath: Path,
        with_rels: bool = False,
        url: str = "https://current.geneontology.org/ontology/go-basic.obo",
    ):
        """
        Initializes the Ontology object by loading the ontology from a file or cache.

        Parameters:
        -----------
        obo_filepath : Path
            The path to the ontology file.
        with_rels : bool
            Whether to include relationships in the ontology.
        url : str
            URL to download the ontology file if it doesn't exist locally.
        """
        self.ancestors = {}
        self.ont = self.load_ontology(obo_filepath, url, with_rels)
        self.go_terms = self.get_all_terms()
        self.num_classes = len(self.go_terms)
        self.label_encoder, self.index_to_term = create_label_encoder(
            self.get_all_terms()
        )
        self.ic = None
        self.icdepth = None
        self.calculate_ic([set(self.get_ancestors(x)) for x in self.go_terms])

    ### Ontology Loading and Parsing Functions

    def load_ontology(
        self, obo_filepath: Path, go_url: str, with_rels: bool
    ) -> Dict[str, Dict]:
        """
        Loads the ontology from a cache file or downloads it if not available.

        Parameters:
        -----------
        obo_filepath : Path
            The path to the ontology file.
        url : str
            The URL to download the file from.

        Returns:
        --------
        Dict[str, Dict]
            A dictionary containing ontology terms and their properties.
        """
        obo_filepath = Path(obo_filepath) / "go-basic.obo"
        cache_path = obo_filepath.with_suffix(".pkl")
        self.matrix_path = obo_filepath.with_suffix(".npz")

        if cache_path.exists():
            return self.load_from_cache(cache_path)

        if not obo_filepath.exists():
            self.download_ontology(obo_filepath, go_url)

        ontology = self.parse_ontology(obo_filepath, with_rels)
        self.save_to_cache(ontology, cache_path)

        return ontology

    @staticmethod
    def download_ontology(obo_filepath: Path, go_url: str):
        """
        Downloads the ontology file from the specified URL.

        Parameters:
        -----------
        obo_filepath : Path
            The path to save the downloaded file.
        url : str
            The URL to download the file from.
        """
        response = requests.get(go_url)
        response.raise_for_status()
        obo_filepath.write_text(response.text)

    @staticmethod
    def parse_ontology(obo_filepath: Path, with_rels: bool) -> Dict[str, Dict]:
        """
        Loads the ontology from a file.

        Parameters:
        -----------
        filepath : Path
            The path to the ontology file.

        Returns:
        --------
        Dict[str, Dict]
            A dictionary containing ontology terms and their properties.
        """
        assert obo_filepath.exists(), f"Ontology file {obo_filepath} does not exist"

        ont = dict()
        obj = None
        with open(obo_filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == "[Term]":
                    if obj is not None:
                        ont[obj["id"]] = obj
                    obj = dict()
                    obj["is_a"] = list()
                    obj["relationships"] = list()
                    obj["alt_ids"] = list()
                    obj["is_obsolete"] = False
                    continue
                elif line == "[Typedef]":
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == "id":
                        obj["id"] = l[1]
                    elif l[0] == "alt_id":
                        obj["alt_ids"].append(l[1])
                    elif l[0] == "namespace":
                        obj["namespace"] = l[1]
                    elif l[0] == "is_a":
                        obj["is_a"].append(l[1].split(" ! ")[0])
                    elif with_rels and l[0] == "relationship":
                        obj["relationships"].append(l[1])

                    elif l[0] == "name":
                        obj["name"] = l[1]
                    elif l[0] == "is_obsolete" and l[1] == "true":
                        obj["is_obsolete"] = True
        if obj is not None:
            ont[obj["id"]] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]["alt_ids"]:
                ont[t_id] = ont[term_id]
            if ont[term_id]["is_obsolete"]:
                del ont[term_id]
        for term_id, val in ont.items():
            if "children" not in val:
                val["children"] = set()
            for p_id in val["is_a"]:
                if p_id in ont:
                    if "children" not in ont[p_id]:
                        ont[p_id]["children"] = set()
                    ont[p_id]["children"].add(term_id)
        return ont

    ### Cache Management Functions

    @staticmethod
    def save_to_cache(ontology: Dict[str, Dict], cache_path: Path):
        """
        Saves the ontology to a cache file.

        Parameters:
        -----------
        ontology : Dict[str, Dict]
            The ontology dictionary to save.
        cache_path : Path
            The path to the cache file.
        """
        with cache_path.open("wb") as cache_file:
            pickle.dump(ontology, cache_file)

    @staticmethod
    def load_from_cache(cache_path: Path) -> Dict[str, Dict]:
        """
        Loads the ontology from a cache file.

        Parameters:
        -----------
        cache_path : Path
            The path to the cache file.

        Returns:
        --------
        Dict[str, Dict]
            A dictionary containing ontology terms and their properties.
        """
        with cache_path.open("rb") as cache_file:
            return pickle.load(cache_file)

    ### Ontology Operation Functions

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        self.icdepth = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])
            self.ic[go_id] = math.log(min_n / n, 2)
            self.icdepth[go_id] = (
                math.log(
                    self.get_depth(go_id, NAMESPACES_REVERT[self.get_namespace(go_id)]),
                    2,
                )
                * self.ic[go_id]
            )

    def get_ancestors(self, term_id: str) -> Set[str]:
        """
        Gets the ancestors of a term.

        Parameters:
        -----------
        term_id : str
            The ID of the term to get ancestors for.

        Returns:
        --------
        Set[str]
            A set of ancestor term IDs.
        """
        if term_id not in self.ont:
            return set()
        if term_id in self.ancestors:
            return self.ancestors[term_id]

        ancestors = set()
        queue = deque([term_id])
        while queue:
            current_id = queue.popleft()
            if current_id not in ancestors:
                ancestors.add(current_id)
                if "is_a" in self.ont.get(current_id, {}):
                    queue.extend(self.ont[current_id]["is_a"])

        self.ancestors[term_id] = ancestors
        return ancestors

    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]["is_a"]:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set

    def get_depth(self, term_id, ont):
        q = deque()
        q.append(term_id)
        layer = 1
        while len(q) > 0:
            all_p = set()
            while len(q) > 0:
                t_id = q.popleft()
                p_id = self.get_parents(t_id)
                all_p.update(p_id)
            if all_p:
                layer += 1
                for item in all_p:
                    if item == FUNC_DICT[ont]:
                        return layer
                    q.append(item)
        return layer

    def get_namespace(self, term_id: str) -> str:
        return self.ont[term_id]["namespace"]

    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj["namespace"] == namespace:
                terms.add(go_id)
        return terms

    def get_all_terms(self) -> List[str]:
        """
        Gets all terms in the ontology, sorted by their IDs.

        Returns:
        --------
        List[str]
            A list of all term IDs sorted alphabetically.
        """
        return sorted(self.ont.keys())

    def get_term_set(self, term_id: str) -> Set[str]:
        """
        Gets the set of all terms that are subclasses of the given term.

        Parameters
        ----------
        term_id: str
            The ID of the term to get subclasses for.

        Returns
        -------
        term_set: Set[str]
            A set of term IDs that are subclasses of the given term.
        """
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]["children"]:
                    q.append(ch_id)
        return term_set

    ### Child Matrix Creation and Management

    def create_child_matrix(self) -> torch.Tensor:
        """
        Creates a child matrix where CM[i][j] = 1 if the j-th term is a subclass of the i-th term.
        If the matrix already exists at the given path, it loads it instead of recomputing.

        Returns:
        --------
        torch.Tensor
            A tensor representing the child matrix.
        """
        if self.matrix_path.exists():
            return self.load_child_matrix(self.matrix_path)

        matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32)

        for term_id in self.go_terms:
            term_idx = self.label_encoder[term_id]
            # Only consider "is_a" relationships
            ancestors = self.get_ancestors_is_a(term_id)
            for ancestor in ancestors:
                if ancestor in self.label_encoder:
                    matrix[term_idx][self.label_encoder[ancestor]] = 1.0

        self.save_child_matrix(matrix, self.matrix_path)
        return matrix

    def get_ancestors_is_a(self, term_id: str) -> Set[str]:
        """
        Gets the ancestors of a term considering only "is_a" relationships.

        Parameters:
        -----------
        term_id : str
            The ID of the term to get ancestors for.

        Returns:
        --------
        Set[str]
            A set of ancestor term IDs considering only "is_a" relationships.
        """
        if term_id not in self.ont:
            return set()
        ancestors = set()
        queue = deque([term_id])
        while queue:
            current_id = queue.popleft()
            if current_id not in ancestors:
                ancestors.add(current_id)
                if "is_a" in self.ont.get(current_id, {}):
                    queue.extend(self.ont[current_id]["is_a"])
        return ancestors

    @staticmethod
    def save_child_matrix(child_matrix: torch.Tensor, filepath: Path):
        """
        Saves the child matrix to a .npz file.

        Parameters:
        -----------
        child_matrix : torch.Tensor
            The child matrix tensor to save.
        filepath : Path
            The path to save the .npz file.
        """
        ssp.save_npz(filepath, ssp.csr_matrix(child_matrix.numpy()))

    @staticmethod
    def load_child_matrix(filepath: Path) -> torch.Tensor:
        """
        Loads the child matrix from a .npz file.

        Parameters:
        -----------
        filepath : Path
            The path to the .npz file.

        Returns:
        --------
        torch.Tensor
            The loaded child matrix tensor.
        """
        return torch.tensor(ssp.load_npz(filepath).toarray(), dtype=torch.float32)

    ### Information Content (IC) Management

    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception("Not yet calculated")
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def get_icdepth(self, go_id):
        if self.icdepth is None:
            raise Exception("Not yet calculated")
        if go_id not in self.icdepth:
            return 0.0
        return self.icdepth[go_id]

    ### Utility Functions

    def has_term(self, term_id: str) -> bool:
        """
        Checks if a term exists in the ontology.

        Parameters:
        -----------
        term_id : str
            The ID of the term to check.

        Returns:
        --------
        bool
            True if the term exists, False otherwise.
        """
        return term_id in self.ont


class PrositeToUniprot:
    """Searches uniprot entries for prosite patterns."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.prosite_query_output = self.output_dir / "prosite_query_output"
        if not self.prosite_query_output.exists():
            self.prosite_query_output.mkdir(parents=True, exist_ok=True)

    def load_cache(self, prosite_id: str) -> List[str]:
        """Load cache for a specific Prosite ID from a JSON file."""
        cache_file = self.prosite_query_output / f"{prosite_id}.json"
        if cache_file.exists():
            with open(cache_file, "r") as file:
                return json.load(file)
        return []

    def save_cache(self, prosite_id: str, data: List[str]) -> None:
        """Save cache for a specific Prosite ID to a JSON file."""
        cache_file = self.prosite_query_output / f"{prosite_id}.json"
        with open(cache_file, "w") as file:
            json.dump(data, file)

    def prosite_to_uniprot(self, prosite_id: str) -> List[str]:
        """
        Retrieve the UniProt entry IDs for a given Prosite ID.

        Retrieves entries that were reviewed in Uniprot.

        Parameters
        ----------
        prosite_id : str
            The Prosite ID to query (e.g., "PS51325").

        Returns
        -------
        List[str]
            A list of UniProt entry IDs associated with the Prosite ID.
        """
        # Load cached data if available
        cached_result = self.load_cache(prosite_id)
        if cached_result:
            return cached_result

        # If not cached, perform the request

        url = f"https://prosite.expasy.org/rule/{prosite_id}"
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        soup = BeautifulSoup(response.text, "html.parser")

        # Locate the specific table row containing the "All" category link
        all_link = soup.find("td", string="All").find_next("a")

        # Extract the URL from the 'href' attribute
        uniprot_url = all_link["href"]
        # Extract query parameters from the URL
        query_str = uniprot_url.split("?query=")[1]

        base_url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": query_str,
            "format": "json",
            "fields": "accession,id",
            "size": 100,
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()

        results = response.json().get("results", [])
        uniprot_ids = (
            [entry["primaryAccession"] for entry in results] if results else []
        )

        # Save the result to the cache
        self.save_cache(prosite_id, uniprot_ids)

        return uniprot_ids

    def prosite_list_to_uniprot(self, prosite_list: List[str]) -> Dict[str, List[str]]:
        """
        Retrieve the UniProt entry IDs for a list of Prosite IDs.

        Parameters
        ----------
        prosite_list : List[str]
            A list of Prosite IDs to query.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary mapping Prosite IDs to lists of UniProt entry IDs.
        """
        prosite_to_uniprot_dict = {}

        # Add TQDM progress bar for the loop
        for prosite_id in tqdm(prosite_list, desc="Processing Prosite IDs"):
            uniprot_ids = self.prosite_to_uniprot(prosite_id)
            prosite_to_uniprot_dict[prosite_id] = uniprot_ids

        return prosite_to_uniprot_dict

    def process(
        self, go_to_prosite_rule: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Process the Prosite Rule IDs for the given GO terms.

        Parameters
        ----------
        go_to_prosite_rule: Dict[str, Dict[str, str]]
            Dictionary containing GO terms and their corresponding Prosite IDs.

        Returns
        -------
        go_to_prosite_uniprot: Dict[str, Dict[str, List[str]]]
            Dictionary containing GO terms, the Prosite ID and their corresponding UniProt IDs.
        """
        go_to_prosite_uniprot = {}

        # Iterate over each GO term and its associated Prosite dictionary
        for go_term, prosite_dict in go_to_prosite_rule.items():
            # Create a set of PRU codes for the current GO term
            pru_set = set(prosite_dict["prosite"].keys())

            # if self.unique_only:
            #     # Initialize an empty set to hold the union of PRU sets from other GO terms
            #     other_pru_sets = set()
            #
            #     # Iterate over all GO terms and their PRU sets
            #     for k, other_prosite_dict in go_to_prosite_rule.items():
            #         # Skip the current GO term to only include PRU sets from other GO terms
            #         if k != go_term:
            #             # Add the PRU codes from this GO term to the union set
            #             other_pru_sets.update(set(other_prosite_dict["prosite"].keys()))
            #
            #     # Remove PRU codes that are present in other GO terms (retain only unique PRU codes)
            #     pru_set -= other_pru_sets

            # If there are any PRU codes left after filtering, process them
            if pru_set:
                uniprot_dict = self.prosite_list_to_uniprot(list(pru_set))
                go_to_prosite_uniprot[go_term] = uniprot_dict

        return go_to_prosite_uniprot


def create_label_encoder(
    go_terms: List[str],
) -> t.Tuple[Dict[str, int], Dict[int, str]]:
    """
    Creates a label encoder for the given list of GO terms and its inverse mapping.

    Parameters:
    -----------
    go_terms : List[str]
        A list of GO terms.

    Returns:
    --------
    Tuple[Dict[str, int], Dict[int, str]]
        A tuple containing:
        - A dictionary mapping GO terms to unique integer indices.
        - A dictionary mapping integer indices to GO terms.
    """
    label_encoder = {term: idx for idx, term in enumerate(go_terms)}
    index_to_term = {idx: term for term, idx in label_encoder.items()}
    return label_encoder, index_to_term


def encode_single_protein(
    protein_id: str,
    go_terms: t.Set[str],
    label_encoder: t.Dict[str, int],
    num_classes: int,
) -> t.Tuple[str, torch.Tensor]:
    """
    Encodes a single protein's GO terms into a sparse one-hot vector.

    Parameters:
    -----------
    protein_id : str
        The protein identifier.
    go_terms : t.Set[str]
        The set of GO terms associated with the protein.
    label_encoder : t.Dict[str, int]
        A dictionary mapping GO terms to unique integer indices.
    num_classes : int
        The total number of unique GO terms.

    Returns:
    --------
    t.Tuple[str, torch.Tensor]
        A tuple containing the protein identifier and its one-hot encoded label sparse tensor.
    """
    indices = []
    values = []
    for term in go_terms:
        if term in label_encoder:
            indices.append([0, label_encoder[term]])
            values.append(1.0)  # Keep values as float for compatibility

    indices = torch.LongTensor(indices).t()
    values = torch.FloatTensor(values)
    one_hot_vector = torch.sparse_coo_tensor(
        indices, values, torch.Size([1, num_classes]), dtype=torch.float
    )
    return protein_id, one_hot_vector.coalesce()


def one_hot_encode_go_labels(
    label_encoder: t.Dict[str, int],
    protein_dict: t.Dict[str, t.Set[str]],
    num_classes: int,
    num_workers: int = 4,
) -> t.Tuple[t.List[str], torch.Tensor]:
    """
    One-hot encodes the labels for the given protein dictionary using sparse tensors.
    """
    with mp.Pool(processes=num_workers) as pool:
        results = [
            pool.apply_async(
                encode_single_protein,
                args=(protein_id, go_terms, label_encoder, num_classes),
            )
            for protein_id, go_terms in protein_dict.items()
        ]
        ordered_proteins = []
        one_hot_vectors = []
        for res in results:
            protein_id, one_hot_vector = res.get()
            ordered_proteins.append(protein_id)
            one_hot_vectors.append(one_hot_vector)

    # Combine sparse tensors into a single sparse tensor
    one_hot_labels = torch.cat(one_hot_vectors, dim=0)
    return ordered_proteins, one_hot_labels.coalesce()


def download_uniprot_files(
    uniprot_downloader: UniprotDownloader, data_dict: t.Dict
) -> t.Tuple[t.Dict, t.Dict]:
    """
    Download the Uniprot files for the given data dictionary.

    Parameters
    ----------
    uniprot_downloader: UniprotDownloader
        Uniprot downloader object.
    data_dict: t.Dict
        Dictionary containing the data split ["train", "validation", "test"].

    Returns
    -------
    uniprot_to_path_dict: t.Dict
        Dictionary containing the Uniprot ID to the path of the downloaded file for each data split ["train", "validation", "test"].
    uniprot_to_accession_dict: t.Dict
        Dictionary containing the Uniprot ID to the accession number for each data split ["train", "validation", "test"].
    """
    uniprot_to_path_dict, uniprot_to_accession_dict = {}, {}
    for data_split in data_dict.keys():
        (
            uniprot_to_path_dict[data_split],
            uniprot_to_accession_dict[data_split],
        ) = uniprot_downloader.process(data_dict[data_split]["labels"])

    return uniprot_to_path_dict, uniprot_to_accession_dict


if __name__ == "__main__":
    # Check if prosite to uniprot works
    # prosite_to_uniprot = PrositeToUniprot(Path("data/"))
    # go_to_uniprot = prosite_to_uniprot.process(go_to_prosite)
    # print(f"Prosite ID: {go_to_uniprot}")
    # Check if Ontology works
    ontology = Ontology(Path("data/"))
    child_matrix = ontology.create_child_matrix()
    # TODO: Move to tests
    # Verify if GO:0000430 is a subclass of GO:0000429 and GO:0046015
    term_index = {term: idx for idx, term in enumerate(ontology.get_all_terms())}
    go_0000430_idx = term_index.get("GO:0000430")
    go_0000429_idx = term_index.get("GO:0000429")
    go_0046015_idx = term_index.get("GO:0046015")

    if (
        go_0000430_idx is not None
        and go_0000429_idx is not None
        and go_0046015_idx is not None
    ):
        is_subclass_1 = child_matrix[go_0000430_idx][go_0000429_idx].item() == 1.0
        is_subclass_2 = child_matrix[go_0000430_idx][go_0046015_idx].item() == 1.0

        print(f"GO:0000430 is a subclass of GO:0000429: {is_subclass_1}")
        print(f"GO:0000430 is a subclass of GO:0046015: {is_subclass_2}")
    else:
        raise ValueError("One or more terms not found in the ontology")

    all_go_terms = ontology.get_all_terms()
    label_encoder = {term: idx for idx, term in enumerate(all_go_terms)}
    num_classes = len(all_go_terms)
    ontology.get_ancestors('GO:0005829')
    # Example protein dictionary
    protein_dict = {
        "5MP1_HUMAN": {
            "GO:0009889",
            "GO:0050794",
            "GO:0031326",
            "GO:2000112",
            "GO:0031323",
            "GO:0008150",
            "GO:0060255",
            "GO:0051246",
            "GO:0034248",
            "GO:0050789",
            "GO:0005622",
            "GO:0010468",
            "GO:0080090",
            "GO:0006446",
            "GO:0065007",
            "GO:0005575",
            "GO:0051171",
            "GO:0006417",
            "GO:0019222",
            "GO:0010608",
            "GO:0010556",
            "GO:0110165",
            "GO:0005737",
        },
        "5MP2_HUMAN": {
            "GO:0009889",
            "GO:0050794",
            "GO:0031326",
            "GO:2000112",
            "GO:0031323",
            "GO:0008150",
            "GO:0060255",
            "GO:0051246",
            "GO:0034248",
            "GO:0050789",
        },  # Add more proteins as needed
    }

    # One-hot encode the labels using multiprocessing
    ordered_proteins, one_hot_labels = one_hot_encode_go_labels(
        label_encoder, protein_dict, num_classes
    )

    # Print the ordered proteins and the shape of the one-hot encoded labels tensor for verification
    print(f"Ordered Proteins: {ordered_proteins}")
    print(f"One-hot labels tensor shape: {one_hot_labels.shape}")

    # Verification for 5MP1_HUMAN
    expected_go_terms = {
        "GO:0009889",
        "GO:0050794",
        "GO:0031326",
        "GO:2000112",
        "GO:0031323",
        "GO:0008150",
        "GO:0060255",
        "GO:0051246",
        "GO:0034248",
        "GO:0050789",
        "GO:0005622",
        "GO:0010468",
        "GO:0080090",
        "GO:0006446",
        "GO:0065007",
        "GO:0005575",
        "GO:0051171",
        "GO:0006417",
        "GO:0019222",
        "GO:0010608",
        "GO:0010556",
        "GO:0110165",
        "GO:0005737",
    }
    expected_indices = {
        label_encoder[term] for term in expected_go_terms if term in label_encoder
    }

    idx_5MP1_HUMAN = ordered_proteins.index("5MP1_HUMAN")
    one_hot_vector_5MP1_HUMAN = one_hot_labels[idx_5MP1_HUMAN]

    # Get indices where the value is 1
    actual_indices = set(
        torch.nonzero(one_hot_vector_5MP1_HUMAN, as_tuple=True)[0].tolist()
    )

    assert (
        actual_indices == expected_indices
    ), f"Indices mismatch: expected {expected_indices}, got {actual_indices}"

    print("5MP1_HUMAN has all the expected GO terms correctly one-hot encoded.")
