import argparse
import requests
import json
import hashlib
import os
from typing import List


def sanitize_filename(s: str) -> str:
    """
    Sanitizes a string to be safe for use as a filename.
    Replaces unsafe characters with underscores.
    """
    return "".join(c if c.isalnum() or c in (" ", ".", "_") else "_" for c in s).strip()


def fetch_pdb_by_go_terms(
    go_codes: List[str],
    go_codes_to_exclude: List[str] = None,
    save_query: bool = False,
    output_dir: str = ".",
) -> List[str]:
    """
    Fetches a list of PDB IDs that are associated with all the provided Gene Ontology (GO) codes,
    and excludes any that are associated with the GO codes to exclude. Before making the API call,
    it checks if the results for the given query already exist.

    Parameters:
    go_codes (List[str]): A list of GO codes to include (e.g., "GO:0003677" for DNA binding).
    go_codes_to_exclude (List[str], optional): A list of GO codes to exclude (e.g., "GO:0003723" for RNA binding).
    save_query (bool): If True, saves the query JSON to a file.
    output_dir (str): Directory to save the query JSON and cache files.

    Returns:
    List[str]: A list of PDB IDs that match all the inclusion GO codes and none of the exclusion GO codes.
    """

    base_url = "https://search.rcsb.org/rcsbsearch/v2/query?json"

    # Build the main query
    main_query = {"type": "group", "logical_operator": "and", "nodes": []}

    # Build inclusion group if go_codes is not empty
    if go_codes:
        inclusion_group = {"type": "group", "logical_operator": "and", "nodes": []}
        for go_code in go_codes:
            inclusion_group["nodes"].append(
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "operator": "exact_match",
                        "value": go_code,
                        "attribute": "rcsb_polymer_entity_annotation.annotation_id",
                    },
                }
            )
        main_query["nodes"].append(inclusion_group)

    # Build exclusion group if go_codes_to_exclude is not empty
    if go_codes_to_exclude:
        exclusion_group = {"type": "group", "logical_operator": "or", "nodes": []}
        for go_code in go_codes_to_exclude:
            exclusion_group["nodes"].append(
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "operator": "exact_match",
                        "value": go_code,
                        "attribute": "rcsb_polymer_entity_annotation.annotation_id",
                    },
                }
            )
        # Wrap the exclusion group in a NOT group
        not_group = {
            "type": "group",
            "logical_operator": "not",
            "nodes": [exclusion_group],
        }
        main_query["nodes"].append(not_group)

    # Complete the query
    query = {
        "query": main_query,
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }

    # Create a hash of the query for caching
    query_str = json.dumps(query, sort_keys=True)
    query_hash = hashlib.md5(query_str.encode("utf-8")).hexdigest()

    # Generate a filename based on GO codes
    go_codes_str = "_".join(go_codes)
    exclude_codes_str = "_".join(go_codes_to_exclude) if go_codes_to_exclude else ""
    filename_base = f"include_{go_codes_str}_exclude_{exclude_codes_str}"
    filename_base = sanitize_filename(filename_base)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Paths for query and cache files
    query_file = os.path.join(output_dir, f"{filename_base}_query.json")
    cache_file = os.path.join(output_dir, f"{filename_base}_cache_{query_hash}.json")

    # Save the query to a JSON file if requested
    if save_query:
        with open(query_file, "w") as f:
            json.dump(query, f, indent=2)
        print(f"Query saved to {query_file}")

    # Check if results are cached
    if os.path.exists(cache_file):
        print(f"Loading results from cache: {cache_file}")
        with open(cache_file, "r") as f:
            pdb_ids = json.load(f)
        return pdb_ids
    # Send the request
    response = requests.post(base_url, json=query)
    if response.status_code != 200:
        print("API Response:", response.text)
        raise Exception(
            f"Failed to query RCSB API with status code {response.status_code}"
        )
    # Extract PDB IDs
    pdb_ids = set()
    for result in response.json().get("result_set", []):
        pdb_id = result["identifier"]
        pdb_ids.add(pdb_id)
    pdb_ids = list(pdb_ids)
    # Save results to cache
    with open(cache_file, "w") as f:
        json.dump(pdb_ids, f)
    print(f"Results saved to cache: {cache_file}")

    return pdb_ids


def main():
    parser = argparse.ArgumentParser(description="Fetch PDB IDs by GO codes")
    parser.add_argument(
        "--go_codes",
        nargs="+",
        required=True,
        help="List of GO codes to include (e.g., GO:0003677)",
    )
    parser.add_argument(
        "--exclude_codes",
        nargs="*",
        help="List of GO codes to exclude (e.g., GO:0003723)",
    )
    parser.add_argument(
        "--save_query", action="store_true", help="Save the query to a JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the query JSON and cache files",
    )

    args = parser.parse_args()

    go_codes = args.go_codes
    exclude_codes = args.exclude_codes if args.exclude_codes else []

    pdb_ids = fetch_pdb_by_go_terms(
        go_codes,
        go_codes_to_exclude=exclude_codes,
        save_query=args.save_query,
        output_dir=args.output_dir,
    )

    print(f"PDB IDs matching GO codes {go_codes} and excluding {exclude_codes}:")
    print(pdb_ids)


if __name__ == "__main__":
    main()
