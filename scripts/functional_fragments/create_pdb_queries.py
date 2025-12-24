import os
import json
import requests
from itertools import combinations
from typing import Dict, List


def create_query(selected_go_codes: List[str], all_go_codes: List[str]) -> Dict:
    """
    Create a PDB search query for a unique combination of GO codes,
    excluding other GO codes to prevent dual functions.
    """
    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": []},
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": 10000},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
            "scoring_strategy": "combined",
            "group_by_return_type": "representatives",
            "group_by": {
                "aggregation_method": "sequence_identity",
                "ranking_criteria_type": {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc",
                },
                "similarity_cutoff": 30,
            },
        },
    }

    for go_code in selected_go_codes:
        go_group = {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_uniprot_annotation.annotation_lineage.id",
                        "operator": "exact_match",
                        "value": go_code,
                        "negation": False,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_uniprot_annotation.type",
                        "operator": "exact_match",
                        "value": "GO",
                        "negation": False,
                    },
                },
            ],
            "label": "nested-attribute",
        }
        query["query"]["nodes"].append(go_group)

    for negated_code in all_go_codes:
        if negated_code not in selected_go_codes:
            negation_group = {
                "type": "group",
                "logical_operator": "or",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_uniprot_annotation.annotation_lineage.id",
                            "operator": "exact_match",
                            "value": negated_code,
                            "negation": True,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_uniprot_annotation.type",
                            "operator": "exact_match",
                            "value": "GO",
                            "negation": True,
                        },
                    },
                ],
                "label": "nested-attribute",
            }
            query["query"]["nodes"].append(negation_group)

    return query


def save_query(
    query: Dict, selected_go_codes: List[str], directory: str = "query"
) -> str:
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = "+".join(sorted(selected_go_codes)) + ".json"
    file_path = os.path.join(directory, filename)
    with open(file_path, "w") as file:
        json.dump(query, file, indent=4)
    return file_path


def fetch_results(query: Dict) -> Dict:
    """
    Send the query to the RCSB search endpoint and retrieve results.
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query"  # Adjusted to the correct search endpoint
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=query)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def save_results(
    results: Dict, selected_go_codes: List[str], directory: str = "query_results"
) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = "+".join(sorted(selected_go_codes)) + "_results.json"
    file_path = os.path.join(directory, filename)
    with open(file_path, "w") as file:
        json.dump(results, file, indent=4)


def generate_and_save_combinations(go_codes: List[str]) -> None:
    """
    Generate and save unique PDB search queries for all single, pair, and triplet GO codes,
    execute the queries, and save the results.
    """
    for num in [1, 2, 3]:  # Include single GO codes
        for combination in combinations(go_codes, num):
            selected_go_codes = list(combination)

            # Step 1: Create and save the query file
            query = create_query(selected_go_codes, go_codes)
            save_query(query, selected_go_codes)

            # Step 2: Execute the query and save the results
            try:
                results = fetch_results(query)
                save_results(results, selected_go_codes)
            except Exception as e:
                print(f"Failed to fetch results for {selected_go_codes}: {e}")


# Example GO codes list to be used for generating queries
go_codes = ["GO:0003677", "GO:0003723", "GO:0005525", "GO:0046872", "GO:0005524"]

# Generate and save the queries and their results for single, pairs, and triplets
generate_and_save_combinations(go_codes)
