import argparse
import typing as t
from pathlib import Path

import ampal
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tessera.difference_fn.difference_processing import select_first_ampal_assembly
from tessera.difference_fn.shape_difference import RmsdBiopythonStrategy


def sort_categories(categories: t.List[str]) -> t.List[str]:
    """Sort categories: first those without a `+`, then those with a `+`."""
    without_plus = sorted([cat for cat in categories if "+" not in cat])
    with_plus = sorted([cat for cat in categories if "+" in cat])
    return without_plus + with_plus


def main(args):
    assert (
        args.original_pdb_path.exists()
    ), f"Input file {args.input_path} does not exist"
    assert (
        args.design_pdb_path.exists()
    ), f"Input file {args.design_pdb_path} does not exist"
    results_path = args.design_pdb_path / "rmsd_results.csv"
    if results_path.exists():
        results_df = pd.read_csv(results_path)
    else:
        # List all pdb files in the design_pdb_path
        designs_pdbs = list(args.design_pdb_path.glob("*.pdb"))
        # Sort the pdb files by name
        designs_pdbs.sort()
        results = []
        # Iterate over the pdb files to calculate the RMSD
        for curr_design in designs_pdbs:
            # Extract name of the design
            curr_design_name, curr_category, _, _, _ = curr_design.stem.split("_")
            # Original PDB path
            original_pdb_path = (
                args.original_pdb_path / f"{curr_design_name}_{curr_category}.pdb1"
            )
            # Check if the original pdb path exists
            assert (
                original_pdb_path.exists()
            ), f"Original PDB file {original_pdb_path} does not exist"
            # Load PDB to Ampal
            original_pdb = ampal.load_pdb(original_pdb_path)
            original_pdb = select_first_ampal_assembly(original_pdb)
            # Load design PDB to Ampal
            design_pdb = ampal.load_pdb(curr_design)
            design_pdb = select_first_ampal_assembly(design_pdb)
            # Calculate RMSD
            rmsd = RmsdBiopythonStrategy.biopython_calculate_rmsd(
                original_pdb, design_pdb
            )
            results.append(
                {
                    "original_pdb": original_pdb_path,
                    "design_pdb": curr_design,
                    "rmsd": rmsd,
                    "category": curr_category,
                }
            )
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        # Save results to CSV
        results_df.to_csv(results_path, index=False)

    # Plot RMSD by category with standard deviation  # Plot RMSD by category using violin plot with Seaborn
    sorted_categories = sort_categories(results_df["category"].unique())
    results_df["category"] = pd.Categorical(
        results_df["category"], categories=sorted_categories, ordered=True
    )

    # Plot RMSD by category using violin plot with Seaborn
    plt.figure(figsize=(14, 7))

    sns.violinplot(
        x="category", y="rmsd", data=results_df, inner="quartile", palette="Set2"
    )

    # Add horizontal line at 3 Ångströms
    plt.axhline(y=3, color="red", linestyle="--", linewidth=2, label="3 Å")

    # Customize labels and title
    plt.xlabel("Category", fontsize=16)
    plt.ylabel("RMSD (Å)", fontsize=16)
    plt.title("RMSD by Category", fontsize=16)

    # Set y-axis to start at 0
    plt.ylim(0, 15)

    # Rotate category labels to 90 degrees
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=16)

    # Show legend
    plt.legend(fontsize=16)

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the plot as a PDF inside the design_pdb_path folder
    plot_output_path = args.design_pdb_path / "rmsd_design.pdf"
    plt.savefig(plot_output_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--original_pdb_path", type=Path, help="Path to input file", required=True
    )
    parser.add_argument(
        "--design_pdb_path", type=Path, help="Path to input file", required=True
    )
    params = parser.parse_args()
    main(params)
