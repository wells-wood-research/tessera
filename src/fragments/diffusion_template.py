"""
This script generates a PDB file with fragments at specific positions to be used by RFDiffusion.
"""
import argparse
import typing as t
from pathlib import Path

import ampal
import numpy as np
from scipy.spatial.transform import Rotation

from src.difference_fn.difference_processing import select_first_ampal_assembly
from src.fragments.fragments_graph import StructureFragmentGraphIO
from src.visualization.fold_coverage import load_graph_creator


def convert_pdb_to_fg(
    pdb_path: Path,
    fragment_path: Path,
    output_path: Path,
    verbose: bool = False,
    workers: int = 1,
) -> Path:
    """
    Convert a PDB file to a FG file.

    Parameters
    ----------
    pdb_path: Path
        Path to the PDB file
    fragment_path: Path
        Path to the fragment directory
    output_path: Path
        Path to the output directory
    verbose: bool
        Verbosity
    workers: int
        Number of workers

    Returns
    -------
    Path
        Path to the output FG file

    """
    graph_creator = load_graph_creator(
        fragment_path=fragment_path,
        workers=workers,
        pdb_path=output_path,
        verbose=verbose,
    )
    fg_path = graph_creator.classify_and_save_graph(pdb_path)
    return fg_path


def find_rotation_translation_matrix(
    fragment: ampal.Polypeptide, original: ampal.Polypeptide
):
    """
    Computes the optimal rotation and translation to align `fragment` to `original`
    using `scipy.spatial.transform.Rotation.align_vectors`.

    Parameters
    ----------
    fragment : AMPAL Object
        The fragment to be aligned.
    original : AMPAL Object
        The reference fragment.

    Returns
    -------
    angle_degrees : float
        Rotation angle in degrees.
    axis : numpy.ndarray
        Rotation axis as a normalized 3D vector.
    centroid_A : numpy.ndarray
        Centroid of the original fragment.
    centroid_B : numpy.ndarray
        Centroid of the fragment to be aligned.

    Notes
    -----
    The alignment is computed using the `align_vectors` method, which finds the rotation
    that best aligns two sets of vectors in a least-squares sense.

    Steps:
    1. **Extract Coordinates**:
       Obtain the coordinates of atoms from both `fragment` and `original`.
    2. **Compute Centroids**:
       Compute the centroids of both sets of points.
    3. **Center Coordinates**:
       Center the coordinates by subtracting their respective centroids.
    4. **Compute Optimal Rotation**:
       Use `align_vectors` to compute the rotation that aligns `fragment` to `original`.
    5. **Extract Rotation Angle and Axis**:
       Convert the rotation object to angle-axis representation.

    References
    ----------
    - [scipy.spatial.transform.Rotation.align_vectors](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html)
    """
    assert len(fragment) == len(
        original
    ), "Fragment and original must have the same number of amino acids"
    fragment_atoms = fragment.backbone.get_atoms()
    original_atoms = original.backbone.get_atoms()

    # Step 1: Extract coordinates
    points_A = np.array([atom._vector for atom in original_atoms])
    points_B = np.array([atom._vector for atom in fragment_atoms])

    # Step 2: Compute centroids
    centroid_A = np.mean(points_A, axis=0)
    centroid_B = np.mean(points_B, axis=0)

    # Step 3: Center coordinates
    AA = points_A - centroid_A
    BB = points_B - centroid_B

    # Step 4: Compute optimal rotation
    rotation, rmsd = Rotation.align_vectors(AA, BB)
    print("RMSD:", rmsd)
    # Step 5: Extract rotation angle and axis
    rotvec = rotation.as_rotvec()
    angle_rad = np.linalg.norm(rotvec)
    if angle_rad != 0:
        axis = rotvec / angle_rad
    else:
        axis = np.array([1, 0, 0])  # Default axis
    angle_degrees = np.degrees(angle_rad)

    return angle_degrees, axis, centroid_A, centroid_B


def align_peptides(
    fragment: ampal.Polypeptide,
    original: ampal.Polypeptide,
    rmsd_threshold: float,
    maintain_backbone: bool,
    verbose: bool = True,
) -> t.Optional[ampal.Polypeptide]:
    """
    Aligns fragment to original using the computed rotation and translation.
    Returns the aligned fragment if the RMSD is below the threshold, otherwise None.
    """
    if maintain_backbone:
        return original
    else:
        # Step 1: Compute alignment parameters
        angle_degrees, axis, centroid_A, centroid_B = find_rotation_translation_matrix(
            fragment, original
        )
        original_rmsd = original.backbone.rmsd(fragment.backbone)
        # Step 2: Translate fragment to origin
        fragment.translate(-centroid_B)
        # Step 3: Rotate fragment around the origin
        fragment.rotate(angle=angle_degrees, axis=axis, radians=False)
        # Step 4: Translate fragment to the original's centroid
        fragment.translate(centroid_A)
        # Step 5: Compute RMSD after alignment
        current_rmsd = original.backbone.rmsd(fragment.backbone)
        if verbose:
            print(f"Original RMSD: {original_rmsd:.3f}")
            print(f"Aligned RMSD: {current_rmsd:.3f}")
        # Step 6: Check if RMSD is below the threshold
        if current_rmsd < rmsd_threshold:
            return fragment
        else:
            if verbose:
                print(
                    f"Fragment rejected. RMSD {current_rmsd:.3f} exceeds threshold {rmsd_threshold}."
                )
            return None


def merge_fragments_into_assembly(
    fragment_list: t.List[ampal.Polypeptide],
) -> ampal.Polypeptide:
    """
    Merge a list of fragments into a single polypeptide.

    Parameters
    ----------
    fragment_list : List[AMPAL Object]
        List of fragments to be merged.

    Returns
    -------
    merged_fragment : AMPAL Object
        Merged polypeptide.
    """
    merged_fragment = ampal.Assembly()

    for fragment in fragment_list:
        merged_fragment.append(fragment)
    merged_fragment.relabel_all()
    return merged_fragment


def process_fragment(
    fragment: t.Any,
    original_pdb: ampal.Polypeptide,
    fragment_start_idx: int,
    fragment_end_idx: int,
    args: t.Any,
    fragments_pdb: t.List[ampal.Polypeptide],
) -> t.Tuple[int, bool]:
    """
    Handles fragment slicing, alignment, and updates fragment list if aligned successfully.
    Returns the length of the aligned fragment or None if alignment fails.
    """
    sliced_original_pdb = original_pdb[fragment_start_idx : fragment_end_idx + 1]
    curr_fragment_path = args.fragment_path / str(fragment.fragment_class)
    # fragment_paths = sorted(curr_fragment_path.rglob("*.pdb1"))

    # Load the first available fragment PDB
    # fragment_pdb = select_first_ampal_assembly(ampal.load_pdb(fragment_paths[0]))

    # Align fragment and check against RMSD threshold
    aligned_fragment_pdb = align_peptides(
        sliced_original_pdb,
        sliced_original_pdb,
        rmsd_threshold=args.rmsd_threshold,
        maintain_backbone=args.maintain_backbone,
        verbose=args.verbose,
    )

    if aligned_fragment_pdb:  # If fragment passes RMSD threshold
        fragments_pdb.append(aligned_fragment_pdb)
        return len(aligned_fragment_pdb), True
    else:  # Fragment rejected due to RMSD
        if args.verbose:
            print(f"Fragment {fragment.fragment_class} rejected due to high RMSD.")
        return len(sliced_original_pdb), False


def handle_unknown_fragments(
    curr_unknown_count: int, rf_diffusion_text: str
) -> t.Tuple[str, int]:
    # Manages RF Diffusion text for unknown fragments
    if curr_unknown_count > 0:
        rf_diffusion_text += (
            f"{max(1, curr_unknown_count - 5)}-{curr_unknown_count + 5}/"
        )
        curr_unknown_count = 0
    return rf_diffusion_text, curr_unknown_count


def main(args):
    assert args.input_path.exists(), f"Input file {args.input_path} does not exist"
    assert (
        args.fragment_path.exists()
    ), f"Fragment path {args.fragment_path} does not exist"
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Check input path extension
    if args.input_path.suffix in {".pdb", ".pdb1"}:
        fg_path = args.output_path / f"{args.input_path.stem}.fg"
        if fg_path.exists():
            if args.verbose:
                print(f"FG file {fg_path} already exists")
        else:
            fg_path = convert_pdb_to_fg(
                pdb_path=args.input_path,
                fragment_path=args.fragment_path,
                output_path=args.output_path,
                verbose=args.verbose,
                workers=args.workers,
            )
    else:
        raise NotImplementedError(
            f"File extension {args.input_path.suffix} not supported"
        )

    original_pdb = select_first_ampal_assembly(ampal.load_pdb(args.input_path))
    fragment_graph = StructureFragmentGraphIO.load(fg_path)

    assert len(fragment_graph.structure_fragment.classification) == len(
        original_pdb
    ), "Fragment classification does not match the length of the original pdb"

    fragments_pdb = []
    rf_diffusion_text = "contigmap.contigs=["
    curr_unknown_count = 0

    for fragment in fragment_graph.structure_fragment.classification_map:
        if fragment.fragment_class == 0:
            curr_unknown_count += fragment.end_idx - fragment.start_idx + 1
        else:
            fragment_length, aln_status = process_fragment(
                fragment,
                original_pdb,
                fragment.start_idx,
                fragment.end_idx,
                args,
                fragments_pdb,
            )

            if aln_status:  # Only add contig if fragment was aligned successfully
                rf_diffusion_text, curr_unknown_count = handle_unknown_fragments(curr_unknown_count, rf_diffusion_text)
                chain_letter = chr(65 + len(fragments_pdb) - 1)
                rf_diffusion_text += f"{chain_letter}1-{fragment_length}/"
            else:
                curr_unknown_count += fragment_length

    rf_diffusion_text, _ = handle_unknown_fragments(
        curr_unknown_count, rf_diffusion_text
    )

    # If final character is / remove it
    if rf_diffusion_text[-1] == "/":
        rf_diffusion_text = rf_diffusion_text[:-1]
    rf_diffusion_text += "]\n"

    # Merge fragments into a single assembly
    merged_fragment = merge_fragments_into_assembly(fragments_pdb)
    output_pdb_path = args.output_path / f"{args.input_path.stem}_fragments.pdb"

    # Save to PDB
    with output_pdb_path.open("w") as f:
        f.write(merged_fragment.pdb)

    rf_diffusion_text = (
        f"python scripts/run_inference.py inference.output_prefix=fragment_designs/{output_pdb_path.stem}_design inference.input_pdb={output_pdb_path} inference.num_designs=10 "
        + rf_diffusion_text
    )

    # If rfdiff_generate_structures.sh exists, append to it, otherwise create it
    rfdiff_script_path = args.output_path / "rfdiff_generate_structures.sh"
    if rfdiff_script_path.exists():
        with rfdiff_script_path.open("a") as f:
            f.write(rf_diffusion_text)
    else:
        with rfdiff_script_path.open("w") as f:
            f.write(rf_diffusion_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input_path", type=Path, required=True, help="Path to input file"
    )
    parser.add_argument(
        "--output_path", type=Path, required=True, help="Path to output dir"
    )
    parser.add_argument(
        "--fragment_path", type=Path, required=True, help="Path to fragment dir"
    )
    parser.add_argument("--workers", type=int, help="Number of workers", default=10)
    parser.add_argument("--verbose", action="store_true", help="Print details")
    parser.add_argument(
        "--maintain_backbone",
        action="store_true",
        help="Rather than using fragments, use exact backbone atoms from the original structure",
    )
    # Add optional threshold for rmsd
    parser.add_argument(
        "--rmsd_threshold",
        type=float,
        default=3.0,
        help="RMSD threshold for fragment alignment, if RMSD is greater than this value, the fragment will not be used.",
    )
    params = parser.parse_args()
    main(params)
