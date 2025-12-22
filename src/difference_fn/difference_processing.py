import gzip
import typing as t
from pathlib import Path

import ampal
import numpy as np
import numpy.lib.stride_tricks as stride_tricks

from src.difference_fn.difference_selector import difference_function_selector


def select_first_ampal_assembly(
    ampal_structure: t.Union[
        ampal.AmpalContainer,
        ampal.Assembly,
        ampal.Polypeptide,
    ]
) -> ampal.Polypeptide:
    """
    This is a placeholder function to select the first ampal assembly. Ideally it should be removed as the library should handle
    different chains.

    Parameters
    ----------
    ampal_structure: t.Union[ampal.AmpalContainer, ampal.Assembly, ampal.Polypeptide]
        The ampal structure to select the first assembly from
    """
    if isinstance(ampal_structure, ampal.AmpalContainer):
        ampal_structure = ampal_structure[0]
    if isinstance(ampal_structure, ampal.Assembly):
        # Select the first polypeptide
        for p in ampal_structure:
            if isinstance(p, ampal.Polypeptide):
                ampal_structure = p
                ampal_structure.relabel_all()
                return ampal_structure
    # Relabelling ensures that the residue ids are unique and sequential
    ampal_structure.relabel_all()

    return ampal_structure


def load_fragment_pdb(folder_path: Path) -> t.Dict[str, Path]:
    pdb_to_paths = {}
    for file in folder_path.rglob("*.pdb1"):
        fragment_number = (
            file.parent.name
        )  # Assuming the folder name represents the fragment number
        pdb_to_paths[fragment_number] = file
        break
    return pdb_to_paths


def get_residue_ids(protein_structure: ampal.Polypeptide) -> t.List[str]:
    protein_structure.relabel_all()
    return [residue.id for residue in protein_structure.get_monomers()]


class StructureConvolutionOperator:
    def __init__(
        self,
        difference_name: str,
        step_size: int = 1,
        fix_convolution_edges: bool = True,
        max_sequence_length: int = 1000,
    ) -> None:

        self.difference_fn = difference_function_selector(
            difference_name=difference_name,
        )
        self.step_size = step_size
        self.fix_convolution_edges = fix_convolution_edges
        self.max_sequence_length = max_sequence_length

    @staticmethod
    def _load_structure(structure_pdb_path: Path) -> ampal.Polypeptide:
        valid_extensions = {".pdb", ".pdb1", ".pdb.gz", ".pdb1.gz"}
        if not any(str(structure_pdb_path).endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Invalid file extension: {structure_pdb_path}. Expected one of {valid_extensions}"
            )

        if str(structure_pdb_path).endswith(".gz"):
            with gzip.open(str(structure_pdb_path), "rb") as inf:
                structure_ampal = ampal.load_pdb(inf.read().decode(), path=False)
        else:
            structure_ampal = ampal.load_pdb(structure_pdb_path)

        return structure_ampal

    def load_and_prepare_structure(self, structure_pdb_path: Path) -> ampal.Polypeptide:
        structure_ampal = self._load_structure(structure_pdb_path)

        structure_ampal = select_first_ampal_assembly(structure_ampal)
        if not isinstance(structure_ampal, ampal.Polypeptide):
            raise ValueError("Reference structure is not a polypeptide")

        if len(structure_ampal.sequence) > self.max_sequence_length:
            raise ValueError(
                f"Reference structure sequence length {len(structure_ampal.sequence)} exceeds the maximum sequence length {self.max_sequence_length}"
            )

        if self.difference_fn.data_type == "angle":
            structure_ampal.tag_torsion_angles()

        return structure_ampal

    def convolve(
        self, reference_data: np.ndarray, fragment_data: np.ndarray
    ) -> np.ndarray:
        if self.difference_fn.data_type == "angle":
            return self.convolve_angles(reference_data, fragment_data)
        elif self.difference_fn.data_type == "sequence":
            return self.convolve_sequence(reference_data, fragment_data)
        else:
            raise ValueError(f"Invalid data type: {self.difference_fn.data_type}")

    def convolve_angles(
        self, reference_angles: np.ndarray, fragment_angles: np.ndarray
    ) -> np.ndarray:
        """
        Convolves reference angles (structure_length, 2) against fragments
        (n_fragments, frag_length, 2) and returns the difference score.
        """
        reference_length, angle_dim = reference_angles.shape
        # Ensure fragment_angles is at least 3D: (n_fragments, frag_length, 2)
        fragment_angles = np.atleast_3d(fragment_angles)
        n_fragments, fragment_length, _ = fragment_angles.shape

        self._check_convolution_lengths(
            fragment_length, reference_length, self.step_size
        )

        # Create sliding windows over reference_angles.
        W = reference_length - fragment_length + 1  # number of windows
        stride_shape = (W, fragment_length, angle_dim)
        stride = (reference_angles.strides[0],) * 2 + (reference_angles.strides[1],)
        reference_windows = stride_tricks.as_strided(
            reference_angles, shape=stride_shape, strides=stride
        )
        # reference_windows has shape (W, fragment_length, 2)

        # Tile the reference windows so that each fragment is compared with every window.
        # ref_tiled will have shape (n_fragments, W, fragment_length, 2)
        ref_tiled = np.tile(reference_windows, (n_fragments, 1, 1, 1))
        # Repeat fragment_angles along the window axis to match: shape (n_fragments, W, fragment_length, 2)
        frag_tiled = np.repeat(fragment_angles[:, None, ...], W, axis=1)

        # Flatten the first two dimensions to create a batch of window-fragment pairs.
        ref_flat = ref_tiled.reshape(-1, fragment_length, 2)
        frag_flat = frag_tiled.reshape(-1, fragment_length, 2)

        # Compute differences for each pair.
        # The difference_fn.calculate_difference expects two arrays of shape (frag_length, 2) and returns a scalar difference.
        diffs_flat = self.difference_fn.calculate_difference(ref_flat, frag_flat)
        # Reshape the result back to (n_fragments, W)
        differences = diffs_flat.reshape(n_fragments, W)

        # Downsample differences using step_size along the window dimension.
        differences = differences[:, :: self.step_size]

        # Preallocate output array of shape (n_fragments, reference_length)
        distance_array = np.full(
            (n_fragments, reference_length), np.nan, dtype=np.float64
        )

        # Compute the mid positions for each valid window.
        mid_positions = np.arange(
            fragment_length // 2,
            reference_length - fragment_length + fragment_length // 2 + 1,
            self.step_size,
        )
        # Assign computed differences to the corresponding mid positions.
        distance_array[:, mid_positions] = differences

        if self.fix_convolution_edges:
            distance_array = np.apply_along_axis(
                self._fix_convolution_edges, axis=1, arr=distance_array
            )
        return distance_array

    def convolve_sequence(
        self, reference_sequence: np.ndarray, fragment_sequence: np.ndarray
    ) -> np.ndarray:
        """
        Convolve a reference sequence against a fragment sequence using a given difference function.

        Parameters:
        -----------
        reference_sequence : np.ndarray
            Reference sequence as a NumPy array.
        fragment_sequence : np.ndarray
            Fragment sequences as a NumPy array.
        difference_fn : callable
            Function that computes sequence similarity (e.g., BLOSUM or identity-based function).
        step_size : int
            Step size for sliding window.

        Returns:
        --------
        np.ndarray
            Array containing the computed difference scores at each step.
        """
        reference_length = len(reference_sequence)
        fragment_length = fragment_sequence.shape[1]

        # Ensure fragment is shorter than reference
        assert (
            fragment_length < reference_length
        ), "Fragment must be shorter than the reference sequence."

        # Compute the number of valid sliding windows
        num_windows = reference_length - fragment_length + 1

        # Create sliding windows using NumPy striding
        reference_windows = np.lib.stride_tricks.as_strided(
            reference_sequence,
            shape=(num_windows, fragment_length),
            strides=(reference_sequence.strides[0], reference_sequence.strides[0]),
        )

        # Compute differences for each window using the difference function
        differences = np.array(
            [
                self.difference_fn.calculate_difference(window, fragment_sequence)
                for window in reference_windows
            ]
        )

        # Ensure differences are a 1D array
        differences = np.squeeze(differences)

        # Downsample differences using step_size
        differences = differences[:: self.step_size]

        # Preallocate output array of shape (reference_length,)
        distance_array = np.full(reference_length, np.nan, dtype=np.float64)

        # Compute the mid positions for each valid window
        mid_positions = np.arange(
            fragment_length // 2,
            reference_length - fragment_length + fragment_length // 2 + 1,
            self.step_size,
        )

        # Assign computed differences to the corresponding mid positions
        distance_array[mid_positions] = differences

        if self.fix_convolution_edges:
            distance_array = np.apply_along_axis(
                self._fix_convolution_edges, axis=0, arr=distance_array
            )

        return distance_array

    @staticmethod
    def _check_convolution_lengths(
        fragment_length: int, reference_length: int, step_size: int
    ) -> None:
        """
        Check that the fragment length is smaller than the reference length and that the step size is valid.

        Parameters
        ----------
        fragment_length: int
            The length of the fragment
        reference_length: int
            The length of the reference structure
        step_size

        Returns
        -------

        """
        assert step_size > 0, "Step size must be greater than 0"
        assert (
            fragment_length < reference_length
        ), "Fragment length must be smaller than the reference length"
        assert (
            reference_length > 0
        ), "Reference structure must have at least one residue"
        assert fragment_length > 0, "Fragment structure must have at least one residue"
        assert (
            fragment_length > step_size
        ), "Fragment length must be greater than the step size"

    @staticmethod
    def _fix_convolution_edges(distance_array: np.ndarray) -> np.ndarray:
        valid_indices = np.where(~np.isnan(distance_array))[0]
        if valid_indices.size > 0:
            first_valid_index = valid_indices[0]
            last_valid_index = valid_indices[-1]

            distance_array[:first_valid_index] = distance_array[first_valid_index]
            distance_array[last_valid_index + 1 :] = distance_array[last_valid_index]

        return distance_array
