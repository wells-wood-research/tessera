import typing as t
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from src.difference_fn.angle_difference import AngleDifferenceStrategy
from src.difference_fn.difference_processing import StructureConvolutionOperator
from src.difference_fn.probability_processing import (
    FragmentDetail,
    ProbabilityProcessor,
)
from src.fragments.fragments_graph import StructureFragmentGraph
from src.fragments.reference_fragments import ReferenceFragmentCreator


class StructureFragment:
    def __init__(
        self,
        structure_path: Path,
        classification: np.ndarray,
        classification_map: t.List[FragmentDetail],
        probability_distance_data: np.ndarray,
        raw_distance_data: np.ndarray,
        angles: np.ndarray,
        backbone_coords: np.ndarray,
    ) -> None:
        if not structure_path.exists():
            raise FileNotFoundError(f"{structure_path} does not exist")
        self.structure_path = structure_path
        self.classification = classification
        self.classification_map = classification_map
        self.raw_distance_data = raw_distance_data
        self.probability_distance_data = probability_distance_data
        self.angles = angles
        self.backbone_coords = backbone_coords

    def __repr__(self):
        return f"{self.__class__.__name__}(structure_path={self.structure_path}, classification={self.classification})"

    def save_probability_to_csv(
        self, csv_save_path: Path, prefix: str = "", delete_existing: bool = False
    ):
        """
        Save the raw and probability distance data_paths to a CSV file.

        Args:
        - csv_save_path (Path): Path to save the CSV file.
        - prefix (str, optional): Prefix to add to the CSV file name. Default is "".
        - delete_existing (bool, optional): If True, delete existing files with the same name. Default is False.
        """
        structure_name = self.structure_path.stem
        raw_distance_path = (
            csv_save_path / f"{prefix}_{structure_name}_raw_distance_data.csv"
        )
        probability_distance_path = (
            csv_save_path / f"{prefix}_{structure_name}_probability_distance_data.csv"
        )

        # Check if files exist and force is False
        if not delete_existing:
            if raw_distance_path.exists() or probability_distance_path.exists():
                raise FileExistsError(
                    f"File(s) already exist: {raw_distance_path}, {probability_distance_path}"
                )

        np.savetxt(
            raw_distance_path,
            self.raw_distance_data,
            delimiter=",",
        )
        np.savetxt(
            probability_distance_path,
            self.probability_distance_data,
            delimiter=",",
        )

    def to_graph(
        self,
        edge_distance_threshold: float = 12,
        extra_graph_attributes: t.Dict[str, t.Any] = {},
    ) -> "StructureFragmentGraph":
        """
        Convert the StructureFragment to a StructureFragmentGraph.

        Parameters
        ----------
        edge_distance_threshold: float
            The threshold distance for edges in the graph_dataset.
        extra_graph_attributes: Dict[str, Any]
            Extra attributes to add to the graph_dataset.

        Returns
        -------
        StructureFragmentGraph: The StructureFragmentGraph object.

        """
        return StructureFragmentGraph.from_structure_fragment(
            self, edge_distance_threshold, extra_graph_attributes
        )


class StructureToFragmentClassifier:
    def __init__(
        self,
        fragment_path: Path,
        difference_name: str = "logpr",
        n_processes: int = 1,
        step_size: int = 1,
        fix_convolution_edges: bool = True,
        classification_metric: str = "max",
        classification_threshold_type: str = "optimal",
        classification_allowed_overlap: int = 2,
        max_sequence_length: int = 3000,
    ) -> None:
        # TODO below should be handled by a factory
        self.convolution_operator = self._create_convolution_operator(
            difference_name,
            step_size,
            fix_convolution_edges,
            max_sequence_length,
        )
        self.probability_processor = self._create_probability_processor(
            classification_metric,
            classification_threshold_type,
            classification_allowed_overlap,
        )
        self.fragment_dict = self._create_fragment_dict(
            fragment_path, self.convolution_operator.difference_fn
        )
        self.n_processes = n_processes

    @staticmethod
    def _create_convolution_operator(
        difference_name: str,
        step_size: int,
        fix_convolution_edges: bool,
        max_sequence_length: int,
    ):
        return StructureConvolutionOperator(
            difference_name,
            step_size,
            fix_convolution_edges,
            max_sequence_length,
        )

    @staticmethod
    def _create_probability_processor(
        metric: str, threshold_type: str, allowed_overlap: int
    ):
        return ProbabilityProcessor(metric, threshold_type, allowed_overlap)

    @staticmethod
    def _create_fragment_dict(fragment_path: Path, difference_fn: Callable):
        return ReferenceFragmentCreator(
            folder_path=fragment_path, difference_fn=difference_fn
        ).create_all_fragments()

    def classify_to_fragment(
        self, structure_path: Path, use_all_fragments: bool = False
    ) -> StructureFragment:
        fragment_identifiers = sorted(self.fragment_dict.keys())

        # Load structure data:
        structure = self.convolution_operator.load_and_prepare_structure(structure_path)
        structure_data = self.convolution_operator.difference_fn.get_ampal_data(
            structure
        )

        (
            structure_fragment_tuple_list,
            fragments_paths_map,
        ) = self._generate_fragment_data_map(
            structure_data, fragment_identifiers, use_all_fragments
        )

        results_array = self._convolve_all_fragments(structure_fragment_tuple_list)

        mean_results_array = self.calculate_mean_per_fragment(
            fragments_paths_map, results_array
        )
        assert not np.isnan(
            mean_results_array
        ).any(), "NaN values found in mean results array. This is either indicates a bug or that the difference function is broken."
        mean_results_array_transposed = mean_results_array.T

        normalised_difference = (
            self.convolution_operator.difference_fn.normalise_difference(
                mean_results_array_transposed
            )
        )
        assert not np.isnan(
            normalised_difference
        ).any(), "NaN values found in normalised results array. This is either indicates a bug or that the difference function is broken."

        probability_distribution = (
            self.probability_processor.convert_signal_to_probability(
                normalised_difference
            )
        )
        assert not np.isnan(
            probability_distribution
        ).any(), "NaN values found in probability_distribution array. This is either indicates a bug or that the difference function is broken."
        best_fragments, fragment_detail_list = self.probability_processor.classify(
            probability_distribution
        )

        # For "angle" type, the structure_data already contains the angles.
        if self.convolution_operator.difference_fn.data_type == "angle":
            angles = structure_data
        else:
            # You must ensure that AngleDifferenceStrategy.get_angles is implemented.
            angles = AngleDifferenceStrategy.get_angles(structure)
        backbone_coords = np.array(structure.get_reference_coords())

        return StructureFragment(
            structure_path=structure_path,
            raw_distance_data=mean_results_array_transposed,
            probability_distance_data=probability_distribution,
            classification=best_fragments,
            classification_map=fragment_detail_list,
            angles=angles,
            backbone_coords=backbone_coords,
        )

    def _generate_fragment_data_map(
        self,
        structure_data: np.ndarray,
        fragment_identifiers: List[int],
        use_all_fragments: bool,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int]]:
        """
        Generates a mapping between a given structure and fragments using their data arrays.

        Args:
            structure_data (np.ndarray): Numpy array representing the structure.
            fragment_identifiers (List[int]): List of fragment numbers to consider.
            use_all_fragments (bool): If True, use all fragment variants; otherwise, use only the first variant.

        Returns:
            Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int]]:
                - List of tuples (structure_data, fragment_data) for comparison.
                - Corresponding list of fragment identifiers.
        """
        structure_fragment_tuple_list = []
        fragments_data_map = []

        for fragment_number in fragment_identifiers:
            # Ensure fragment_data is at least 3D
            fragment_data = np.atleast_3d(self.fragment_dict[fragment_number].data)
            # Select all variants or only the first based on the flag
            frag_variants = fragment_data if use_all_fragments else fragment_data[:1]
            # For each variant, reintroduce a batch dimension and create the tuple
            tuples = [(structure_data, v[None, ...]) for v in frag_variants]
            structure_fragment_tuple_list.extend(tuples)

            fragments_data_map.extend([fragment_number] * frag_variants.shape[0])

        return structure_fragment_tuple_list, fragments_data_map

    def _convolve_all_fragments(
        self, structure_fragment_tuple_list: t.List[t.Tuple[str, str]]
    ) -> np.ndarray:

        if self.n_processes > 1:
            chunk_size = max(
                1, len(structure_fragment_tuple_list) // self.n_processes
            )  # Prevent excessive context switching
            with Pool(self.n_processes) as pool:
                results = pool.starmap(
                    self.convolution_operator.convolve,
                    structure_fragment_tuple_list,
                    chunksize=chunk_size,
                )
        else:
            results = [
                self.convolution_operator.convolve(*pair)
                for pair in structure_fragment_tuple_list
            ]

        return np.vstack(results)

    @staticmethod
    def calculate_mean_per_fragment(
        fragments_data_map: List[int], results_array: np.ndarray
    ) -> np.ndarray:
        """
        Computes the mean of rows in results_array grouped by fragment number.

        Args:
            fragments_data_map (List[int]): A list mapping each row in results_array to a fragment number.
            results_array (np.ndarray): Array of shape (n_variants_total, n_positions) with the convolution results.

        Returns:
            np.ndarray: Array of shape (n_unique_fragments, n_positions) where each row
                        is the mean over all instances of that fragment.
        """

        # Convert fragment mappings to a NumPy array
        frag_ids = np.array(fragments_data_map)

        # Identify unique fragment identifiers and get inverse indices
        # unique_fragments: array of unique fragment identifiers
        # inv_indices: maps each fragment instance to its index in unique_fragments
        unique_fragments, inv_indices = np.unique(frag_ids, return_inverse=True)

        # Preallocate arrays to store sum and count of values per fragment
        sum_array = np.zeros(
            (len(unique_fragments), results_array.shape[1]), dtype=np.float64
        )
        count_array = np.zeros(len(unique_fragments), dtype=np.int32)

        # Efficiently sum values per fragment using NumPy's add.at function
        # This accumulates values at the correct indices in sum_array
        np.add.at(sum_array, inv_indices, results_array)

        # Count occurrences of each fragment (how many instances contribute to each fragment's mean)
        np.add.at(count_array, inv_indices, 1)

        # Compute the mean for each fragment
        return sum_array / count_array[:, None]  # Broadcasting ensures division per row


class EnsembleFragmentClassifier:
    def __init__(
        self,
        fragment_path: Path,
        difference_names: t.List[str] = ["logpr", "ramrmsd"],
        n_processes: int = 1,
        step_size: int = 1,
        fix_convolution_edges: bool = True,
        classification_metric: str = "max",
        classification_threshold_type: str = "optimal",
        classification_allowed_overlap: int = 2,
    ) -> None:
        self.classifiers = [
            StructureToFragmentClassifier(
                fragment_path,
                difference_name,
                n_processes,
                step_size,
                fix_convolution_edges,
                classification_metric,
                classification_threshold_type,
                classification_allowed_overlap,
            )
            for difference_name in difference_names
        ]

    def classify_to_fragment(
        self, structure_path: Path, use_all_fragments: bool = False
    ) -> StructureFragment:
        # Run classification with each classifier
        results = [
            classifier.classify_to_fragment(structure_path, use_all_fragments)
            for classifier in self.classifiers
        ]

        # Combine the probability distributions and determine final classification
        return self._combine_classifications(results)

    def _combine_classifications(
        self, results: t.List[StructureFragment]
    ) -> StructureFragment:
        """
        Combines classifications from multiple classifiers using mean probability distribution.

        Args:
            results: List[StructureFragment]): List of classification results.

        Returns:
            StructureFragment: The ensemble classification result.
        """
        mean_probabilities = np.mean(
            [result.probability_distance_data for result in results], axis=0
        )

        # Use the first classifier's probability processor to classify
        base_classifier = self.classifiers[0]
        (
            best_fragments,
            fragment_detail_list,
        ) = base_classifier.probability_processor.classify(mean_probabilities)

        return StructureFragment(
            structure_path=results[0].structure_path,
            raw_distance_data=mean_probabilities,
            probability_distance_data=mean_probabilities,
            classification=best_fragments,
            classification_map=fragment_detail_list,
            angles=results[0].angles,
            backbone_coords=results[0].backbone_coords,
        )


if __name__ == "__main__":
    fragment_path = "../../data/fragments/"


    structure_path = "../../data/pdbs/1AGJ.pdb1"
    #
    # # Example usage
    assert Path(fragment_path).exists(), "Fragment all_pdb_paths does not exist"
    assert Path(structure_path).exists(), "Structure all_pdb_paths does not exist"
    classifier = EnsembleFragmentClassifier(
        Path(fragment_path),
        difference_types=["sequence", "angle"],
        difference_names=["BLOSUM", "RamaRmsd"],
        n_processes=1,
        step_size=1,
    )
    import time

    time_start = time.time()
    structure_fragment = classifier.classify_to_fragment(
        Path(structure_path), use_all_fragments=False
    )
    time_end = time.time()
    print(f"Time taken: {time_end - time_start}")
    print(structure_fragment.classification_map)

    time_start = time.time()
    # Convert to graph_dataset:
    structure_fragment_graph = StructureFragmentGraph.from_structure_fragment(
        structure_fragment,
        edge_distance_threshold=10,
    )
    time_end = time.time()
    print(f"Time taken: {time_end - time_start}")
    # Plot the graph_dataset:

    raise ValueError
