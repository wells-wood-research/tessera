import typing as t
from collections import namedtuple

from tessera.fragments.classification_config import (
    fragment_lengths,
    get_threshold,
    median_fragment_length,
    min_fragment_length,
)

import numpy as np
from scipy.stats import entropy

FragmentDetail = namedtuple(
    "FragmentDetail", ["fragment_class", "category", "start_idx", "end_idx"]
)


class ProbabilityProcessor:
    def __init__(
        self,
        metric: str,
        threshold_type: str,
        allowed_overlap: int,
    ) -> None:
        self.metric = metric
        self.threshold_type = threshold_type
        self.fragment_lengths = fragment_lengths
        self.allowed_overlap = allowed_overlap

        self.threshold = get_threshold(metric, threshold_type)
        if self.metric == "inverse_median":
            self.process_probability = self.probability_to_inverse_median
        elif self.metric == "max":
            self.process_probability = self.probability_to_max
        elif self.metric == "inverse_entropy":
            self.process_probability = self.probability_to_inverse_entropy

    def __repr__(self):
        return f"{self.__class__.__name__}(metric={self.metric}, threshold_type={self.threshold_type}, threshold={self.threshold})"

    @staticmethod
    def convert_signal_to_probability(signal: np.ndarray) -> np.ndarray:
        """
        Convert a signal to a pseudo probability distribution.

        Args:
            signal (np.ndarray): The input signal to convert (n_positions, n_fragments).
        Returns:
            np.ndarray: The probability distribution.
        """
        # Get similarity:
        similarity = 1 - signal
        # Normalize by position
        normalised_probability = similarity / similarity.sum(axis=1, keepdims=True)
        sums = normalised_probability.sum(axis=1)
        # Check if the sum of probabilities for each position is approximately 1
        assert np.allclose(
            sums, 1
        ), "The sum of probabilities for each position is not approximately 1."
        # Check if the probabilities are between 0 and 1
        assert np.all(
            (0 <= normalised_probability) & (normalised_probability <= 1)
        ), "The probabilities are not between 0 and 1."

        return normalised_probability

    @staticmethod
    def probability_to_inverse_median(probabilities: np.ndarray) -> np.ndarray:
        """
        Convert probabilities to inverse median values.

        Args:
            probabilities (np.ndarray): The input probabilities to convert (n_positions, n_fragments).
        Returns:
            np.ndarray: The inverse median values.
        """
        return 1 / np.median(probabilities, axis=1)

    @staticmethod
    def probability_to_max(probabilities: np.ndarray) -> np.ndarray:
        """
        Convert probabilities to max values.

        Args:
            probabilities (np.ndarray): The input probabilities to convert (n_positions, n_fragments).
        Returns:
            np.ndarray: The max values.
        """
        return np.max(probabilities, axis=1)

    @staticmethod
    def probability_to_inverse_entropy(probabilities: np.ndarray) -> np.ndarray:
        """
        Convert probabilities to inverse entropy values.

        Args:
            probabilities (np.ndarray): The input probabilities to convert (n_positions, n_fragments).
        Returns:
            np.ndarray: The inverse entropy values.
        """
        return 1 / entropy(probabilities, axis=1)

    def classify(self, raw_probabilities: np.ndarray) -> (np.ndarray, list):
        n_positions, num_classes = raw_probabilities.shape
        max_probs = np.max(raw_probabilities, axis=1)
        max_classes = np.argmax(raw_probabilities, axis=1)

        # Apply threshold_distance and sort positions by highest probability
        valid_indices = np.where(max_probs >= self.threshold)[0]
        sorted_indices = valid_indices[np.argsort(-max_probs[valid_indices])]

        assigned_fragments = np.zeros(n_positions, dtype=int)  # 0 indicates unassigned
        fragment_tuples_map = []  # List to store named tuple-based fragment details

        for pos in sorted_indices:
            frag_class = max_classes[pos]
            frag_length = self.fragment_lengths[frag_class + 1]
            start = max(0, pos - frag_length // 2)
            end = min(n_positions - 1, start + frag_length - 1)
            # Check for overlapping and ensure no overlap before assignment
            if not assigned_fragments[start : end + 1].any():
                assigned_fragments[start : end + 1] = frag_class + 1
                fragment_tuples_map.append(
                    FragmentDetail(frag_class + 1, "fragment", start, end)
                )

        # Process unknown regions
        zeros_fragments_tuples_map = self._process_unknown_regions(assigned_fragments)
        fragment_tuples_map.extend(zeros_fragments_tuples_map)

        sorted_fragment_details = sorted(fragment_tuples_map, key=lambda x: x.start_idx)

        return assigned_fragments, sorted_fragment_details

    def _process_unknown_regions(
        self, assigned_fragments: np.ndarray
    ) -> t.List[FragmentDetail]:
        mask = assigned_fragments == 0
        # Compute differences to detect boundaries
        diff = np.diff(mask.astype(int))
        start_indices = np.where(diff == 1)[0] + 1
        end_indices = np.where(diff == -1)[0]
        if mask[0]:
            start_indices = np.insert(start_indices, 0, 0)
        if mask[-1]:
            end_indices = np.append(end_indices, len(mask) - 1)

        fragment_details = []
        for start, end in zip(start_indices, end_indices):
            fragment_details.extend(self._process_zero_segment(start, end))
        return fragment_details

    @staticmethod
    def _process_zero_segment(start: int, end: int) -> t.List:
        """
        Process a contiguous zero segment from start to end.
        """
        segment_length = end - start + 1
        num_full_chunks = segment_length // median_fragment_length
        remainder = segment_length % median_fragment_length

        details = []
        # Precompute chunk boundaries with vectorized arange if desired.
        if num_full_chunks > 0:
            chunk_starts = np.arange(start, start + num_full_chunks * median_fragment_length, median_fragment_length)
            chunk_ends = chunk_starts + median_fragment_length - 1
            details.extend([FragmentDetail(0, "unknown fragment", s, e) for s, e in zip(chunk_starts, chunk_ends)])

        # Handle remainder
        remainder_start = start + num_full_chunks * median_fragment_length
        if remainder > 0:
            if remainder < min_fragment_length:
                details.append(FragmentDetail(0, "unknown connector", remainder_start, end))
            else:
                details.append(FragmentDetail(0, "unknown fragment", remainder_start, end))
        return details

