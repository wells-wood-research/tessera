import unittest
import pytest
import numpy as np
import random

from src.difference_fn.probability_processing import (
    ProbabilityProcessor,
)


@pytest.fixture(autouse=True)
def set_random_seed():
    random.seed(42)
    np.random.seed(42)


# Mock ReferenceFragment for testing purposes
class MockPDBFragment:
    def __init__(self, fragment_length, position_means, position_stds):
        self.fragment_length = fragment_length
        self.position_means = position_means
        self.position_stds = position_stds


@pytest.fixture
def valid_processor():
    """Fixture to provide a ProbabilityProcessor instance with valid settings."""
    return ProbabilityProcessor("max", "optimal", 1)


@pytest.mark.usefixtures("valid_processor")
class TestProbabilityProcessor:
    def test_initialization(self):
        """Test correct initialization and invalid threshold_distance handling."""
        # Valid initialization
        processor = ProbabilityProcessor("max", "optimal", 1)
        assert processor.metric == "max"
        assert processor.threshold_type == "optimal"
        assert processor.allowed_overlap == 1

        # Invalid threshold_distance
        with pytest.raises(AssertionError):
            ProbabilityProcessor("max", "invalid", 1)

    def test_convert_signal_to_probability(self, valid_processor):
        """Test conversion and classification of signals."""
        fragment_dict = {
            1: MockPDBFragment(3, np.random.rand(3), np.random.rand(3)),
            2: MockPDBFragment(5, np.random.rand(5), np.random.rand(5)),
            3: MockPDBFragment(8, np.random.rand(8), np.random.rand(8)),
        }
        n_positions = 15
        mean_array_transposed = np.random.rand(n_positions, len(fragment_dict.keys()))

        probabilities = valid_processor.convert_signal_to_probability(
            mean_array_transposed
        )

        assert isinstance(
            probabilities, np.ndarray
        ), "Probabilities should be a numpy array"
        assert probabilities.shape == mean_array_transposed.shape
        assert (
            0 <= probabilities.min() and probabilities.max() <= 1
        ), "Probabilities out of range [0, 1]"
        # Check the shape of probabilities to be the same as the input signal
        assert (
            probabilities.shape == mean_array_transposed.shape
        ), "Shape mismatch in probabilities and input signal"

    def test_classification(self, valid_processor):
        # Create a set of raw probabilities
        raw_probabilities = np.zeros((80, 40))
        # Assign high probabilities to the first and last fragments
        raw_probabilities[30, 2] = 1
        raw_probabilities[61, 16] = 1

        assigned_fragments, fragment_details = valid_processor.classify(
            raw_probabilities
        )
        # Check that positions 30-15 and 30+15 are assigned to fragment 3
        for i in range(15):
            assert (
                assigned_fragments[30 - i] == 3
            ), f"Failed to classify fragment 3 at position {30 - i}"
            assert (
                assigned_fragments[30 + i] == 3
            ), f"Failed to classify fragment 3 at position {30 + i}"

        # Check that positions spanning 9 are assigned to fragment 17
        for i in range(4):
            assert (
                assigned_fragments[61 - i] == 17
            ), f"Failed to classify fragment 17 at position {61 - i}"
            assert (
                assigned_fragments[61 + i] == 17
            ), f"Failed to classify fragment 17 at position {61 + i}"

        # Test for unknown regions classification
        # Assuming any position not covered by the high probabilities should be marked as unknown (0)
        all_assigned_positions = list(range(30 - 16, 30 + 16)) + list(
            range(61 - 5, 61 + 5)
        )
        all_other_positions = [i for i in range(80) if i not in all_assigned_positions]

        for pos in all_other_positions:
            assert (
                assigned_fragments[pos] == 0
            ), f"Position {pos} should be classified as unknown (0), but got {assigned_fragments[pos]}"

        previous_start = fragment_details[0].start_idx
        previous_end = fragment_details[0].end_idx
        for frag in fragment_details[1:]:
            assert frag.start_idx >= 0, "Fragment start should be greater than or equal to 0"
            assert frag.end_idx >= 0, "Fragment end should be greater than or equal to 0"
            assert frag.end_idx > frag.start_idx, "Fragment start should be less than fragment end"
            assert frag.start_idx > previous_end, "Fragments should not overlap"
            assert previous_start < frag.start_idx, "Fragments should be sorted"
            if frag.fragment_class > 0:
                assert frag.end_idx - frag.start_idx + 1 == valid_processor.fragment_lengths[frag.fragment_class], "Fragment length mismatch"
            previous_start = frag.start_idx
            previous_end = frag.end_idx
