import numpy as np
import pytest
import random
from src.difference_fn.sequence_difference import (
    BlosumDifferenceStrategy,
    SeqIdentityDifferenceStrategy,
)


@pytest.fixture(autouse=True)
def set_random_seed():
    random.seed(42)
    np.random.seed(42)


def create_mock_ampal(sequence):
    class MockResidue:
        def __init__(self, amino_acid: str):
            self.sequence = amino_acid

    class MockPolypeptide:
        def __init__(self, residues):
            self.residues = residues
            self.sequence = "".join([residue.sequence for residue in residues])

        def get_monomers(self, ligands=False):
            return self.residues

    residues = [MockResidue(aa) for aa in sequence]
    return MockPolypeptide(residues)


@pytest.fixture(
    params=[
        SeqIdentityDifferenceStrategy(),
        BlosumDifferenceStrategy(),
    ],
    ids=[
        "SequenceDifferenceStrategy",
        "BlosumDifferenceStrategy",
    ],
)
def strategy_fixture(request):
    return request.param


@pytest.mark.parametrize(
    "test_type",
    [
        "identical_structures",
        "similar_structures",
        "different_structures",
        "theoretical_bounds",
        "maximum_difference",
        "variable_size_test",
    ],
)
def test_strategy_behaviors(strategy_fixture, test_type):
    # Setup common test data_paths and mocks
    n = 5
    seq_identical = "I"*n + "D" *n + "S" * n
    seq_different = "S"*n + "L" * n + "W" * n
    seq_similar = "V" * n + "E" * n + "T" * n
    mock_ampal_identical = create_mock_ampal(seq_identical)
    mock_ampal_different = create_mock_ampal(seq_different)
    mock_ampal_similar = create_mock_ampal(seq_similar)

    if test_type == "identical_structures":
        difference = strategy_fixture.calculate_difference(mock_ampal_identical, mock_ampal_identical)
        assert_identical_structures_assertion(strategy_fixture, difference)
    elif test_type == "similar_structures" and strategy_fixture.__class__.__name__ == "BlosumDifferenceStrategy":
        difference = strategy_fixture.calculate_difference(
            mock_ampal_identical, mock_ampal_similar
        )
        assert_identical_structures_assertion(strategy_fixture, difference)
    elif test_type == "different_structures":
        difference = strategy_fixture.calculate_difference(
            mock_ampal_identical, mock_ampal_different
        )
        assert_different_structures_assertion(strategy_fixture, difference)


def assert_identical_structures_assertion(strategy, difference):
    assert np.isclose(
        difference, strategy.theoretical_min, atol=1e-2
    ), f"{strategy.__class__.__name__} failed for identical structures with difference {difference}, expected {strategy.theoretical_min}."


def assert_different_structures_assertion(strategy, difference):
    assert np.isclose(
        difference, strategy.theoretical_max, atol=1e-2
    ), f"{strategy.__class__.__name__} failed for different structures with difference {difference}, expected {strategy.theoretical_max}."



