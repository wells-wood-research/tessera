import numpy as np
import pytest
from pathlib import Path
import random
from src.difference_fn.angle_difference import (
    CorrDifferenceStrategy,
    LogPrDifferenceStrategy,
    RamRmsdDifferenceStrategy,
    RmsDifferenceStrategy,
    SphereDifferenceStrategy,
    load_fragments_angles,
)


@pytest.fixture(autouse=True)
def set_random_seed():
    random.seed(42)
    np.random.seed(42)


def create_mock_ampal(angles):
    class MockResidue:
        def __init__(self, phi, psi):
            self.tags = {"phi": phi, "psi": psi}
            self._coordinates = np.array([0, 0, 0])

    class MockPolypeptide:
        def __init__(self, residues):
            self.residues = residues
            self.sequence = "A" * len(residues)

        def get_monomers(self, ligands=False):
            return self.residues

        def tag_torsion_angles(self):
            pass

    residues = [MockResidue(phi, psi) for phi, psi in angles]
    return MockPolypeptide(residues)


def generate_variable_angle_pairs():
    """
    Generates sets of angle pairs with variable sizes to test scenarios where
    the reference and fragment have a different number of angles.
    """
    angles = np.linspace(
        0, 180, num=360, endpoint=False
    )  # Generate angles from 0 to just under 2*pi
    # Create variable-sized sets of angle pairs
    variable_angle_sets = [
        [
            (np.random.choice(angles), np.random.choice(angles))
            for _ in range(5)
        ]
        for _ in range(30)  # Generate 10 sets with random sizes and angles
    ]
    return variable_angle_sets


def test_load_fragments_angles(tmp_path: Path):
    # Create a mock .npy file
    fragment_number = "1"
    folder = tmp_path / fragment_number
    folder.mkdir()
    angles = np.random.rand(10, 3)
    np.save(folder / "angles.npy", angles)

    # Call the function under test
    fragments_to_angles = load_fragments_angles(tmp_path)

    # Assert that the data_paths is loaded correctly
    assert fragment_number in fragments_to_angles
    loaded_angles = fragments_to_angles[fragment_number]
    assert np.allclose(
        loaded_angles, np.mean(angles, axis=0)
    ), "The loaded angles do not match the expected values."


@pytest.fixture(
    params=[
        RmsDifferenceStrategy(),
        SphereDifferenceStrategy(),
        RamRmsdDifferenceStrategy(),
        LogPrDifferenceStrategy(),
        CorrDifferenceStrategy(),
    ],
    ids=[
        "RmsDifference",
        "SphereDifference",
        "RamRmsdDifference",
        "LogPrDifference",
        "CorrDifference",
    ],
)
def strategy_fixture(request):
    return request.param


@pytest.mark.parametrize(
    "test_type",
    [
        "identical_structures",
        "theoretical_bounds",
        "maximum_difference",
        "variable_size_test",
    ],
)
def test_strategy_behaviors(strategy_fixture, test_type):
    # Setup common test data_paths and mocks
    angles_identical = [(-60, -45), (-45, -30), (60, 45)]
    angles_different = [(60, 45), (45, 30), (-60, -45)]
    mock_ampal_identical = create_mock_ampal(angles_identical)
    mock_ampal_different = create_mock_ampal(angles_different)

    if test_type == "identical_structures":
        difference = strategy_fixture.calculate_difference(
            mock_ampal_identical, mock_ampal_identical
        )
        assert_identical_structures_assertion(strategy_fixture, difference)
    elif test_type == "theoretical_bounds":
        difference = strategy_fixture.calculate_difference(
            mock_ampal_identical, mock_ampal_different
        )
        assert_theoretical_bounds_assertion(strategy_fixture, difference)
    elif test_type == "maximum_difference":
        angles_max_diff1 = [(1, 181)] * 10
        angles_max_diff2 = [(181, 361)] * 10
        mock_ampal_max_diff1 = create_mock_ampal(angles_max_diff1)
        mock_ampal_max_diff2 = create_mock_ampal(angles_max_diff2)
        difference = strategy_fixture.calculate_difference(
            mock_ampal_max_diff1, mock_ampal_max_diff2
        )
        assert_maximum_difference_assertion(strategy_fixture, difference)
    if test_type == "variable_size_test":
        # Generate variable-sized angle sets for testing
        variable_angle_sets = generate_variable_angle_pairs()
        for index, angle_set in enumerate(variable_angle_sets):
            mock_ampal1 = create_mock_ampal(
                angle_set
            )  # Use the angle set for the first structure
            for comparison_set in variable_angle_sets[
                index + 1 :
            ]:  # Compare with every other set
                mock_ampal2 = create_mock_ampal(comparison_set)
                difference = strategy_fixture.calculate_difference(
                    mock_ampal1, mock_ampal2
                )
                # Use an assertion that checks if the difference is within the expected theoretical bounds
                assert_theoretical_bounds_assertion(strategy_fixture, difference)


def assert_identical_structures_assertion(strategy, difference):
    if strategy.__class__.__name__ == "SphereDifferenceStrategy":
        assert np.isclose(
            difference, 1.0, atol=1e-2
        ), f"{strategy.__class__.__name__} failed for identical structures with difference {difference}, expected 1.0."
    elif strategy.__class__.__name__ == "CorrDifferenceStrategy":
        assert np.isclose(
            np.max(difference), 1.0, atol=1e-2
        ), f"{strategy.__class__.__name__} failed for identical structures with difference {difference}, expected max of 1.0."
    else:
        assert np.isclose(
            difference, strategy.theoretical_min, atol=1e-2
        ), f"{strategy.__class__.__name__} failed for identical structures with difference {difference}, expected {strategy.theoretical_min}."


def assert_theoretical_bounds_assertion(strategy, difference):
    if strategy.__class__.__name__ == "CorrDifferenceStrategy":
        assert np.all(difference >= strategy.theoretical_min) and np.all(
            difference <= strategy.theoretical_max
        ), f"{strategy.__class__.__name__}'s output {difference} is outside the theoretical range [0, 1]."
    else:
        assert (
            strategy.theoretical_min <= difference <= strategy.theoretical_max
        ), f"{strategy.__class__.__name__}'s output {difference} is outside the theoretical range [{strategy.theoretical_min}, {strategy.theoretical_max}]."


def assert_maximum_difference_assertion(strategy, difference):
    if strategy.__class__.__name__ == "CorrDifferenceStrategy":
        assert np.isclose(
            np.min(np.abs(difference)), 0.5, atol=1e-2
        ), f"{strategy.__class__.__name__} did not reach its theoretical maximum difference of 1.0, only achieved {np.max(difference)}."
    else:
        assert np.isclose(
            difference, strategy.theoretical_max, atol=1e-2
        ), f"{strategy.__class__.__name__} did not reach its theoretical maximum difference of {strategy.theoretical_max}, only achieved {difference}."
