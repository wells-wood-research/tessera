import pytest
import numpy as np
from pathlib import Path
from src.fragments.fragments_classifier import StructureToFragmentClassifier, EnsembleFragmentClassifier

@pytest.fixture(
    params=[
        {"types": ["angle", "angle", "angle"], "names": ["LogPr", "RamRmsd", "Sphere"]},
        {"types": ["angle", "angle", "sequence"], "names": ["LogPr", "RamRmsd", "Blosum"]},
        {"types": ["angle", "angle"], "names": ["LogPr", "RamRmsd"]},
        {"types": ["angle", "angle"], "names": ["LogPr", "Sphere"]},
        {"types": ["angle"], "names": ["LogPr", "Rms"]},
        {"types": ["angle"], "names": ["LogPr", "Corr"]},
    ],
    ids=[
        "logpr_ramrmsd_sphere",
        "logpr_ramrmsd_blosum",
        "logpr_ramrmsd",
        "logpr_sphere",
        "logpr_rms",
        "logpr_corr",
    ]
)
def classifier_fixture(request):
    fragment_path = Path("test_fragments/")
    return {
        "classifier": EnsembleFragmentClassifier(
            fragment_path,
            request.param["types"],
            request.param["names"],
            n_processes=1,
            step_size=1
        ),
        "types": request.param["types"],
        "names": request.param["names"]
    }

def test_ensemble_integration(classifier_fixture):
    test_structure_path = Path("test_pdbs/1A8P.pdb1")
    # Run classification using the ensemble classifier
    ensemble_results = classifier_fixture["classifier"].classify_to_fragment(
        test_structure_path, use_all_fragments=True
    )

    # Check if ensemble_results are the correct average of individual classifier results
    individual_results = []
    for type_, name in zip(classifier_fixture["types"], classifier_fixture["names"]):
        classifier = StructureToFragmentClassifier(
            Path("test_fragments/"),
            type_,
            name,
            n_processes=1,
            step_size=1
        )
        result = classifier.classify_to_fragment(test_structure_path, use_all_fragments=True)
        individual_results.append(result.probability_distance_data)

    # Compute expected average of individual results
    expected_average = np.mean(individual_results, axis=0)

    # Assert the ensemble's results match the expected average
    assert np.allclose(ensemble_results.probability_distance_data, expected_average), "Ensemble probabilities do not match the expected average."

