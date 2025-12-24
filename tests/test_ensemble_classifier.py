import pytest
import numpy as np
from pathlib import Path
from tessera.fragments.fragments_classifier import EnsembleFragmentClassifier

@pytest.fixture(
    params=[
        ["logpr", "RamRmsd"],
        ["logpr"],
    ],
    ids=[
        "logpr_ramrmsd",
        "logpr_only",
    ]
)
def classifier_fixture(request):
    fragment_path = Path("data/fragments/")
    return {
        "classifier": EnsembleFragmentClassifier(
            fragment_path,
            difference_names=request.param,
            n_processes=1,
            step_size=1
        ),
        "names": request.param
    }

def test_ensemble_integration(classifier_fixture):
    # Skip this test if fragment data not available
    # Use fragment 13 (37 residues) - longest fragment to ensure it's longer than all others
    test_structure_path = Path("data/fragments/13")
    if not test_structure_path.exists():
        pytest.skip("Test fragment data not available")
    
    # Get first PDB file from fragment 13 directory
    import glob
    pdb_files = glob.glob(str(test_structure_path / "*.pdb*"))
    if not pdb_files:
        pytest.skip("No PDB files found in fragment directory")
    test_structure_path = Path(pdb_files[0])
    
    # Run classification using the ensemble classifier - just check it doesn't crash
    ensemble_results = classifier_fixture["classifier"].classify_to_fragment(
        test_structure_path, use_all_fragments=False
    )

    # Basic sanity checks
    assert ensemble_results is not None
    assert ensemble_results.classification is not None
    assert ensemble_results.classification_map is not None
    assert len(ensemble_results.classification_map) > 0

