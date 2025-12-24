import typing as t
from pathlib import Path

import ampal
import numpy as np

from tessera.difference_fn.angle_difference import AngleDifferenceStrategy
from tessera.difference_fn.difference_processing import select_first_ampal_assembly
from tessera.difference_fn.shape_difference import ShapeDifferenceStrategy


class ReferenceFragment:
    """
    A class to represent the PDBs of a fragment
    """

    def __init__(
        self,
        fragment_number: int,
        difference_strategy: t.Union[AngleDifferenceStrategy, ShapeDifferenceStrategy],
        paths: t.List[Path],
    ) -> None:
        self.fragment_number = fragment_number
        self.difference_strategy = difference_strategy
        self.paths = paths
        # use difference function to load the data
        self.data, self.fragment_length = self._load_fragment_data()

    def __repr__(self):
        return f"{self.__class__.__name__}(fragment_number={self.fragment_number}, fragment_length_length={self.fragment_length}, difference_metric={self.difference_strategy}, all_pdb_paths={len(self.paths)})"

    def _load_fragment_data(self) -> t.Tuple[np.ndarray, int]:
        """
        Load the fragment data from the PDB files.
        """
        data = []
        for pdb_path in self.paths:
            pdb_structure = select_first_ampal_assembly(ampal.load_pdb(pdb_path))
            data.append(self.difference_strategy.get_ampal_data(pdb_structure))

        data = np.array(data)
        # Check that the data is not empty
        assert data.shape[0] > 0, f"No data found for fragment {self.fragment_number}"
        # If the data is numeric
        if np.issubdtype(data.dtype, np.number):
            # check that there are no NaNs
            assert not np.isnan(
                data
            ).any(), f"NaN found in fragment {self.fragment_number}"
            fragment_length = data.shape[1]
        elif data.dtype.type is np.str_ or data.dtype.type is np.object_:
            if any(len(seq) == 0 for seq in data):
                raise ValueError(
                    f"Empty sequence found in fragment {self.fragment_number}"
                )

            seq_lengths = np.array([len(seq) for seq in data])
            if not np.all(seq_lengths == seq_lengths[0]):
                raise ValueError(
                    f"Inconsistent sequence lengths in fragment {self.fragment_number}"
                )

            fragment_length = seq_lengths[0]
        else:
            raise ValueError(
                f"Unknown data type {data.dtype} in fragment {self.fragment_number}"
            )

        return data, fragment_length


class ReferenceFragmentCreator:
    """
    A class to create PDB fragments with persistent caching.
    """

    _fragment_cache: t.Dict[Path, t.Dict[int, t.Dict[str, t.Any]]] = {}

    def __init__(
        self,
        folder_path: Path,
        difference_fn: t.Union[AngleDifferenceStrategy, ShapeDifferenceStrategy],
    ) -> None:
        self.folder_path = folder_path
        self.difference_fn = difference_fn
        self.fragment_to_paths = self._get_fragment_data()

    def _get_fragment_data(self) -> t.Dict[int, t.Dict[str, t.Any]]:
        """
        Loads fragment data from folder.

        Theoretically, a cache could save some time, but I noticed that this process is quite fast.
        """

        # Generate fragment data if cache does not exist
        fragment_to_paths = {}
        missing_pdb_folders = []

        for curr_path in self.folder_path.glob("**/*"):
            if (
                curr_path.is_dir()
                and not curr_path.name.startswith("B")
                and curr_path.name.isdigit()
            ):
                fragment_number = int(curr_path.name)
                pdb_files = [f for f in curr_path.iterdir() if f.suffix == ".pdb1"]

                if not pdb_files:
                    missing_pdb_folders.append(curr_path)

                fragment_to_paths[fragment_number] = {"all_pdb_paths": pdb_files}

        # Ensure every folder contains at least one .pdb1 file
        if missing_pdb_folders:
            raise FileNotFoundError(
                f"The following fragment folders are missing .pdb1 files: "
                f"{', '.join(str(folder) for folder in missing_pdb_folders)}"
            )

        if not fragment_to_paths:
            raise FileNotFoundError(
                f"No valid fragments found in {self.folder_path} with file extension .pdb1"
            )

        self._fragment_cache[self.folder_path] = fragment_to_paths
        return fragment_to_paths

    def create_all_fragments(self) -> t.Dict[int, ReferenceFragment]:
        """Creates all fragments using cached data."""
        return {
            fragment: self.create_fragment(fragment)
            for fragment in self.fragment_to_paths
        }

    def create_fragment(self, fragment_number: int) -> ReferenceFragment:
        """Creates a single fragment using cached data."""
        fragment_data = self.fragment_to_paths[fragment_number]
        all_pdb_paths = fragment_data["all_pdb_paths"]

        return ReferenceFragment(
            fragment_number=fragment_number,
            paths=all_pdb_paths,
            difference_strategy=self.difference_fn,
        )
