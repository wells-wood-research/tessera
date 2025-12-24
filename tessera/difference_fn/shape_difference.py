import io
import typing as t
from abc import ABC, abstractmethod

import ampal
import numpy as np
from Bio.PDB import PDBExceptions, PDBParser
from Bio.PDB.cealign import CEAligner

try:
    import pymol
    from pymol import cmd
except ImportError:
    import warnings

    # warnings.warn("PyMOL not found. Please install PyMOL to use this function.")
    cmd = None


class ShapeDifferenceStrategy(ABC):
    @abstractmethod
    def calculate_difference(self, reference_ampal, fragment_ampal):
        pass


class RmsdBiopythonStrategy(ShapeDifferenceStrategy):
    def __init__(self):
        self.arg_fn = np.argmin
        self.data_type = "shape"
        self.n_processes = 1

    def calculate_difference(
        self, reference_ampal: ampal.Polypeptide, fragment_ampal: ampal.Polypeptide
    ) -> t.Union[float, t.Any]:
        """
        Wrapper function to calculate the RMSD between two AMPAL objects using the CEAligner from BioPython.

        Parameters
        ----------
        reference_ampal: ampal.Polypeptide
        fragment_ampal: ampal.Polypeptide

        Returns
        -------
        float
        """
        return self.biopython_calculate_rmsd(reference_ampal, fragment_ampal)

    @staticmethod
    def biopython_calculate_rmsd(
        reference_ampal: ampal.Polypeptide, fragment_ampal: ampal.Polypeptide
    ) -> float:
        """
        Calculate the RMSD between two AMPAL objects using the CEAligner from BioPython.

        NOTE: This function may hang due to BioPython's behaviour. I've opened an issue here: https://github.com/biopython/biopython/issues/4888

        Parameters
        ----------
        reference_ampal : ampal.Polypeptide
            The reference structure in AMPAL format.
        fragment_ampal : ampal.Polypeptide
            The fragment structure in AMPAL format.

        Returns
        -------
        float
            The RMSD value. Returns np.inf in case of error.
        """
        try:
            # Initialize the PDB parser
            parser = PDBParser()  # Suppress warnings

            # Convert AMPAL objects to file-like objects using StringIO
            with io.StringIO(
                reference_ampal.make_pdb()
            ) as file_like_structure1, io.StringIO(
                fragment_ampal.make_pdb()
            ) as file_like_structure2:

                # Parse the two structures from the file-like objects
                structure1 = parser.get_structure("structure1", file_like_structure1)
                structure2 = parser.get_structure("structure2", file_like_structure2)

                # Initialize the CEAligner
                ce_aligner = CEAligner()

                # Perform the alignment
                ce_aligner.set_reference(structure1)
                ce_aligner.align(structure2, transform=False)

                # Get the RMSD from the alignment
                rmsd = ce_aligner.rms

        except PDBExceptions.PDBException as e:
            print(f"An error occurred: {e}. Using default RMSD value.")
            rmsd = np.inf  # Set a default high RMSD value in case of exception

        return rmsd

    @staticmethod
    def biopython_calculate_rmsd_fast(reference_pdb: str, fragment_pdb: str) -> float:
        """
        Calculate the RMSD between two AMPAL objects using the CEAligner from BioPython.

        NOTE: This function may hang due to BioPython's behaviour. I've opened an issue here: https://github.com/biopython/biopython/issues/4888

        Parameters
        ----------


        -------
        float
            The RMSD value. Returns np.inf in case of error.
        """
        try:
            # Initialize the PDB parser
            parser = PDBParser()  # Suppress warnings

            # Convert AMPAL objects to file-like objects using StringIO
            with io.StringIO(reference_pdb) as file_like_structure1, io.StringIO(
                fragment_pdb
            ) as file_like_structure2:

                # Parse the two structures from the file-like objects
                structure1 = parser.get_structure("structure1", file_like_structure1)
                structure2 = parser.get_structure("structure2", file_like_structure2)

                # Initialize the CEAligner
                ce_aligner = CEAligner()

                # Perform the alignment
                ce_aligner.set_reference(structure1)
                ce_aligner.align(structure2, transform=False)

                # Get the RMSD from the alignment
                rmsd = ce_aligner.rms

        except PDBExceptions.PDBException as e:
            print(f"An error occurred: {e}. Using default RMSD value.")
            rmsd = np.inf  # Set a default high RMSD value in case of exception

        return rmsd


class RmsdPyMOLStrategy(ShapeDifferenceStrategy):
    def __init__(self):
        if not cmd:
            raise ImportError(
                "PyMOL not found. Please install PyMOL to use this function."
            )
        self.arg_fn = np.argmin
        self.data_type = "shape"
        self.n_processes = 1

    def calculate_difference(self, reference_ampal, fragment_ampal):
        try:
            # Launch PyMOL and clean up
            pymol.pymol_argv = ["pymol", "-qc"]
            pymol.finish_launching()
            cmd.delete("all")

            # Load structures into PyMOL
            cmd.read_pdbstr(reference_ampal.make_pdb(), "reference")
            cmd.read_pdbstr(fragment_ampal.make_pdb(), "fragment")

            # Perform the alignment
            alignment_result = cmd.cealign(
                target="reference and name CA", mobile="fragment and name CA"
            )
            rmsd = alignment_result["RMSD"]
        except Exception as e:
            rmsd = 1000.0  # Set a default high RMSD value in case of failure
        return rmsd
