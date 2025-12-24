import typing as t

from abc import ABC, abstractmethod

import ampal
import numpy as np


class AngleDifferenceStrategy(ABC):
    def __init__(self, arg_fn: t.Callable = np.argmin, data_type: str = "angle"):
        self.arg_fn = arg_fn
        self.data_type = data_type

    @property
    @abstractmethod
    def theoretical_max(self) -> float:
        pass

    @property
    @abstractmethod
    def theoretical_min(self) -> float:
        pass

    @staticmethod
    def calculate_difference(reference_ampal, fragment_ampal):
        pass

    @staticmethod
    def get_angles(ampal_obj: ampal.Polypeptide) -> np.ndarray:
        ampal_obj.tag_torsion_angles()
        angles = np.array(
            [
                [residue.tags["phi"], residue.tags["psi"]]
                for residue in ampal_obj.get_monomers(ligands=False)
            ],
            dtype=float,
        )
        return angles


    def get_ampal_data(self, ampal_obj: ampal.Polypeptide) -> np.ndarray:
        """
        Get the angles from the AMPAL object.

        Args:
            ampal_obj (ampal.Polypeptide): The AMPAL object.

        Returns:
            np.ndarray: An array of angles.
        """

        # Call prepare angles
        angles = self.get_angles(ampal_obj)
        return self._prepare_angles(angles)

    @staticmethod
    def convert_to_sin_cos(angles: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Convert angles to sin and cos components.

        Args:
            angles (np.ndarray): An array of angles of shape (n, 3) for omega, phi, and psi where n is the number of residues.

        Returns:
            tuple: A tuple containing the sin and cos components of shape ((n, 3), (n, 3).
        """
        sin_angles = np.sin(np.radians(angles))
        cos_angles = np.cos(np.radians(angles))
        return sin_angles, cos_angles

    @staticmethod
    def normalise_angles(angles: np.ndarray) -> np.ndarray:
        """
        Normalise the angles to be between 0 and 1.

        Args:
            angles (np.ndarray): An array of angles.

        Returns:
            np.ndarray: The normalised angles.
        """
        return np.nan_to_num(angles + 180, nan=0)

    def _prepare_angles(
        self, angles: t.Union[ampal.Polypeptide, np.ndarray]
    ) -> np.ndarray:
        """
        Prepare the angles from the AMPAL object.

        Args:
            angles (ampal.Polypeptide): The AMPAL object.

        Returns:
            np.ndarray: An array of angles.
        """
        angles = self.normalise_angles(angles)
        angles = np.radians(angles)

        return angles

    def normalise_difference(
        self, difference: np.ndarray, clip_probabilities: bool = True
    ) -> np.ndarray:
        """
        Normalise the difference between 0 and 1.

        Args:
            difference (np.ndarray): The difference array.

        Returns:
            np.ndarray: The normalised difference.
        """
        normalised_diff = (difference - self.theoretical_min) / (
            self.theoretical_max - self.theoretical_min
        )
        if clip_probabilities:
            return np.clip(normalised_diff, 0, 1, out=normalised_diff)
        else:
            return normalised_diff


class RmsDifferenceStrategy(AngleDifferenceStrategy):
    """
    Create a difference strategy that calculates the RMS difference between two AMPAL objects.

    Formula:
    RMS = sqrt(mean((reference - fragment)^2))
    """

    def __init__(self, arg_fn: t.Callable = np.argmin, data_type: str = "angle"):
        super().__init__(arg_fn, data_type)

    @property
    def theoretical_max(self) -> float:
        return np.pi

    @property
    def theoretical_min(self) -> float:
        return 0

    @staticmethod
    def calculate_difference(
        reference_angles: np.ndarray, fragment_angles: np.ndarray
    ) -> np.ndarray:
        return np.sqrt(
            np.nanmean((reference_angles - fragment_angles) ** 2, axis=(1, 2))
        )


class SphereDifferenceStrategy(AngleDifferenceStrategy):
    """
    Convert the angles to 3D sphere coordinates and calculate the difference using the dot product.

    0 means the vectors are orthogonal, 1 means they are identical, and -1 means they are opposite.

    Formula:
    dot_product(reference, fragment)
    `
    """

    def __init__(self, arg_fn: t.Callable = np.argmin, data_type: str = "angle"):
        super().__init__(arg_fn, data_type)

    @property
    def theoretical_max(self) -> float:
        return 1

    @property
    def theoretical_min(self) -> float:
        return -1

    @staticmethod
    def calculate_difference(
        reference_coords: np.ndarray, fragment_coords: np.ndarray
    ) -> np.ndarray:
        return np.einsum("ijk,ijk->i", reference_coords, fragment_coords)

    def _prepare_angles(self, angles: np.ndarray) -> np.ndarray:
        """
        Prepare the angles from the AMPAL object.

        Args:
            angles (ampal.Polypeptide): The AMPAL object.

        Returns:
            np.ndarray: An array of angles.
        """
        angles = self.normalise_angles(angles)
        angles = np.nan_to_num(angles, nan=0)
        angles = np.radians(angles)

        sin_angles, cos_angles = self.convert_to_sin_cos(angles)

        sin_angles_phi = sin_angles[:, 0]
        sin_angles_psi = sin_angles[:, 1]

        cos_angles_phi = cos_angles[:, 0]
        cos_angles_psi = cos_angles[:, 1]

        x = sin_angles_phi * cos_angles_psi
        y = sin_angles_phi * sin_angles_psi
        z = cos_angles_phi

        coords = np.column_stack((x, y, z))
        return coords


class RamRmsdDifferenceStrategy(AngleDifferenceStrategy):
    def __init__(self, arg_fn: t.Callable = np.argmin, data_type: str = "angle"):
        super().__init__(arg_fn, data_type)

    @property
    def theoretical_max(self) -> float:
        """
        Maximum Euclidean distance for a single pair of angles:
        = sqrt((Δφ)^2 + (Δψ)^2)
        = sqrt((π)^2 + (π)^2)
        = sqrt(2π^2)
        = π√2

        Since the RMSD is the square root of the mean of these squared distances and each distance is the same in the maximum case:
        RMSD = sqrt(mean((π√2)^2))
             = sqrt(mean(2π^2))
             = sqrt(2π^2)
             = π√2 ≈ 4.442882938158366
        """
        return np.pi * np.sqrt(2)

    @property
    def theoretical_min(self) -> float:
        return 0

    @staticmethod
    def calculate_difference(
        reference_angles: np.ndarray, fragment_angles: np.ndarray
    ) -> np.ndarray:

        # This is the non-vectorised form - much easier to read but slower
        # reference_phi = reference_angles[:, 0]
        # reference_psi = reference_angles[:, 1]
        #
        # fragment_phi = fragment_angles[:, 0]
        # fragment_psi = fragment_angles[:, 1]
        #
        # delta_phi = reference_phi - fragment_phi
        # delta_psi = reference_psi - fragment_psi
        #
        # # Adjusting for periodic boundary conditions
        # delta_phi = np.where(delta_phi > np.pi, delta_phi - 2 * np.pi, delta_phi)
        # delta_phi = np.where(delta_phi < -np.pi, delta_phi + 2 * np.pi, delta_phi)
        #
        # delta_psi = np.where(delta_psi > np.pi, delta_psi - 2 * np.pi, delta_psi)
        # delta_psi = np.where(delta_psi < -np.pi, delta_psi + 2 * np.pi, delta_psi)
        #
        # # Calculating the Euclidean distances and then the RMSD
        # distances = np.sqrt(delta_phi ** 2 + delta_psi ** 2)
        # ramrmsd = np.sqrt(np.mean(distances ** 2))

        delta_angles = (
            np.mod(reference_angles - fragment_angles + np.pi, 2 * np.pi) - np.pi
        )
        # Sum squared differences for each residue (axis=-1) then average over residues (axis=1)
        return np.sqrt(np.mean(np.sum(delta_angles ** 2, axis=-1), axis=1))


class LogPrDifferenceStrategy(AngleDifferenceStrategy):
    def __init__(self, arg_fn: t.Callable = np.argmin, data_type: str = "angle"):
        super().__init__(arg_fn, data_type)

    @property
    def theoretical_max(self) -> float:
        """
        # Pr_{max} = \left( \frac{1}{\pi} (\pi + 1e-10) \right)^2
        # Approximates to:
        # Pr_{max} \approx \left( \frac{\pi}{\pi} \right)^2 = 1
        # LogPr_{max} = log_{10}(1) = 0
        """
        return 0

    @property
    def theoretical_min(self) -> float:
        """
        Pr_{min} = \left( \frac{1}{\pi} \cdot 1e-10 \right)^2
        The LogPr for the minimal difference scenario
        LogPr_{min} = log_{10}\left( \left( \frac{1}{\pi} \cdot 1e-10 \right)^2 \right)
        """
        return -20.994

    @staticmethod
    def calculate_difference(
        reference_angles: np.ndarray, fragment_angles: np.ndarray
    ) -> np.ndarray:

        # This is the non-vectorised form - much easier to read but slower

        # reference_phi = reference_angles[:, 0]
        # reference_psi = reference_angles[:, 1]
        #
        # fragment_phi = fragment_angles[:, 0]
        # fragment_psi = fragment_angles[:, 1]
        #
        # angular_differences_phi = np.abs(reference_phi - fragment_phi) + 1e-10
        # angular_differences_psi = np.abs(reference_psi - fragment_psi) + 1e-10
        #
        # # Compute rho(omega) for phi and psi angles
        # rho_phi = 1 / np.radians(180)
        # rho_psi = 1 / np.radians(180)
        #
        # # Calculate the Pr values
        # Pr_values = (rho_phi * angular_differences_phi) * (
        #     rho_psi * angular_differences_psi
        # )
        #
        # # Calculate logPr
        # Logpr = np.log10(Pr_values).mean()

        rho = 1 / np.radians(180)
        Pr_values = (rho * (np.abs(reference_angles - fragment_angles) + 1e-10)).prod(
            axis=2
        )
        return np.nanmean(np.log10(Pr_values), axis=1)
