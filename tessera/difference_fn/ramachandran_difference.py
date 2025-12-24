import typing as t

import numpy as np
from ampal import Polypeptide
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde, vonmises

from tessera.difference_fn.angle_difference import AngleDifferenceStrategy


class RamachandranCircularKDE(AngleDifferenceStrategy):
    """
    Class to calculate the difference between two sets of angular data
    using Circular Kernel Density Estimation (KDE) with a von Mises kernel.
    """

    def __init__(
        self,
        arg_fn: t.Callable = np.argmin,
        data_type: str = "angle",
        kappa: float = 1.0,
    ):
        """
        Initialize the Circular KDE calculator.

        Parameters:
            arg_fn (t.Callable): Function to use for comparison (e.g., np.argmin).
            data_type (str): Type of data (default is "angle").
            kappa (float): Concentration parameter for the von Mises distribution.
        """
        self.kappa = kappa
        self.grid = np.linspace(0, 2 * np.pi, 360)  # Precompute the grid
        self.grid_len = len(self.grid)
        super().__init__(arg_fn, data_type)

    @property
    def theoretical_max(self) -> float:
        """Returns the theoretical maximum JS divergence value."""
        return 1

    @property
    def theoretical_min(self) -> float:
        """Returns the theoretical minimum JS divergence value."""
        return 0

    def _kde_density(self, coords, kappa=1.0):
        """
        Calculate the KDE density for a set of phi and psi coordinates.

        Parameters:
            coords (np.ndarray): Array of shape (N, 2) where N is the number of amino acids
                                 and 2 represents the phi and psi angles.
            kappa (float): Concentration parameter for the von Mises distribution.

        Returns:
            np.ndarray: Density values computed over a grid for visualization.
        """
        assert (
            coords.ndim == 2 and coords.shape[1] == 2
        ), "coords should be of shape (N, 2)"

        # Assuming self.grid is a 1D array of angles in radians
        phi_grid, psi_grid = np.meshgrid(self.grid, self.grid)
        grid_points = np.vstack([phi_grid.ravel(), psi_grid.ravel()]).T

        # Calculate the differences in phi and psi separately
        phi_diffs = grid_points[:, 0][:, None] - coords[:, 0]
        psi_diffs = grid_points[:, 1][:, None] - coords[:, 1]

        # Apply von Mises PDF to each set of differences
        phi_densities = vonmises.pdf(phi_diffs, kappa=kappa)
        psi_densities = vonmises.pdf(psi_diffs, kappa=kappa)

        # Combine densities for each point by multiplying corresponding densities
        densities = np.sum(phi_densities * psi_densities, axis=1) / len(coords)

        # Reshape to match the original grid shape
        return densities.reshape(phi_grid.shape)

    def calculate_difference(
        self, reference_ampal: Polypeptide, fragment_ampal: Polypeptide
    ) -> float:
        """
        Calculate the difference between two sets of angles using Circular KDE.

        Parameters:
            reference_ampal (Polypeptide): Reference polypeptide angles.
            fragment_ampal (Polypeptide): Fragment polypeptide angles.

        Returns:
            float: Jensen-Shannon Divergence between the two angle distributions.
        """
        reference_coords = self._prepare_angles(reference_ampal)
        fragment_coords = self._prepare_angles(fragment_ampal)

        reference_density = self._kde_density(reference_coords)
        fragment_density = self._kde_density(fragment_coords)
        # Flatten the 2D density distributions to 1D arrays
        reference_density_flat = reference_density.ravel()
        fragment_density_flat = fragment_density.ravel()

        # Normalize the densities to form proper probability distributions
        reference_prob = reference_density_flat / reference_density_flat.sum()
        fragment_prob = fragment_density_flat / fragment_density_flat.sum()

        # Calculate the Jensen-Shannon Divergence
        js_divergence = jensenshannon(reference_prob, fragment_prob)
        return js_divergence


class RamachandranProjected3DKDE(AngleDifferenceStrategy):
    """
    Class to calculate the difference between two sets of angular data
    using Projected 3D Kernel Density Estimation (KDE) with Gaussian kernels.
    """

    def __init__(self, arg_fn: t.Callable = np.argmin, data_type: str = "angle"):
        """
        Initialize the Projected 3D KDE calculator.

        Parameters:
            arg_fn (t.Callable): Function to use for comparison (e.g., np.argmin).
            data_type (str): Type of data (default is "angle").
        """
        super().__init__(arg_fn, data_type)
        self.grid_phi, self.grid_psi = np.mgrid[
            0 : 2 * np.pi : 360j, 0 : 2 * np.pi : 360j
        ]
        self.grid_coords = self._grid_to_cartesian(self.grid_phi, self.grid_psi)

    @property
    def theoretical_max(self) -> float:
        """Returns the theoretical maximum JS divergence value."""
        return 1

    @property
    def theoretical_min(self) -> float:
        """Returns the theoretical minimum JS divergence value."""
        return 0

    def _project_to_sphere(self, phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """
        Project angular data onto a unit sphere.

        Parameters:
            phi (np.ndarray): Array of phi angles in radians.
            psi (np.ndarray): Array of psi angles in radians.

        Returns:
            np.ndarray: Cartesian coordinates on a unit sphere.
        """
        return np.vstack(
            [np.cos(phi) * np.cos(psi), np.sin(phi) * np.cos(psi), np.sin(psi)]
        )

    def _grid_to_cartesian(self, phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """
        Convert grid of spherical coordinates to Cartesian coordinates.

        Parameters:
            phi (np.ndarray): Grid of phi angles in radians.
            psi (np.ndarray): Grid of psi angles in radians.

        Returns:
            np.ndarray: Cartesian coordinates.
        """
        x = np.cos(phi) * np.cos(psi)
        y = np.sin(phi) * np.cos(psi)
        z = np.sin(psi)
        return np.vstack([x.ravel(), y.ravel(), z.ravel()])

    def calculate_difference(
        self, reference_ampal: Polypeptide, fragment_ampal: Polypeptide
    ) -> float:
        angles_ref = self._prepare_angles(reference_ampal)
        angles_frag = self._prepare_angles(fragment_ampal)

        reference_phi, reference_psi = angles_ref[:, 0], angles_ref[:, 1]
        fragment_phi, fragment_psi = angles_frag[:, 0], angles_frag[:, 1]

        ref_coords = self._project_to_sphere(reference_phi, reference_psi)
        frag_coords = self._project_to_sphere(fragment_phi, fragment_psi)

        kde_ref = gaussian_kde(ref_coords)
        kde_frag = gaussian_kde(frag_coords)

        density_ref = kde_ref(self.grid_coords)
        density_frag = kde_frag(self.grid_coords)

        # Flatten and normalize
        density_ref_flat = density_ref / density_ref.sum()
        density_frag_flat = density_frag / density_frag.sum()

        return jensenshannon(density_ref_flat, density_frag_flat)


class RamachandranNormalKDE(AngleDifferenceStrategy):
    """
    Class to calculate the difference between two sets of angular data
    using standard Kernel Density Estimation (KDE) with Gaussian kernels.
    """

    def __init__(self, arg_fn: t.Callable = np.argmin, data_type: str = "angle"):
        """
        Initialize the Normal KDE calculator.

        Parameters:
            arg_fn (t.Callable): Function to use for comparison (e.g., np.argmin).
            data_type (str): Type of data (default is "angle").
        """
        super().__init__(arg_fn, data_type)
        self.grid_phi, self.grid_psi = np.meshgrid(
            np.linspace(0, 2 * np.pi, 360), np.linspace(0, 2 * np.pi, 360)
        )

    @property
    def theoretical_max(self) -> float:
        """Returns the theoretical maximum JS divergence value."""
        return 1

    @property
    def theoretical_min(self) -> float:
        """Returns the theoretical minimum JS divergence value."""
        return 0

    def calculate_difference(
        self, reference_ampal: Polypeptide, fragment_ampal: Polypeptide
    ) -> float:
        reference_coords = self._prepare_angles(reference_ampal)
        fragment_coords = self._prepare_angles(fragment_ampal)

        kde_ref = gaussian_kde(reference_coords.T)
        kde_frag = gaussian_kde(fragment_coords.T)

        # Flatten the grid for evaluation
        grid_coords = np.vstack([self.grid_phi.ravel(), self.grid_psi.ravel()])

        # Evaluate densities on the grid and flatten the results
        density_ref = kde_ref(grid_coords)
        density_frag = kde_frag(grid_coords)

        # Flatten densities to 1D
        density_ref_flat = density_ref.ravel()
        density_frag_flat = density_frag.ravel()

        # Normalize to create probability distributions
        density_ref_prob = density_ref_flat / np.sum(density_ref_flat)
        density_frag_prob = density_frag_flat / np.sum(density_frag_flat)

        # Calculate the Jensen-Shannon Divergence and return it
        js_divergence = jensenshannon(density_ref_prob, density_frag_prob)
        return js_divergence
