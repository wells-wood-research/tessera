import argparse
import typing as t
from pathlib import Path
import ampal

from src.difference_fn.angle_difference import AngleDifferenceStrategy
from src.difference_fn.difference_processing import select_first_ampal_assembly
import matplotlib.pyplot as plt

import numpy as np

import numpy as np
import matplotlib.pyplot as plt

from pycirclize import Circos
import numpy as np


def plot_circos_phi_psi_pycirclize(phi_psi_array: np.ndarray) -> None:
    """
    Creates a circos plot of phi and psi angles using pycirclize.
    Phi angles will be colored blue, and psi angles will be colored yellow.

    Args:
        phi_psi_array (np.ndarray): A 2D numpy array with two columns,
                                    first for phi angles, second for psi angles.
    """
    # Ensure the array has the right shape
    assert phi_psi_array.shape[1] == 2, "Array must have two columns, one for phi and one for psi."

    # Create a Circos object
    circos = Circos()

    # Convert angles from degrees to a 0-360 scale
    phi_angles = (phi_psi_array[:, 0] + 180) % 360
    psi_angles = (phi_psi_array[:, 1] + 180) % 360

    # Add tracks for phi and psi
    circos.add_track(1, size=0.1)  # Track for phi
    circos.add_track(2, size=0.1)  # Track for psi

    # Plot phi angles in blue
    for angle in phi_angles:
        circos.get_track(1).axis(0, 1, color="blue", alpha=0.6, lw=2).line(angle)

    # Plot psi angles in yellow
    for angle in psi_angles:
        circos.get_track(2).axis(0, 1, color="yellow", alpha=0.6, lw=2).line(angle)

    # Set title for the plot
    circos.set_title("Phi and Psi Circos Plot")

    # Render the circos plot
    circos.show()


def plot_fourier_transform_phi_psi(phi_psi_array: np.ndarray) -> None:
    """
    Applies Fourier transform to the phi and psi angles and plots the magnitude of the frequencies.
    The mean is subtracted to remove the high 0-frequency component.

    Args:
        phi_psi_array (np.ndarray): A 2D numpy array with two columns,
                                    first for phi angles, second for psi angles.
    """
    # Ensure the array has the right shape
    assert phi_psi_array.shape[1] == 2, "Array must have two columns, one for phi and one for psi."

    # Extract phi and psi angles
    phi_angles = phi_psi_array[:, 0]
    psi_angles = phi_psi_array[:, 1]

    # Subtract the mean (remove DC component)
    phi_angles = phi_angles - np.mean(phi_angles)
    psi_angles = psi_angles - np.mean(psi_angles)

    # Apply Fourier Transform to phi and psi angles
    phi_fft = np.fft.fft(phi_angles)
    psi_fft = np.fft.fft(psi_angles)

    # Get frequency components (magnitude of the FFT)
    phi_magnitude = np.abs(phi_fft)
    psi_magnitude = np.abs(psi_fft)

    # Generate frequency bins
    frequencies = np.fft.fftfreq(len(phi_angles))

    # Plot the Fourier transform magnitudes for phi and psi angles
    plt.figure(figsize=(10, 5))

    # Phi angle frequency domain plot
    plt.subplot(1, 2, 1)
    plt.plot(frequencies, phi_magnitude, color='blue')
    plt.title('Fourier Transform of Phi Angles')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')

    # Psi angle frequency domain plot
    plt.subplot(1, 2, 2)
    plt.plot(frequencies, psi_magnitude, color='yellow')
    plt.title('Fourier Transform of Psi Angles')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')

    # Show plot
    plt.tight_layout()
    plt.show()

def main(args):
    args.input_path = Path(args.input_path)
    assert args.input_path.exists(), f"Input file {args.input_path} does not exist"
    protein = ampal.load_pdb(args.input_path)
    protein = select_first_ampal_assembly(protein)
    protein_angles = AngleDifferenceStrategy.get_ampal_data(protein)
    protein_angles = AngleDifferenceStrategy.normalise_angles(protein_angles)
    plot_circos_phi_psi_pycirclize(protein_angles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_path", type=Path, help="Path to input file")
    params = parser.parse_args()
    main(params)
