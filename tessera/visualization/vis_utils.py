import matplotlib.pyplot as plt
from pathlib import Path
import typing as t
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches

from tessera.fragments.fragments_classifier import StructureFragment


def check_fragment_number(fragment_number: int):
    assert fragment_number in range(
        1, 41
    ), f"Fragment number must be between 1 and 40. Got {fragment_number}."


def check_highlight_regions(highlights: t.List[t.Tuple[int, int]], data_length: int):
    for start, end in highlights:
        assert (
            start < end
        ), f"Start position must be less than end position. Got {start} and {end}."
        assert (
            start >= 0
        ), f"Start position must be greater than or equal to 0. Got {start}."
        assert (
            end <= data_length
        ), f"End position must be less than or equal to the length of the data. Got {end}."


def plot_fragment_distance(
    structure_fragment: StructureFragment,
    fragment_number: int,
    highlights: t.List[t.Tuple[int, int]] = None,
    title_suffix: str = "",
    y_label: str = "Difference Score",
    pdf_save_path: Path = None,
    plot_inline: bool = True,
):
    # Data for the specific fragment
    fragment_data = structure_fragment.raw_distance_data[:, fragment_number - 1]
    # Validation checks
    check_fragment_number(fragment_number)
    if highlights:
        check_highlight_regions(highlights, len(structure_fragment.raw_distance_data))

    # Plot setup
    plt.figure(figsize=(10, 6))
    plt.plot(fragment_data, linewidth=2)
    plt.xlabel("Residue Position")
    plt.ylabel(y_label)
    plt.title(
        f"{structure_fragment.structure_path.name} - Fragment {fragment_number}\n{title_suffix}"
    )

    # Highlighting regions
    if highlights:
        for start, end in highlights:
            plt.axvspan(
                start, end, color="yellow", alpha=0.5, label="Highlighted Region"
            )

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.grid(True)
    plt.tight_layout()

    # Save or show plot
    if pdf_save_path:
        plt.savefig(pdf_save_path, format="pdf")
    if plot_inline:
        plt.show()
    else:
        plt.close()


def calculate_fragment_stats(
    structure_fragment: StructureFragment,
    fragment_number: int,
    region_tuple: t.Optional[t.Tuple[int, int]] = None,
    arg_fn: t.Callable = np.argmin,
) -> t.Tuple[float, float, float, float]:
    """
    Calculates:

        1. the distance between the position of the peak middle and the position of the lowest peak.
        2. the percentage increase of the highest peak vs the median of the signal
        3. signal to noise ratio

    Args:
        fragment_number (int): The fragment number to calculate the stats for.
        region_tuple (Tuple[int, int]): The start and end positions of the region to consider.
        arg_fn (Callable, optional): The function to use to find the index of the minimum value. Default is np.argmin.

    Returns:
        tuple: A tuple containing the three calculated statistics.
    """
    # Validation checks
    check_fragment_number(fragment_number)

    # Extract the fragment data_paths
    fragment_data = structure_fragment.raw_distance_data[:, fragment_number - 1]

    # Determine the region for analysis
    start, end = region_tuple if region_tuple else (0, len(fragment_data))
    region_data = fragment_data[start:end]

    # Validate the region
    check_highlight_regions([(start, end)], len(fragment_data))

    # Find peak and calculate statistics
    peak_idx = arg_fn(region_data) + start
    peak_value = fragment_data[peak_idx]
    median = np.median(fragment_data)
    distance_peak_middle = peak_idx - (start + end) // 2
    percentage_increase_median = (
        ((peak_value - median) / median) * 100 if median != 0 else 0
    )
    signal_to_noise_ratio = 10 * np.log10(
        np.mean(fragment_data) ** 2 / np.std(fragment_data) ** 2
    )

    return distance_peak_middle, percentage_increase_median, signal_to_noise_ratio


def plot_highest_probability_and_entropy(structure_fragment: StructureFragment):
    probabilities = structure_fragment.probability_distance_data
    n, num_fragments = probabilities.shape

    # Calculate highest probability and its index for each position
    highest_probabilities = np.max(probabilities, axis=1)
    fragment_indices = (
        np.argmax(probabilities, axis=1) + 1
    )  # Adding 1 for 1-indexed fragments

    # Calculate Shannon entropy
    entropy = -np.sum(
        probabilities * np.log2(probabilities + 1e-9), axis=1
    )  # Small value added to avoid log(0)
    max_entropy = -np.log2(1 / num_fragments)
    normalized_entropy = entropy / max_entropy

    # Plotting
    fig, ax1 = plt.subplots()

    # Highest probability and index
    ax1.bar(range(n), highest_probabilities, color="b", alpha=0.6)
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Highest Probability", color="b")
    for i, idx in enumerate(fragment_indices):
        ax1.text(i, highest_probabilities[i], str(idx), ha="center", va="bottom")
    ax1.set_ylim([0, 1])  # Setting y-axis limits to 0-1

    # Entropy
    ax2 = ax1.twinx()
    ax2.plot(range(n), normalized_entropy, color="r", marker="o", linestyle="-")
    ax2.set_ylabel("Normalized Entropy", color="r")
    ax2.set_ylim([0, 1])  # Normalized entropy also between 0 and 1

    plt.title("Highest Probability and Normalized Entropy per Position")
    plt.show()


def plot_probability_distribution_violin(structure_fragment: StructureFragment):
    probabilities = structure_fragment.probability_distance_data
    n_positions, n_fragments = probabilities.shape

    # Preparing the data_paths for plotting
    data = []
    for pos in range(n_positions):
        for frag in range(n_fragments):
            data.append(
                {
                    "Position": pos,
                    "Fragment Index": frag + 1,  # Adding 1 for 1-indexed fragments
                    "Probability": probabilities[pos, frag],
                }
            )

    df = pd.DataFrame(data)

    # Creating the violin plot
    fig, ax = plt.subplots(figsize=(20, 6))
    sns.violinplot(
        x="Position",
        y="Fragment Index",
        data=df,
        scale="width",
        inner="quartile",
        ax=ax,
    )

    ax.set_xlabel("Position")
    ax.set_ylabel("Fragment Index")
    plt.title("Probability Distribution of Fragments at Each Position")
    plt.tight_layout()
    plt.show()


def plot_highest_and_second_highest_probabilities(
    structure_fragment: StructureFragment,
):
    probabilities = structure_fragment.probability_distance_data
    n = probabilities.shape[0]

    # Calculate highest and second highest probabilities and their indices
    sorted_indices = np.argsort(probabilities, axis=1)
    highest_indices = sorted_indices[:, -1] + 1  # Adding 1 for 1-indexed fragments
    second_highest_indices = sorted_indices[:, -2] + 1  # Second highest
    highest_probabilities = np.take_along_axis(
        probabilities, sorted_indices[:, -1:], axis=1
    ).flatten()
    second_highest_probabilities = np.take_along_axis(
        probabilities, sorted_indices[:, -2:-1], axis=1
    ).flatten()

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.4
    ax.bar(
        np.arange(n) - bar_width / 2,
        highest_probabilities,
        width=bar_width,
        label="Highest",
        color="skyblue",
    )
    ax.bar(
        np.arange(n) + bar_width / 2,
        second_highest_probabilities,
        width=bar_width,
        label="Second Highest",
        color="lightgreen",
    )

    # Adding text for the probability values and indices
    for i in range(n):
        ax.text(
            i - bar_width / 2,
            highest_probabilities[i],
            f"{highest_indices[i]}\n({highest_probabilities[i]:.2f})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            i + bar_width / 2,
            second_highest_probabilities[i],
            f"{second_highest_indices[i]}\n({second_highest_probabilities[i]:.2f})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Position")
    ax.set_ylabel("Probability")
    ax.set_title("Highest and Second Highest Probabilities per Position")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_probability_heatmap(structure_fragment: StructureFragment):
    probabilities = structure_fragment.probability_distance_data
    n_positions, n_fragments = probabilities.shape

    # Find the fragment with the highest probability at each position
    highest_prob_indices = np.argmax(probabilities, axis=1)

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(25, 5))
    sns.heatmap(
        probabilities.T, cmap="viridis", ax=ax
    )  # Transposed for correct orientation

    # Highlight the highest probability fragment for each position
    for pos, frag in enumerate(highest_prob_indices):
        # Draw a red rectangle around the cell
        rect = patches.Rectangle(
            (pos, frag), 1, 1, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

    # Set x-axis ticks
    ax.set_xticks(np.arange(n_positions))
    ax.set_xticklabels(np.arange(1, n_positions + 1))

    ax.set_xlabel("Position")
    ax.set_ylabel("Fragment Index")
    ax.set_title(
        "Probability Heatmap of Fragments at Each Position (Highest probabilities highlighted)"
    )

    plt.show()


from scipy.stats import entropy


def plot_entropy_with_uniform_line(structure_fragment: StructureFragment):
    probabilities = structure_fragment.probability_distance_data
    n_positions, n_fragments = probabilities.shape

    # Calculate Shannon entropy for each position
    position_entropies = [
        entropy(probabilities[i, :], base=2) for i in range(n_positions)
    ]

    # Entropy of a uniform distribution
    uniform_entropy = entropy([1 / n_fragments] * n_fragments, base=2)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(position_entropies, label="Position Entropy", marker="o")
    plt.axhline(
        y=uniform_entropy,
        color="r",
        linestyle="-",
        label=f"Uniform Entropy ({uniform_entropy:.2f})",
    )

    plt.xlabel("Position")
    plt.ylabel("Entropy (bits)")
    plt.title("Shannon Entropy per Position with Uniform Entropy Line")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_violin_for_each_position(structure_fragment: StructureFragment):
    probabilities = structure_fragment.probability_distance_data
    n_positions, n_fragments = probabilities.shape

    # Preparing the data_paths for seaborn
    data = {"Position": [], "Fragment Index": [], "Probability": []}
    for position in range(n_positions):
        for fragment in range(n_fragments):
            data["Position"].append(position)
            data["Fragment Index"].append(fragment + 1)
            data["Probability"].append(probabilities[position, fragment])

    df = pd.DataFrame(data)

    # Plotting
    plt.figure(figsize=(15, 6))
    sns.violinplot(
        x="Position", y="Probability", data=df, scale="width", inner="quartile"
    )

    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.title("Probability Distribution of Fragments at Each Position")
    plt.show()


def plot_and_save_each_position(structure_fragment: StructureFragment, save_dir: Path):
    probabilities = structure_fragment.probability_distance_data
    n_positions, n_fragments = probabilities.shape

    # Ensure the save directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create and save a plot for each position
    for pos in range(n_positions):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, n_fragments + 1), probabilities[pos, :], marker="o")

        plt.xlabel("Fragment Number")
        plt.ylabel("Probability")
        plt.title(f"Probability Distribution at Position {pos + 1}")
        plt.grid(True)

        # Save the plot
        file_path = save_dir / f"position_{pos + 1}.png"
        plt.savefig(file_path)
        plt.close()


def plot_all_positions_together(structure_fragment: StructureFragment):
    probabilities = structure_fragment.probability_distance_data
    n_positions, n_fragments = probabilities.shape

    plt.figure(figsize=(15, 8))

    start_pos = 25
    end_pos = n_positions - 25

    # Plotting the probability distribution for the selected range of positions
    for pos in range(start_pos, end_pos):
        plt.plot(
            np.arange(1, n_fragments + 1),
            probabilities[pos, :],
            alpha=0.2,
            color="black",
        )

    plt.xlabel("Fragment Number", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.title("Probability Distribution Across All Positions", fontsize=16)
    plt.xticks(
        np.arange(1, n_fragments + 1), fontsize=12
    )  # Ensure each fragment number is shown
    plt.grid(True)
    plt.show()


def check_probability_sum(structure_fragment: StructureFragment):
    probabilities = structure_fragment.probability_distance_data
    n_positions = probabilities.shape[0]

    # Summing probabilities vertically for each position
    sums = np.sum(probabilities, axis=1)

    # Check if sums are approximately 1
    are_sums_one = np.allclose(sums, 1.0)

    return are_sums_one, sums


def plot_probability_frequency(structure_fragment: StructureFragment, bins=15):
    check_probability_sum(structure_fragment)
    probabilities = structure_fragment.probability_distance_data
    n_fragments = probabilities.shape[1]

    plt.figure(figsize=(12, 8))

    # Plot a line histogram for each fragment
    for i in range(n_fragments):
        hist, bin_edges = np.histogram(probabilities[:, i], bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.plot(bin_centers, hist, alpha=0.25, color="black")

    plt.xlabel("Probability", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.title("Probability Frequency Distribution for Each Fragment", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_position_probability_frequency(
    structure_fragment: StructureFragment, bins=100
):
    probabilities = structure_fragment.probability_distance_data
    n_positions, n_fragments = probabilities.shape

    plt.figure(figsize=(12, 8))

    # Plot a line histogram for each position
    for pos in range(n_positions):
        hist, bin_edges = np.histogram(probabilities[pos, :], bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.plot(bin_centers, hist, alpha=0.25, color="black")

    plt.axvline(
        x=1 / n_fragments, color="red", linestyle="--", label=f"1/{n_fragments}"
    )

    plt.xlabel("Probability", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.title("Probability Frequency Distribution for Each Position", fontsize=16)
    plt.grid(True)
    plt.show()


def plot_probability_frequency_by_position(
    structure_fragment: StructureFragment, bins=5
):
    probabilities = structure_fragment.probability_distance_data
    n_positions = probabilities.shape[0]
    n_fragments = probabilities.shape[1]

    plt.figure(figsize=(12, 8))

    max_frequency = 0  # Initialize a variable to store the maximum frequency

    # Plotting the probability distribution for the selected range of positions
    start_pos = 25
    end_pos = n_positions - 25
    for pos in range(start_pos, end_pos):
        hist, bin_edges = np.histogram(probabilities[pos, :], bins=bins, density=False)
        max_frequency = max(
            max_frequency, max(hist)
        )  # Update the maximum frequency if needed
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.plot(
            bin_centers, hist, alpha=0.15, color="black", linewidth=2
        )  # Increased linewidth

    plt.axvline(
        x=1 / n_fragments, color="red", linestyle="--", label=f"1/{n_fragments}"
    )

    plt.xlabel("Probability Value", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)

    # Set y-axis ticks at every unit of frequency up to the maximum frequency
    plt.yticks(np.arange(0, max_frequency + 1, 1))

    plt.title(
        f"Frequency of Probabilities for Each Position\n Bins = {bins}", fontsize=16
    )
    plt.grid(True)
    plt.legend()
    plt.show()
