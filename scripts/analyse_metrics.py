import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

from tessera.fragments.fragments_classifier import StructureToFragmentClassifier


# Function to process a single file and extract data_paths
def process_file(file_path):
    name_parts = file_path.stem.split("_")
    metric_name, fragment_n = name_parts[0], int(name_parts[2])
    start_pos, end_pos = int(name_parts[4]), int(name_parts[6]) - 1
    pdb_code = name_parts[7]
    data_array = pd.read_csv(file_path).to_numpy()

    if end_pos <= data_array.shape[0]:
        middle_index = start_pos + (end_pos - start_pos) // 2
        sliced_array = data_array[middle_index, :]

        result_value = np.argmax(sliced_array) + 1
        top_3_indices = np.argsort(sliced_array)[-3:][::-1] + 1

        return pdb_code, metric_name, fragment_n, result_value, top_3_indices
    return None


# Function to aggregate data_paths from all files in a directory
def aggregate_data_from_directory(metrics_path):
    pdb_data = {}
    for file_path in metrics_path.glob("*.csv"):
        result = process_file(file_path)
        if result:
            # Updated to unpack top_3_indices from the process_file return
            pdb_code, metric_name, fragment_n, result_value, top_3_indices = result

            if pdb_code not in pdb_data:
                pdb_data[pdb_code] = {"ground_truth": [fragment_n]}
            pdb_data[pdb_code][f"{metric_name}"] = result_value
            pdb_data[pdb_code][f"{metric_name}_3"] = top_3_indices
    return pdb_data


def calculate_top3_accuracy(ground_truths, top3_indices):
    """
    Calculates the top-3 accuracy.

    Parameters:
    - ground_truths: An array of ground truth labels of shape (n,)
    - top3_indices: An array of top-3 predicted indices of shape (n, 3)

    Returns:
    - The top-3 accuracy as a float.
    """
    # Ensure ground_truths is a numpy array for compatibility with advanced indexing
    ground_truths = np.array(ground_truths)
    top3_indices = np.array(top3_indices)
    all_score = []
    for i in range(len(ground_truths)):
        if ground_truths[i] in top3_indices[i]:
            all_score.append(1)
        else:
            all_score.append(0)
    # Check if ground truth is in top-3 predictions for each sample
    # Here, we're looking across each row (axis=1) of top3_indices
    # correct_predictions = np.any(top3_indices == ground_truths[:, None], axis=1)

    # Calculate accuracy
    top3_accuracy = np.mean(all_score)

    return top3_accuracy

def plot_f1_scores(f1_scores, model_name):
    """
    Plots the F1 scores for each class.

    Parameters:
    - f1_scores: Array of F1 scores for each class.
    - model_name: Name of the model used.
    """
    plt.figure(figsize=(12, 8))
    classes = np.arange(1, len(f1_scores) + 1)
    plt.bar(classes, f1_scores, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Scores per Class for {model_name}')
    plt.xticks(classes)
    plt.ylim([0, 1])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"f1_{model_name}.pdf")
    plt.close()

def plot_f1_scores_with_fragments(f1_scores, model_name, fragment_counts):
    """
    Plots the F1 scores against fragment counts in a scatter plot.

    Parameters:
    - f1_scores: Array of F1 scores for each class.
    - model_name: Name of the model used.
    - fragment_counts: Dictionary mapping fragment number to fragment counts.
    """
    # Prepare data_paths for plotting
    classes = np.arange(1, len(f1_scores) + 1)
    counts = [fragment_counts.get(fragment, 0) for fragment in classes]
    correlation = np.corrcoef(counts, f1_scores)[0, 1]

    plt.figure(figsize=(12, 8))
    plt.scatter(counts, f1_scores, color='darkcyan', edgecolor='black')
    plt.xlabel('Fragment Size')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Scores vs Fragment Counts for {model_name}\nPearson Correlation: {correlation:.2f}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"f1_vs_count_{model_name}.pdf")
    plt.ylim([0, 1])
    plt.close()

# Adjusted Function to calculate and display metrics
def calculate_and_display_metrics(pdb_data, expected_model_names, fragment_counts):
    aggregated_predictions = {}
    aggregated_ground_truths = []

    for model_name in expected_model_names:
        aggregated_predictions[model_name] = {}
        aggregated_predictions[model_name]["prediction"] = []
        aggregated_predictions[model_name]["top_3"] = []

    for pdb_code, data in pdb_data.items():
        ground_truth = data["ground_truth"]
        model_names = data.keys()

        # Check if this PDB entry has predictions from all expected models
        if not all(model in model_names for model in expected_model_names):
            print(f"Skipping PDB {pdb_code} due to missing model(s).")
            continue

        # Aggregate data_paths for this PDB entry
        aggregated_ground_truths += ground_truth
        for model_name in expected_model_names:
            aggregated_predictions[model_name]["prediction"].extend([data[model_name]])
            aggregated_predictions[model_name]["top_3"].append(
                [data[model_name + "_3"]]
            )

    # Calculate metrics for each model
    for model_name in aggregated_predictions.keys():
        predictions = aggregated_predictions[model_name]["prediction"]
        top3_predictions = aggregated_predictions[model_name]["top_3"]
        predictions = np.array(
            predictions
        ).flatten()  # Ensure predictions is a flat list

        accuracy = accuracy_score(aggregated_ground_truths, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            aggregated_ground_truths, predictions, average="macro", zero_division=0
        )
        top3_accuracy = calculate_top3_accuracy(
            aggregated_ground_truths, top3_predictions
        )
        print(
            f"Model: {model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Top-3 Accuracy: {top3_accuracy:.4f}"
        )
        # Plot F1 scores
        _, _, f1_scores, _ = precision_recall_fscore_support(aggregated_ground_truths, predictions, average=None, zero_division=0)
        plot_f1_scores(f1_scores, model_name)
        plot_f1_scores_with_fragments(f1_scores, model_name, fragment_counts)

        cm = confusion_matrix(
            aggregated_ground_truths, predictions, labels=range(1, 41)
        )

        # Normalize the confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(9, 8))
        sns.heatmap(
            cm_normalized,
            cmap="viridis",
            cbar_kws={"label": "Normalized Frequency"},
            fmt=".2f",
            vmin=0,
            vmax=1,
            xticklabels=range(1, 41),
            yticklabels=range(1, 41),
        )
        plt.title(f"Confusion Matrix for {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        # Save the plot
        plt.savefig(f"confusion_matrix_{model_name}.pdf")
        plt.close()


# Main execution
if __name__ == "__main__":
    metrics_path = Path("../../data/metrics")
    assert metrics_path.exists()
    pdb_data = aggregate_data_from_directory(metrics_path)
    differences_angles = ["logpr", "ramrmsd", "rms", "logprramrmsd", "logprrms", "rmsramrmsd"]
    differences_sequences = ["seqidentity", "blosum", "blosumlogprramrmsd", "blosumlogprrms", "blosumrmsramrmsd"]
    # differences_sequences = []
    differences = differences_angles + differences_sequences
    print(0)
    fragment_path = "../../data/fragments/"
    classifier = StructureToFragmentClassifier(
        Path(fragment_path),
        difference_type="angle",
        difference_name="rms",
        n_processes=10,
        step_size=1,
    )
    # Calculate the number of instances for each fragment
    fragment_counts = {}
    for k, v in classifier.fragment_dict.items():
        fragment_counts[k] = v.fragment_length
    calculate_and_display_metrics(pdb_data, differences, fragment_counts)
