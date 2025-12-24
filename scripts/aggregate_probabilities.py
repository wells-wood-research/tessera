from pathlib import Path

import numpy as np

metrics_path = Path("../../data/mean_metrics_single_fragment")

# List all files in the directory starting with "blosum" and ending with ".csv"
blosum_files = list(metrics_path.glob("rms_*probability_distance_data.csv"))
# From these filenames, remove "blosum_"
blosum_files = [f.stem.replace("rms_", "") for f in blosum_files]
models_to_aggregate = ["logpr", "ramrmsd"]

model_to_paths = {k: [] for k in models_to_aggregate}
# Load the files for the models to aggregate:
for blosum_file in blosum_files:
    for model in models_to_aggregate:
        path_to_file = Path(metrics_path / f"{model}_{blosum_file}.csv")
        if not path_to_file.exists():
            print(f"File {path_to_file} does not exist")
        else:
            model_to_paths[model].append(path_to_file)

        # assert path_to_file.exists(), f"File {path_to_file} does not exist"


all_models = "".join(models_to_aggregate)
# For each file, load the data_paths for each model and multiply the probabilities together to get the final probability
# for each fragment
for blosum_file in blosum_files:
    print(f"Processing {blosum_file}")
    for model in models_to_aggregate:
        print(f"  Processing {model}")
        path_to_file = Path(metrics_path / f"{model}_{blosum_file}.csv")
        assert path_to_file.exists(), f"File {path_to_file} does not exist"
        # Load the data_paths
        data = np.loadtxt(path_to_file, delimiter=",")
        # If this is the first file, initialise the final data_paths array
        if model == models_to_aggregate[0]:
            final_data = np.zeros_like(data)
        # Add the probabilities together
        final_data += data
    final_data = final_data / len(models_to_aggregate)
    # Save the final data_paths to a new file
    np.savetxt(metrics_path / f"{all_models}_{blosum_file}.csv", final_data, delimiter=",")

