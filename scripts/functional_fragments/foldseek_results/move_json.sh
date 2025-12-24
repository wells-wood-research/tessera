#!/bin/bash

# Ensure the script uses the directory where it resides as the working directory
cd "$(dirname "$0")"

# Create the 'done' directory if it doesn't exist
mkdir -p done

# Loop through all .html files in the current directory
for html_file in *.html; do
    # Extract the base name without the extension
    base_name="${html_file%.html}"
    
    # Check if the corresponding .json file exists
    if [[ -f "${base_name}.json" ]]; then
        # Move both files to the 'done' directory
        mv "$html_file" "done/"
        mv "${base_name}.json" "done/"
        echo "Moved $html_file and ${base_name}.json to the 'done' directory."
    fi
done
