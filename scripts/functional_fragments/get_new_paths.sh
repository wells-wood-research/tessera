#!/bin/bash

# Define the directory to search and the output file
search_dir="$1"  # First argument as the directory to search
output_file="matching_files.txt"

declare -a keys=('3FD6' '6TUN' '7WTW' '7RAH' '5GL7' '7E2M' '2NWH' '1DAG' '8H7K' '5MW8' '2RUK' '3OC3' '5EOG' '7T8M' '4OHZ' '8ASA' '6ADM' '3KUJ' '6L1R' '7OBK' '1R5B' '7ME5' '8RFJ' '5VZ5' '3TDH' '5M0J' '3AQ4' '8UE1' '6V4H' '2CKC' '4JNH' '7ULE' '3JRV' '6C46' '5CFD' '1FP0' '4YKY' '2X72' '7QEP' '6VP8' '8OWO' '3BBU' '5NVB' '6B86' '8Q6Q' '1ZAE' '5JB3' '6EGX' '1OB9' '6RO1' '4Z4G' '6WQ1' '7NOK' '5O3N' '5KH0' '7T9Y' '7LIU' '4B8O' '3ZXO' '6WMP' '2O26' '6XK9' '1Z9B' '3SZR' '4MHC' '7D8U' '7UAA' '6JIG' '7U8X' '6GM5' '5M3L' '7RQG' '8G54' '2XKA' '5TVP' '6Y32' '5HH7' '1GVN' '1QHH' '1YNX' '6T8O' '6LXG' '1RFL' '1IWF' '2K8F' '6JTO' '6Y6F' '6JTG' '5D1W' '3FLD' '3KD8' '2V0E' '1Z0R' '8OP3' '2VDW' '8GRQ' '8BI0' '3U1I' '1YRA' '1T5G' '1Q2Z' '6FGZ' '7PHT' '3QO3' '7V9B' '1N1A' '2BBP' '2FOM' '6SY2' '8J07' '8G28' '1Z9I' '5VQI' '2LY4' '2NVU' '5S5A' '7D63' '8WBC' '2DGY' '7JGR' '2JQI' '2JMX' '7DOJ' '3DD7' '4PKD' '2KWU' '3J9X' '8AFG' '1ZUN' '2MC3' '2FF4' '5J28' '7YUG' '8CMG' '4ZVB' '2A1A' '3UAI' '4QQ4' '1BS2' '5AOO' '8FFV' '4HKQ' '1VJ7' '2OH5' '2M26' '4QN1' '3HJ7' '2XTP' '7ZES' '6IY8' '4P22' '2DYF' '6HZ4' '7VRC' '8EFT' '1FAF' '5MJV' '4NGG' '8D8L' '2AYX' '5JPX' '3SLC' '3IUO' '5C2K' '4H1G' '7MCA' '4AB7' '2N5N' '3LDU' '7JIJ' '4ZIA' '4RPT' '6NON' '6J4V' '3MDO' '5YP8' '5IL7' '7Q6Z' '6JDR' '7QST' '6WBE' '8SRO' '8GHO' '2RMS' '1QVP' '6IFC' '7WGE' '8I8G' '3LLU' '4OX9' '7W5B' '4G3A' '8TZX' '6U07' '3OPY' '7NQC' '2AJM' '3RVF' '6D2S' '2LIF' '2P8Q' '4BOS' '1ZQV' '3QP8' '4UCL' '7CRV' '8OH2' '4M59' '5NQ5' '6NJ0' '3WE2' '6NX5' '6QP0' '4KW3' '5HD9' '8FKR' '3UI2' '4P0A' '4QUU' '2AYA' '2FXA' '3MVP' '8FKS' '2J8P' '8S9X' '1VCD' '1WE2' '4K8B' '7JSH' '4LNR' '4Z31' '3IBV' '1ODF' '4AT9' '1V1C' '1WXQ' '4A4K' '7DL8' '3KUI' '1X9D' '4EDK' '3SJ9' '4GFB' '7TAG' '5LAD' '3REB' '4DNI' '3KYH' '5FRY' '2HUG' '2QTH' '3VOX' '4AZA' '1XZP' '2YQP' '2DYN' '4O64' '2ZWI' '8HRF' '3AL3' '2GOY' '2M4H' '1M9I' '3LDF' '6KSQ' '7EWM' '5WLC' '5V9P' '2RHK' '3GP8' '7D72' '7K5I' '7CR9' '7JNE' '3P88' '4OP0' '6FC1' '7QG6' '7C0J' '4L87' '1WH9' '2KNU' '5MEI' '1TV7' '7BGM' '2MWO' '6E1J' '3VRP' '2OXC' '7Y0D' '2LW7' '2BBW' '7TM8' '2QPT' '2HR7' '5UGW' '7K3S' '8AT8' '2JQ6' '1MBM' '2N5M' '5UJM' '1K7J' '7JYA' '4WNR' '4FF3' '2BMJ' '1QHL' '2E6I' '2B0L' '8H2N' '6FSZ' '5WX1' '5A2Z' '1DSV' '7Q0F' '5FYQ' '2W3O' '7P3B' '2E4H' '3CFV' '2H5G' '3UI6' '7R7C' '2GUI' '3CRV' '8Q3K' '6EP3' '7D48' '5JIU' '2YVQ' '6PV0' '6YM1' '3RZX' '3M9G' '1A17' '8A2F' '2DEI' '6RXV' '2KTF' '3HYR' '1JWY' '3ZOJ' '4N8M' '4C0D' '6JMG' '3HTK' '2HEK' '5KSP' '6VDE' '2B2W' '6C9N' '3F2O' '5WTH' '1Y88' '4YN8' '1HHH' '2WMC' '6K71' '6HLW' '1UFI' '1AB2' '1QA7' '4ZXN' '8R0B' '2MBV' '5CK3' '7ND2' '3M65' '6H9M' '4JXT' '6ITU' '2Y4V' '8PNT')

# Clear or create the output file
> "$output_file"

# Loop through each key and search for matching files
for key in "${keys[@]}"; do
    # Find files in the specified directory (and subdirectories) that match the key pattern
    # Assumes filenames contain the exact key as a substring
    find "$search_dir" -type f -name "*$key*" >> "$output_file"
done

# Notify the user about the results
echo "Search complete. Matching file paths are saved in $output_file."