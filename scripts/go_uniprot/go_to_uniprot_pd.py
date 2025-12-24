import argparse
import pandas as pd
from multiprocessing import Pool, cpu_count

def process_chunk(chunk):
    # Select relevant columns, set CHAIN to 'A', and rename them
    chunk_filtered = chunk[['DB_Object_ID', 'GO_ID']].copy()
    chunk_filtered['CHAIN'] = 'A'
    chunk_filtered = chunk_filtered.rename(columns={'DB_Object_ID': 'UNIPROT', 'GO_ID': 'GO_ID'})
    return chunk_filtered[['UNIPROT', 'CHAIN', 'GO_ID']]

def process_gaf(gaf_file, output_file, chunk_size):
    # Read the GAF file and extract metadata
    with open(gaf_file, 'r') as file:
        metadata = [line for line in file if line.startswith('!')]
    
    # Write metadata to the output file
    with open(output_file, 'w') as file:
        file.write(''.join(metadata) + '\n')

    # Read the GAF data in chunks
    chunk_iter = pd.read_csv(gaf_file, comment='!', delimiter='\t', header=None, 
                             names=["DB", "DB_Object_ID", "DB_Object_Symbol", "Qualifier", "GO_ID",
                                    "DB:Reference", "Evidence_Code", "With_or_From", "Aspect", 
                                    "DB_Object_Name", "DB_Object_Synonym", "DB_Object_Type", 
                                    "Taxon", "Date", "Assigned_By", "Annotation_Extension", "Gene_Product_Form_ID"],
                             chunksize=chunk_size)
    
    # Initialize multiprocessing pool
    with Pool(cpu_count()) as pool:
        for chunk in chunk_iter:
            # Process each chunk in parallel
            filtered_chunk = pool.apply(process_chunk, args=(chunk,))
            # Append processed chunk to the output file
            filtered_chunk.to_csv(output_file, mode='a', header=False, index=False)

def main():
    parser = argparse.ArgumentParser(description='Process GAF file and convert to CSV format.')
    parser.add_argument('gaf_file', type=str, help='Input GAF file path')
    parser.add_argument('output_file', type=str, help='Output CSV file path')
    parser.add_argument('--chunk-size', type=int, default=100000, help='Number of rows per chunk (default: 100000)')
    
    args = parser.parse_args()
    
    process_gaf(args.gaf_file, args.output_file, args.chunk_size)

if __name__ == "__main__":
    main()
