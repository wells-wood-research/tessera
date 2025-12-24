import argparse
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, wait
import dask
import logging
import gc
from distributed import config

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger()

def parse_gaf(gaf_file, output_file, chunk_size):
    logger = logging.getLogger()

    dask.config.set({
        'distributed.worker.memory.target': 0.6,
        'distributed.worker.memory.spill': 0.7,
        'distributed.worker.memory.pause': 0.8,
        'distributed.worker.memory.terminate': 0.9,
        'temporary-directory': '/scratch/protein_graphs/tmp/'
    })

    config.update({
        "distributed.comm.timeouts.connect": "60s",
        "distributed.comm.timeouts.tcp": "120s"
    })

    logger.info("Starting Dask cluster.")
    cluster = LocalCluster(
        n_workers=30,
        threads_per_worker=2,
        memory_limit='1GB'
    )
    
    with Client(cluster) as client:
        try:
            logger.info("Reading GAF file metadata.")

            logger.info("Loading GAF file into Dask DataFrame.")
            df = dd.read_csv(gaf_file, comment='!', delimiter='\t', header=None,
                             blocksize=chunk_size,
                             usecols=[1, 4],
                             names=["DB", "DB_Object_ID", "DB_Object_Symbol", "Qualifier", "GO_ID",
                                    "DB:Reference", "Evidence_Code", "With_or_From", "Aspect",
                                    "DB_Object_Name", "DB_Object_Synonym", "DB_Object_Type",
                                    "Taxon", "Date", "Assigned_By", "Annotation_Extension", "Gene_Product_Form_ID"])

            logger.info("Filtering relevant columns.")
            df_filtered = df[['DB_Object_ID', 'GO_ID']].map_partitions(
                lambda df: df.assign(CHAIN='A').rename(columns={'DB_Object_ID': 'UNIPROT'})
            )

            logger.info("Saving filtered data to output file.")
            df_filtered.to_csv(output_file, single_file=True, mode='a', index=False)
            logger.info("Data saving complete.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            logger.info("Shutting down Dask cluster.")
            client.close()
            cluster.close()
            gc.collect()

def main():
    logger = setup_logging()

    parser = argparse.ArgumentParser(description='Process GAF file and convert to CSV format.')
    parser.add_argument('gaf_file', type=str, help='Input GAF file path')
    parser.add_argument('output_file', type=str, help='Output CSV file path')
    parser.add_argument('--chunk-size', type=int, default=1000000, help='Size of each chunk in bytes (default: 1000000)')

    args = parser.parse_args()

    parse_gaf(args.gaf_file, args.output_file, args.chunk_size)

if __name__ == "__main__":
    main()
