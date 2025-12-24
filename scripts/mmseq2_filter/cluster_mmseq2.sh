#!/bin/bash

# Path to the current directory
BASE_DIR=$(pwd)

# Create the database from the FASTA file
docker run -it --rm -v ${BASE_DIR}:/data soedinglab/mmseqs2 mmseqs createdb /data/sequences.fasta /data/sequencesDB

# Cluster the sequences
docker run -it --rm -v ${BASE_DIR}:/data soedinglab/mmseqs2 mmseqs cluster /data/sequencesDB /data/clusterRes /data/tmp --min-seq-id 0.4

# Convert the clustering results to a tab-separated file
docker run -it --rm -v ${BASE_DIR}:/data soedinglab/mmseqs2 mmseqs createtsv /data/sequencesDB /data/sequencesDB /data/clusterRes /data/clusterRes.tsv
