<div align="center">
  <img src="img/logo.svg" alt="Tessera Logo" width="500">

**A fully vectorised Python library for protein fragment representation and comparison**

[![Paper](https://img.shields.io/badge/bioRxiv-2025.03.19.644162-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2025.03.19.644162)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
</div>

---

## Overview

Tessera is a fully vectorised Python library that decomposes protein structures into structural fragments. By default, it uses a curated vocabulary of 40 evolutionarily conserved structural fragments, enabling:

- **Fast structural comparison**: ~68× faster than RMSD-based methods (including initialisation)
- **Compact representation**: 90% reduction in memory requirements
- **Interpretability**: Biologically meaningful structural motifs
- **Dual representations**: fragment sets for fast search; fragment graphs for structure-aware comparisons

## Installation

```bash
pip install git+https://github.com/wells-wood-research/tessera
```

Or for development:

```bash
git clone https://github.com/wells-wood-research/tessera
cd tessera
pip install -e .
```

## Quick Start

### Classify a protein structure

```python
from pathlib import Path
from tessera.fragments.fragments_classifier import EnsembleFragmentClassifier

# Initialize classifier with reference fragments
classifier = EnsembleFragmentClassifier(
    fragment_path=Path("data/fragments"),
    difference_names=["logpr", "RamRmsd"]
)

# Classify structure
result = classifier.classify_to_fragment("protein.pdb")

# View fragment assignments
for frag in result.classification_map:
    print(f"Fragment {frag.fragment_class}: {frag.start_idx}-{frag.end_idx}")
```

### Compute structural similarity

**0. Prepare graphs**

First, convert your classification results into graphs and ensure attributes are formatted correctly for comparison.

```python
import gmatch4py as gm
from tessera.fragments.fragments_graph import StructureFragmentGraph

# Convert classification results to NetworkX graphs
# (assuming result_a and result_b are outputs from classifier.classify_to_fragment)
graph_a = StructureFragmentGraph.from_structure_fragment(result_a).graph
graph_b = StructureFragmentGraph.from_structure_fragment(result_b).graph

# Pre-processing: Convert attributes to strings for gmatch4py
for g in [graph_a, graph_b]:
    for _, data in g.nodes(data=True):
        data["fragment_class"] = str(data["fragment_class"])
    for _, _, data in g.edges(data=True):
        data["peptide_bond"] = str(data["peptide_bond"])
```
**Option 1: BagOfNodes (fast)**

```python
# Initialize BagOfNodes comparator
bon = gm.BagOfNodes()
bon.set_attr_graph_used(node_attr_key="fragment_class", edge_attr_key="peptide_bond")

# Calculate distance
similarity_matrix = bon.compare([graph_a, graph_b], None)
distance = bon.distance(similarity_matrix)[0, 1]

print(f"BagOfNodes Distance: {distance:.4f}")
```

**Option 2: Fragment graphs (topology-aware)**

```python
# Initialize GED with equal edit costs (node_del, node_ins, edge_del, edge_ins)
ged = gm.GraphEditDistance(1, 1, 1, 1)
ged.set_attr_graph_used(node_attr_key="fragment_class", edge_attr_key="peptide_bond")

# Calculate distance
similarity_matrix = ged.compare([graph_a, graph_b], None)
distance = ged.distance(similarity_matrix)[0, 1]

print(f"Graph Edit Distance: {distance:.4f}")
```

## How it works

Tessera is a library for representing proteins as fragments. It is library-agnostic, allowing users to define custom structural motifs. By default, it uses a curated vocabulary of 40 structural fragments (9-37 residues each) identified by [Alva et al. (2015)](https://elifesciences.org/articles/09410):

1. **Feature extraction**: Compute backbone torsion angles (φ, ψ).
2. **Convolution**:A sliding window algorithm scans the backbone to score similarity against each reference fragment.
3. **Classification**: Assign non-overlapping fragment regions based on distance thresholds.
4. **Representation**: Generate fragments set or graph.

![Fragment-based protein representation](img/fragment_overview.png)

*Figure 1: Fragment-based protein representation of the ZIF268 Zinc Finger (PDB: 1AAY). The structure is represented as a Fragment Graph, which preserves connectivity via peptide bonds (dark edges) and spatial proximity (dotted edges), or as a Fragment Set with unique fragment types. DNA is shown in yellow; zinc ions in purple.*

## Performance

### Query Speed (100 proteins, 35 cores)

| Method | 1 Query | 10 Queries | 100 Queries | Memory |
|--------|---------|------------|-------------|--------|
| **BagOfNodes** | **0.001s** | **0.008s** | **0.07s** | **4.87** ± 2.48 |
| **GraphEditDistance** | 5.65s | 57.17s | 573.05s | 16.39 ± 13.57 |
| BLOSUM | 0.42s | 3.44s | 36.57s | 290.73 ± 217.82 |
| RMSD | 22.25s | 155.38s | 1717.03s | 1744.36 ± 1306.90 |

*Memory = avg. data points per protein (backbone atoms for RMSD, residues for BLOSUM, nodes/elements for ours)*

### Clustering Quality (PFD dataset: 215 proteins, 12 classes)

| Method | ARI ↑ | NMI ↑ | Silhouette ↑ | F1 Score ↑ |
|--------|-------|-------|--------------|------------|
| **BagOfNodes** | 0.005 | 0.346 | **0.823** | 0.166 |
| **GraphEditDistance** | **0.046** | **0.383** | 0.077 | **0.199** |
| BLOSUM | 0.003 | 0.293 | 0.025 | 0.164 |
| RMSD | 0.036 | 0.386 | -0.033 | 0.196 |

*Evaluated using GMM on PCoA embeddings*

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_probability_processing.py -v

# Run with coverage
pytest tests/ --cov=tessera --cov-report=html
```

## Citation

```bibtex
@article{Castorina2025.03.19.644162,
    author = {Castorina, Leonardo V. and Wood, Christopher W. and Subr, Kartic},
    title = {From Atoms to Fragments: A Coarse Representation for Efficient and Functional Protein Design},
    year = {2025},
    doi = {10.1101/2025.03.19.644162},
    journal = {bioRxiv}
}
```
