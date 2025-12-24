# Tessera

**A learned structural vocabulary for efficient protein representation and comparison**

[![Paper](https://img.shields.io/badge/bioRxiv-2025.03.19.644162-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2025.03.19.644162)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

Tessera decomposes protein structures into a vocabulary of 40 evolutionarily conserved structural fragments, enabling:

- **Fast structural comparison** — 68× faster than RMSD-based methods (including initialisation)
- **Compact representation** — 90% reduction in memory requirements
- **Interpretability** — Biologically meaningful structural motifs
- **Dual representations** — Bag-of-fragments for fast search; fragment graphs for structure-aware comparisons

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

**Option 1: Bag-of-fragments (fast)**

```python
from collections import Counter

# Extract composition
fragments_a = Counter([f.fragment_class for f in result_a.classification_map])
fragments_b = Counter([f.fragment_class for f in result_b.classification_map])

# Jaccard similarity
def jaccard(a, b):
    keys = set(a.keys()) | set(b.keys())
    intersection = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return intersection / union

similarity = jaccard(fragments_a, fragments_b)
```

**Option 2: Fragment graphs (topology-aware)**

```python
from tessera.fragments.fragments_graph import StructureFragmentGraph

# Build graph representation
graph = StructureFragmentGraph.from_structure_fragment(
    result,
    edge_distance_threshold=10.0  # Ångströms
)

# Graph Edit Distance (requires gmatch4py)
from gmatch4py.ged import GraphEditDistance

ged = GraphEditDistance(node_del=1, node_ins=1, edge_del=1, edge_ins=1)
ged.set_attr_graph_used("fragment_class", None)
distance = ged.compare([graph_a.graph, graph_b.graph], None)
```

## How it works

Tessera uses a learned vocabulary of 40 structural fragments (9-37 residues each) identified through evolutionary analysis:

1. **Feature extraction** — Compute backbone torsion angles (φ, ψ)
2. **Convolution** — Score similarity to each reference fragment
3. **Classification** — Assign non-overlapping fragment regions
4. **Representation** — Generate bag-of-fragments or graph

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

## License

MIT License - see [LICENSE](LICENSE) file for details.
