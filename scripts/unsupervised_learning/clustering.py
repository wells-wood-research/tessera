import argparse
from pathlib import Path
import gmatch4py as gm
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from src.fragments.fragments_graph import StructureFragmentGraph


def main(args):
    assert args.input_path.exists(), f"Input file {args.input_path} does not exist"
    # Find all files with .fg file extension
    fragment_files = list(args.input_path.glob("*.fg"))
    # Assert the list is not empty
    assert fragment_files, f"No files found in {args.input_path} with .fg extension"
    nx_graphs = []
    graph_categories = []
    # Iterate over the fragment files and extract the graph and category
    for fg in fragment_files:
        curr_design_name, curr_category = fg.stem.split("_")
        graph_categories.append(curr_category)
        fragment_graph = StructureFragmentGraph.load(fg).graph
        # Iterate through nodes and convert fragment_class to str
        for node in fragment_graph.nodes(data=True):
            node[1]["fragment_class"] = str(node[1]["fragment_class"])
        # Iterate through edges and convert peptide_bond to str
        for edge in fragment_graph.edges(data=True):
            edge[2]["peptide_bond"] = str(edge[2]["peptide_bond"])
        nx_graphs.append(fragment_graph)

    # Encode categories to integers for evaluation and coloring
    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(graph_categories)
    category_names = label_encoder.classes_

    # Create a color mapping for categories
    cmap = plt.get_cmap('tab10')
    category_colors = {category: cmap(i / len(category_names)) for i, category in enumerate(category_names)}

    # List of metrics to use
    metrics_list = [
        gm.BagOfNodes,
        gm.WeisfeleirLehmanKernel,
        gm.GraphEditDistance,
        gm.GreedyEditDistance,
        gm.Jaccard,
        gm.MCS,
        gm.VertexEdgeOverlap,
    ]

    evaluation_scores = []

    for class_ in tqdm(metrics_list, desc="Metrics"):
        start_time = time.time()

        # Initialize comparator with appropriate parameters
        if class_ in (gm.GraphEditDistance, gm.GreedyEditDistance):
            comparator = class_(1, 1, 1, 1)  # All edit costs are equal to 1
        elif class_ == gm.WeisfeleirLehmanKernel:
            comparator = class_(h=2)
        else:
            comparator = class_()

        # Set attributes used in comparison
        comparator.set_attr_graph_used(
            node_attr_key="fragment_class", edge_attr_key="peptide_bond"
        )

        # Compute distance or similarity matrix
        result = comparator.compare(nx_graphs, None)

        distance_matrix = comparator.distance(result)
        distance_matrix = np.array(distance_matrix)

        # IF GED Symmetrize the distance matrix
        # if class_ in (gm.GraphEditDistance, gm.GreedyEditDistance):
        distance_matrix = (distance_matrix + distance_matrix.T) / 2

        # Perform Hierarchical Clustering
        linked = linkage(distance_matrix, 'average')

        # Plot Dendrogram
        plt.figure(figsize=(10, 7))

        # Define a function to label leaves with category names
        def llf(idx):
            return graph_categories[int(idx)]

        dendro = dendrogram(
            linked,
            labels=[str(i) for i in range(len(graph_categories))],
            orientation='right',  # Changed to 'right' for left-to-right dendrogram
            distance_sort='descending',
            show_leaf_counts=False,
            leaf_label_func=llf,
        )

        # Color the leaf labels based on categories
        ax = plt.gca()
        ylbls = ax.get_ymajorticklabels()
        for lbl in ylbls:
            lbl_text = lbl.get_text()
            category = lbl_text
            lbl.set_color(category_colors[category])

        title = f"Dendrogram for {class_.__name__}\nTime: {time.time() - start_time:.2f}s"
        plt.title(title, fontsize=16)
        plt.xlabel('Distance', fontsize=14)
        plt.ylabel('Graphs', fontsize=14)
        plt.tight_layout()

        # Save dendrogram as PDF
        dendrogram_filename = f"{class_.__name__}_dendrogram.pdf"
        plt.savefig(dendrogram_filename)
        plt.close()

        # Cut the dendrogram to form flat clusters
        predicted_labels = fcluster(linked, t=len(set(graph_categories)), criterion='maxclust')

        # Adjust predicted_labels to start from 0
        predicted_labels -= 1

        # Evaluate clustering
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        time_taken = time.time() - start_time

        evaluation_scores.append(
            {"metric": class_.__name__, "ari": ari, "nmi": nmi, "time": time_taken}
        )

        # Perform Multidimensional Scaling for 2D visualization
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
        )
        coords = mds.fit_transform(distance_matrix)

        # Plot the results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            coords[:, 0], coords[:, 1], c=true_labels, cmap="tab10", s=100
        )
        title = f"Graph Visualization by Category\nMetric: {class_.__name__}, Time: {time_taken:.2f}s, ARI: {ari:.3f}, NMI: {nmi:.3f}"
        plt.title(title, fontsize=16)
        plt.xlabel("MDS Dimension 1", fontsize=14)
        plt.ylabel("MDS Dimension 2", fontsize=14)

        # Annotate points with their categories
        for i, txt in enumerate(graph_categories):
            plt.annotate(txt, (coords[i, 0], coords[i, 1]), fontsize=9, ha="right")

        # Create a legend mapping colors to categories
        handles = []
        for i, category in enumerate(category_names):
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    color="w",
                    label=category,
                    markerfacecolor=plt.cm.tab10(i / len(category_names)),
                    markersize=10,
                )
            )
        plt.legend(
            handles=handles,
            title="Categories",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        plt.grid(True)
        plt.tight_layout()

        # Save plot as PDF
        filename = f"{class_.__name__}_visualization.pdf"
        plt.savefig(filename)
        plt.close()

    # Print evaluation scores
    print("\nClustering Evaluation Scores:")
    for score in evaluation_scores:
        print(
            f"Metric: {score['metric']}, Time: {score['time']:.2f}s, ARI: {score['ari']:.3f}, NMI: {score['nmi']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Visualization by Category")
    parser.add_argument("--input_path", type=Path, help="Path to input files")
    params = parser.parse_args()
    main(params)