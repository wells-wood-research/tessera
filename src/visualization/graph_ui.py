import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import py3Dmol
import seaborn as sns
import streamlit as st
import torch
import torch.nn.functional as F
from pyvis.network import Network
from stmol import showmol
from streamlit.components.v1 import html

from src.difference_fn.difference_processing import (
    StructureConvolutionOperator,
    select_first_ampal_assembly,
)
from src.fragments.fragments_graph import StructureFragmentGraph
from src.training.data_processing.dataset import create_interpro_encoder
from src.training.train import GraphLevelGNN


# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Streamlit Graph Prediction App")
    parser.add_argument(
        "--fg_base_path",
        type=str,
        required=True,
        help="Base path to directory containing .fg files",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the PyTorch model file"
    )
    return parser.parse_args()


# Function to load and process the .fg file
def load_fg_file(file_path):
    structure_fragment_graph = StructureFragmentGraph.load(file_path)
    return structure_fragment_graph


# Function to load the PyTorch model
def load_model(model_path):
    model = GraphLevelGNN.load_from_checkpoint(model_path)
    model.eval()  # Set model to evaluation mode
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


# Function to get predictions from the model
def get_predictions(model, G):
    node_attributes = [
        "angles_data",
        "fragment_class",
        "percentage_index",
        "probability_data",
    ]
    edge_attributes = ["peptide_bond", "euclidean_distance"]

    # Convert to PyTorch Geometric data
    g_input = G.to_pyg(node_attributes=node_attributes, edge_attributes=edge_attributes)
    g_input = g_input.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Ensure the model is in evaluation mode

    # Pass through the model
    with torch.no_grad():  # Disable gradient computation
        logits = model(g_input)[
            0
        ]  # Obtain logits, assuming your model returns a tuple (logits, targets)

    return logits


# Function to find .fg files in the directory structure
def find_fg_files(base_path):
    fg_files = [
        str(fg_file.relative_to(base_path)) for fg_file in base_path.rglob("*.fg")
    ]
    return fg_files


# Function to load structure data
# Function to load gzipped structure data
def load_structure(file_path):
    ampal_structure = StructureConvolutionOperator._load_structure(file_path)
    ampal_structure = select_first_ampal_assembly(ampal_structure)
    structure_data = ampal_structure.pdb
    return structure_data


def visualize_structure(
    structure_data, fragment_details, attention_weights, head_index
):
    # Initialize a Py3Dmol viewer
    viewer = py3Dmol.view(width=800, height=600)
    viewer.addModel(structure_data, "pdb")  # Load the PDB structure
    viewer.setStyle({"cartoon": {"color": "white"}})  # Set the default color

    # Extract attention weights for the selected head
    attention_weights = (
        attention_weights[0][1][:, head_index].flatten().cpu().detach().numpy()
    )

    # Normalize the attention weights
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = cm.Blues

    # Apply color to each fragment based on its attention weight
    for i, detail in enumerate(fragment_details):
        start = detail.start_idx
        end = detail.end_idx
        # Calculate the average attention value for the fragment
        attention_value = attention_weights[i]
        color = matplotlib.colors.to_hex(cmap(norm(attention_value)))
        viewer.setStyle(
            {"chain": "A", "resi": f"{start + 1}-{end + 1}"},
            {"cartoon": {"color": color}},
        )

    viewer.zoomTo()  # Zoom to the structure
    return viewer


# Streamlit UI for graph visualization
def main():
    # Parse command-line arguments
    args = parse_arguments()
    fg_base_path = Path(args.fg_base_path)
    model_path = Path(args.model_path)

    st.title("Graph Prediction Interface")

    fg_files = find_fg_files(fg_base_path)
    selected_file = st.selectbox("Select a .fg file", fg_files)

    if selected_file:
        G = load_fg_file(fg_base_path / selected_file)

        for node, data in G.graph.nodes(data=True):
            G.graph.nodes[node].update(convert_to_serializable(data))

        for u, v, data in G.graph.edges(data=True):
            G.graph.edges[u, v].update(convert_to_serializable(data))

        G.graph, color_hex = color_peptide_bonds(G.graph)

        model = load_model(model_path)
        st.write("Model Loaded Successfully.")
        logits = get_predictions(model, G)
        probabilities = F.softmax(logits, dim=1).cpu().detach().numpy()

        color_option = st.selectbox(
            "Color nodes based on", ["None", "Attention Weights"]
        )

        if color_option == "Attention Weights":
            attention_weights = model.get_node_attention_weights() # N_heads, (2, edge_index) (edge_weight, N_heads)
            n_heads = attention_weights[0][1].shape[1]
            head_options = [
                f"Head {i}" for i in range(n_heads)
            ]
            # # st.write(G)
            # st.write(attention_weights)
            # # st.write(attention_weights[0][1].shape)
            # # st.write("\n")
            # # st.write(attention_weights)
            # # Aggregate attention weights
            # # TODO Fix first dimension to be the head
            # n_nodes = G.graph.number_of_nodes()
            # node_weights = torch.empty(n_nodes, n_nodes, n_heads)
            # edge_index = attention_weights[0][1]
            # for node in range(n_nodes):
            #     # Find all indeces where the node is the source
            #     source_indeces = torch.where(edge_index[0] == node)
            #     target_indeces = edge_index[1][source_indeces]
            #     node_weights[node, target_indeces] = attention_weights[0][0][source_indeces]
            #     break
            selected_head = st.selectbox("Select Attention Head", head_options)
            selected_head_index = head_options.index(selected_head)

            attention_weights_selected_head = (
                attention_weights[0][1][:, selected_head_index].cpu().detach().numpy()
            )
            G.graph = color_nodes_based_on_attention(
                G.graph, attention_weights_selected_head
            )

            fragment_details = G.structure_fragment.classification_map
            structure_path = G.structure_fragment.structure_path

            structure_data = load_structure(structure_path)
            viewer = visualize_structure(
                structure_data, fragment_details, attention_weights, selected_head_index
            )

        net = Network(
            height="500px",
            width="100%",
            directed=True,
            notebook=True,
            cdn_resources="in_line",
        )

        net.set_options(
            """
        var options = {
          "nodes": {
            "shape": "dot",
            "size": 20,
            "font": {
              "size": 16,
              "color": "black",
              "face": "arial",
              "align": "center"
            },
            "borderWidth": 2,
            "borderWidthSelected": 2
          },
          "edges": {
            "color": {
              "inherit": false
            },
            "smooth": {
              "type": "continuous"
            },
            "font": {
              "size": 14,
              "color": "black",
              "align": "middle",
              "strokeWidth": 1,
              "strokeColor": "white"
            },
            "width": 2,
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.5
              }
            }
          },
          "physics": {
            "enabled": false,
            "stabilization": {
              "enabled": true,
              "iterations": 1000
            },
            "barnesHut": {
              "gravitationalConstant": -1000,
              "centralGravity": 0.1,
              "springLength": 150,
              "springConstant": 0.01,
              "damping": 0.09
            }
          }
        }
        """
        )

        for node, data in G.graph.nodes(data=True):
            node_label = data.get("label", node)
            if color_option == "Attention Weights":
                attention_value = attention_weights_selected_head[node]
                norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
                cmap = cm.Blues
                color = matplotlib.colors.to_hex(cmap(norm(attention_value)))
            else:
                color = "#FFFFFF"
            net.add_node(node, label=node_label, title=str(data), color=color)

        for u, v, data in G.graph.edges(data=True):
            color = data.get("color", "#aaaaaa")
            width = data.get("width", 1)
            label = data.get("label", "")
            net.add_edge(u, v, color=color, width=width, label=label)

        net_html = net.generate_html()
        html(net_html, height=600)

        fig, ax = plt.subplots(figsize=(10, 1))
        cmap = cm.rainbow
        norm = matplotlib.colors.Normalize(vmin=1, vmax=len(color_hex))
        cbar = matplotlib.colorbar.ColorbarBase(
            ax, cmap=cmap, norm=norm, orientation="horizontal"
        )
        cbar.set_ticks(np.linspace(1, len(color_hex), len(color_hex)))
        cbar.set_ticklabels(range(1, len(color_hex) + 1))
        ax.set_title("Peptide Bond Index")
        st.pyplot(fig)
        if color_option == "Attention Weights":
            showmol(viewer, height=500, width=800)
            print(fragment_details)
            st.write(fragment_details)

        go_dictionary = {
            "GO:0046872": "GO:metal ion binding",
            "GO:0006508": "GO:proteolysis",
            "GO:0003723": "GO:RNA binding",
            "GO:0003677": "GO:DNA binding",
            "GO:0016020": "GO:membrane",
        }
        input_labels = list(go_dictionary.keys())
        label_dict = create_interpro_encoder(input_labels)
        labels = [go_dictionary[label] for label in input_labels]

        plt.figure(figsize=(10, 5))
        ax = sns.heatmap(
            probabilities,
            annot=True,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=[""],
        )

        ax.set_ylabel("")
        plt.title("Prediction Probabilities")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(plt.gcf())


# Helper function to convert various types to JSON serializable formats
def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


# Function to color and style peptide bond edges and label nodes
def color_peptide_bonds(G):
    peptide_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("peptide_bond", False)
    ]
    num_peptide_edges = len(peptide_edges)
    colors = [cm.rainbow(i / num_peptide_edges) for i in range(num_peptide_edges)]
    color_hex = [matplotlib.colors.to_hex(c) for c in colors]

    for idx, (u, v) in enumerate(peptide_edges):
        G[u][v]["color"] = color_hex[idx]
        G[u][v]["width"] = 6
        G[u][v]["label"] = f"{idx+1}"

    return G, color_hex


# Function to color nodes based on attention weights
def color_nodes_based_on_attention(G, attention_weights):
    norm = matplotlib.colors.Normalize(
        vmin=min(attention_weights), vmax=max(attention_weights)
    )
    cmap = cm.Blues

    for node in G.nodes:
        weight = attention_weights[node]
        color = cmap(norm(weight))
        G.nodes[node]["color"] = matplotlib.colors.to_hex(color)

    return G


# Run the main function if this script is executed
if __name__ == "__main__":
    main()
