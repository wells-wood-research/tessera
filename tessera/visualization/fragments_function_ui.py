import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import py3Dmol
import streamlit as st
from pyvis.network import Network
from stmol import showmol
from streamlit.components.v1 import html

from tessera.fragments.classification_config import go_to_prosite
from tessera.fragments.fragments_graph import StructureFragmentGraphIO
from tessera.function_prediction.uniprot_processing import UniprotDownloader
from tessera.visualization.fold_coverage import load_graph_creator
from tessera.visualization.graph_ui import color_peptide_bonds, convert_to_serializable


def load_pdb_files(folder: Path):
    """Load PDB files from the specified folder."""
    return list(folder.glob("*.pdb"))


def create_combined_plot(features, fragments):
    """Create a Plotly plot for protein features and fragments."""
    fig = go.Figure()

    # Plot features on the first line (y=1)
    for feature in features:
        fig.add_shape(
            type="rect",
            x0=feature["start"],
            x1=feature["end"],
            y0=1,
            y1=2,
            line=dict(color="black"),
            fillcolor=feature["color"],
            opacity=0.5,
        )
        fig.add_trace(
            go.Scatter(
                x=[(feature["start"] + feature["end"]) / 2],
                y=[1.5],
                text=[feature["name"]],
                mode="text",
                showlegend=False,
            )
        )

    # Plot fragments on the second line (y=0)
    for fragment in fragments:
        fig.add_shape(
            type="rect",
            x0=fragment["start"],
            x1=fragment["end"],
            y0=0,
            y1=1,
            line=dict(color="black"),
            fillcolor=fragment["color"],
            opacity=0.5,
        )
        fig.add_trace(
            go.Scatter(
                x=[(fragment["start"] + fragment["end"]) / 2],
                y=[0.5],
                text=[fragment["name"]],
                mode="text",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Protein Features and Fragments",
        xaxis=dict(
            title="Amino Acid Position",
            range=[
                0,
                max(
                    max([f["end"] for f in features]),
                    max([f["end"] for f in fragments]),
                )
                + 10,
            ],
        ),
        yaxis=dict(
            tickvals=[0.5, 1.5],
            ticktext=["FRAGMENTS", "FEATURES"],
            range=[-0.5, 2.5],
            visible=True,
        ),
        height=300,
        margin=dict(t=50, b=0, l=0, r=0),
    )
    return fig


def visualize_protein(
    pdb_file: str, selected_feature: dict = None, color: str = "lightblue"
):
    """Visualize the protein structure using py3Dmol, optionally highlighting a selected feature."""
    with open(pdb_file, "r") as file:
        pdb_data = file.read()

    xyzview = py3Dmol.view(data=pdb_data)
    xyzview.setStyle({"cartoon": {"color": "white"}})

    if selected_feature:
        selection_query = f"resi {selected_feature['start']}-{selected_feature['end']}"
        xyzview.setStyle({"resi": selection_query}, {"cartoon": {"color": color}})
        xyzview.addLabel(
            selected_feature["name"],
            {
                "fontColor": "black",
                "backgroundColor": color,
                "fontSize": 60,
                "position": {"resi": selected_feature["start"]},
            },
        )
        xyzview.zoomTo({"resi": selection_query})
    else:
        xyzview.zoomTo()

    showmol(xyzview, height=500, width=800)


def main(pdb_folder: str):
    pdb_folder = Path(pdb_folder)
    args.fragment_path = Path(args.fragment_path)
    assert pdb_folder.exists(), f"Folder not found: {pdb_folder}"
    assert args.fragment_path.exists(), f"Fragment classifier not found: {args.fragment_path}"

    st.title("Fragment and Annotations")
    go_colors = {"DNA binding": "purple", "RNA binding": "orange", "GTP binding": "green", "ATP binding": "red",
        "metal ion binding": "blue", }
    pru_to_go = {}
    for go_term, details in go_to_prosite.items():
        description = details["description"]
        for pru, pru_description in details["prosite"].items():
            pru_to_go[pru] = description

    # Allow the user to either input a UniProt code or select a PDB file
    uniprot_code = st.text_input("Enter UniProt code (or leave blank to select PDB file):")
    uniprot_downloader = UniprotDownloader(output_dir=pdb_folder)

    if uniprot_code:
        # Try to download the PDB file using the UniProt code
        try:
            uniprot_downloader.process_protein(uniprot_code)
            selected_pdb = uniprot_downloader.output_dir / f"{uniprot_code}.pdb"
            assert selected_pdb.exists(), f"Downloaded PDB file not found: {selected_pdb}"
            graph_creator = load_graph_creator(fragment_path=args.fragment_path, workers=args.workers, pdb_path=uniprot_downloader.output_dir)
            fragment_path = graph_creator.classify_and_save_graph(selected_pdb)
            if fragment_path.exists():
                st.success(f"Downloaded PDB file and converted to fragment: {fragment_path}")
        except Exception as e:
            st.error(f"Error downloading PDB file: {e}")
            return
    else:
        # Fallback to selecting a PDB file from the folder
        pdb_files = load_pdb_files(uniprot_downloader.output_dir)
        if not pdb_files:
            st.warning("No PDB files found in the folder.")
            return
        selected_pdb = st.selectbox("Select a PDB file:", pdb_files)

    # Rest of your logic to process the selected PDB file and visualize
    # Initialize session state for selected segment
    if "selected_segment" not in st.session_state:
        st.session_state.selected_segment = None

    # Get Uniprot ID if not provided
    uniprot_id = uniprot_code if uniprot_code else selected_pdb.stem
    prosite = uniprot_downloader.get_active_site_from_uniprot(uniprot_id)

    features = []
    # Iterate through the prosite data
    for entry in prosite:
        pru_id = entry["prosite_rule_id"]
        go_description = pru_to_go.get(pru_id, "Unknown")
        color = go_colors.get(go_description, "lightgray")  # Default color if not found

        feature = {
            "name": go_description,
            "start": entry["location_start"],
            "end": entry["location_end"],
            "color": color,
        }
        features.append(feature)
    # Get Fragment graph
    fragment_path = uniprot_downloader.output_dir / uniprot_id[1:3] / (uniprot_id + ".fg")
    curr_fragment_graph = StructureFragmentGraphIO.load(fragment_path)
    # Plot fragment graph

    for node, data in curr_fragment_graph.graph.nodes(data=True):
        curr_fragment_graph.graph.nodes[node].update(convert_to_serializable(data))

    for u, v, data in curr_fragment_graph.graph.edges(data=True):
        curr_fragment_graph.graph.edges[u, v].update(convert_to_serializable(data))

    curr_fragment_graph.graph, color_hex = color_peptide_bonds(curr_fragment_graph.graph)

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
    for node, data in curr_fragment_graph.graph.nodes(data=True):
        node_label = data.get("label", node)
        color = "#FFFFFF"
        net.add_node(node, label=node_label, title=str(data), color=color)

    for u, v, data in curr_fragment_graph.graph.edges(data=True):
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
    # Extract fragment features
    fragments = []
    for fragment_detail in curr_fragment_graph.structure_fragment.classification_map:
        if fragment_detail.fragment_class == 0:
            continue
        fragment = {
            "name": fragment_detail.fragment_class,
            "start": int(fragment_detail.start_idx),
            "end": int(fragment_detail.end_idx),
            "color": "yellow",
        }
        fragments.append(fragment)

    # Combine features and fragments into one list
    all_segments = features + fragments

    if all_segments:
        # Display the combined feature and fragment plot
        st.plotly_chart(create_combined_plot(features, fragments))

        # Create a list of segment names for the selectbox
        segment_names = [f"{f['name']} {f['start']}-{f['end']}" for f in all_segments]

        # Allow the user to select a segment to zoom in
        selected_segment_name = st.selectbox(
            "Select a segment to zoom in:", segment_names
        )

        # Find the selected segment based on the selected name
        selected_segment = next(
            f
            for f in all_segments
            if f"{f['name']} {f['start']}-{f['end']}" == selected_segment_name
        )

        # Update the selected segment in session state
        st.session_state.selected_segment = selected_segment

        # Reset view button
        if st.button("Reset View"):
            st.session_state.selected_segment = None

    # Re-visualize with the selected segment highlighted
    if st.session_state.selected_segment:
        visualize_protein(
            selected_pdb,
            st.session_state.selected_segment,
            st.session_state.selected_segment["color"],
        )
    else:
        visualize_protein(selected_pdb)


if __name__ == "__main__":
    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser(
        description="Protein Feature and Fragment Visualization Tool"
    )
    parser.add_argument(
        "--pdb_folder",
        type=str,
        required=True,
        help="Path to the folder containing PDB files",
    )
    parser.add_argument(
        "--fragment_path", type=str, required=True, help="Path to fragment classifier"
    )
    parser.add_argument("--workers", type=int, help="Number of workers", default=10)

    args = parser.parse_args()

    # Run the Streamlit app with the parsed arguments
    main(args.pdb_folder)
