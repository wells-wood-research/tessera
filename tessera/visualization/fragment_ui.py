import streamlit as st
import networkx as nx
from pathlib import Path
from pyvis.network import Network
from streamlit.components.v1 import html
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.pyplot as plt
import tempfile

# Adjust the import below to match your project structure.
from tessera.fragments.fragments_graph import StructureFragmentGraphIO


def find_fg_files(base_path):
    """Return a list of .fg files (relative paths) found recursively under base_path."""
    return [str(fg_file.relative_to(base_path)) for fg_file in base_path.rglob("*.fg")]


def load_fg_file(file_path):
    """Load a .fg file using your StructureFragmentGraphIO loader and return the NetworkX graph."""
    structure_fragment_graph = StructureFragmentGraphIO.load(file_path)
    return structure_fragment_graph.graph


def color_graph_with_connectivity(G):
    """
    Apply coloring logic similar to your original plot_graph:
      - Nodes with "Frag. 0" in their label become grey and smaller.
      - Other nodes are lightblue.
      - Peptide bonds are colored in a rainbow (with a black outline), while other edges are dotted grey.
    """
    # Identify nodes
    frag_0_nodes = [node for node, data in G.nodes(data=True) if "Frag. 0" in data.get("label", "")]
    non_frag_0_nodes = [node for node in G.nodes if node not in frag_0_nodes]

    # Set node colors and sizes
    node_colors = {node: "grey" if node in frag_0_nodes else "lightblue" for node in G.nodes}
    node_sizes = {node: 5 if node in frag_0_nodes else 10 for node in G.nodes}

    # Identify edges
    peptide_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("peptide_bond", False)]
    other_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("peptide_bond", False)]

    edge_colors = {}
    edge_styles = {}
    edge_widths = {}

    # Color peptide bonds with rainbow colors and add black outlines
    if peptide_edges:
        color_map = cm.get_cmap("rainbow_r", len(peptide_edges))
        peptide_colors = [matplotlib.colors.to_hex(color_map(i)) for i in range(len(peptide_edges))]
        for idx, (u, v) in enumerate(peptide_edges):
            edge_colors[(u, v)] = peptide_colors[idx]  # Rainbow color
            edge_styles[(u, v)] = "solid"
            edge_widths[(u, v)] = 2
            # Black outline for peptide bonds
            edge_colors[(u, v, "outline")] = "black"
            edge_widths[(u, v, "outline")] = 4

    # Other edges: grey dotted
    for u, v in other_edges:
        edge_colors[(u, v)] = "gray"
        edge_styles[(u, v)] = "dotted"
        edge_widths[(u, v)] = 1

    return node_colors, node_sizes, edge_colors, edge_styles, edge_widths


def save_graph_as_pdf(G, filename="graph.pdf"):
    """
    Save the NetworkX graph G as a PDF file using matplotlib.
    The PDF will mimic your original styling: grey dotted non-peptide edges,
    peptide edges with a black outline and rainbow colors, and node colors/sizes.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G)

    # Draw nodes: grey for "Frag. 0", lightblue for others
    frag_0_nodes = [node for node, data in G.nodes(data=True) if "Frag. 0" in data.get("label", "")]
    non_frag_0_nodes = [node for node in G.nodes if node not in frag_0_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=frag_0_nodes, node_color="grey", node_size=50, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=non_frag_0_nodes, node_color="lightblue", node_size=100, ax=ax)

    # Draw edges
    peptide_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("peptide_bond", False)]
    other_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("peptide_bond", False)]
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color="gray", style="dotted", ax=ax)
    if peptide_edges:
        color_map = cm.get_cmap("rainbow_r", len(peptide_edges))
        peptide_colors = [matplotlib.colors.to_hex(color_map(i)) for i in range(len(peptide_edges))]
        for idx, edge in enumerate(peptide_edges):
            # Draw black outline first
            nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color="black", width=4, ax=ax)
            # Then draw the rainbow-colored edge
            nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=peptide_colors[idx], width=2, ax=ax)

    # Draw labels (skip "Frag. 0" labels if desired; here we draw all)
    labels = {node: G.nodes[node].get("label", str(node)) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax)

    ax.set_axis_off()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    st.title("Interactive Graph Viewer with PDF Export")

    # User inputs the base directory for .fg files
    fg_base_path = st.text_input(
        "Enter the base directory for .fg files:",
        "/Users/leo/Documents/code/protein_graphs/data/visuals/"
    )
    fg_base_path = Path(fg_base_path)

    if not fg_base_path.exists():
        st.error("The specified directory does not exist. Please enter a valid path.")
        return

    # Find available .fg files
    fg_files = find_fg_files(fg_base_path)
    if not fg_files:
        st.warning("No .fg files found in the directory.")
        return

    # User selects a .fg file from a dropdown
    selected_file = st.selectbox("Select an .fg file:", fg_files)
    if selected_file:
        st.write(f"Loading graph from: {selected_file}")
        G = load_fg_file(fg_base_path / selected_file)

        # Apply the coloring logic
        node_colors, node_sizes, edge_colors, edge_styles, edge_widths = color_graph_with_connectivity(G)

        # Create interactive PyVis network
        net = Network(height="500px", width="100%", directed=True, notebook=False, cdn_resources="in_line")

        # Add nodes with specified colors and sizes
        for node, data in G.nodes(data=True):
            node_label = data.get("label", str(node))
            net.add_node(node, label=node_label, title=str(data), color=node_colors[node], size=node_sizes[node])

        # Add edges with proper connectivity and styles
        for u, v in G.edges():
            color = edge_colors.get((u, v), "#aaaaaa")
            style = edge_styles.get((u, v), "solid")
            width = edge_widths.get((u, v), 1)
            net.add_edge(u, v, color=color, width=width, title=style)
            # Add black outline for peptide bonds if defined
            if (u, v, "outline") in edge_colors:
                net.add_edge(u, v, color=edge_colors[(u, v, "outline")], width=edge_widths[(u, v, "outline")])

        # Display the interactive graph
        net_html = net.generate_html()
        html(net_html, height=600)

        # Button to generate and download the PDF version of the graph
        if st.button("Download Graph as PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                save_graph_as_pdf(G, tmpfile.name)
                # Read the file data for download
                with open(tmpfile.name, "rb") as f:
                    pdf_data = f.read()
                st.download_button(label="Click to Download PDF", data=pdf_data, file_name="graph.pdf", mime="application/pdf")


if __name__ == "__main__":
    main()
