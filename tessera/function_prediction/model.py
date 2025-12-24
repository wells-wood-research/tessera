import typing as t

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import (
    BatchNorm,
    GATv2Conv,
    GINEConv,
    SAGPooling,
    global_max_pool,
)


class GNNModel(nn.Module):
    def __init__(
        self,
        node_cat_dims: t.List[int],
        node_cont_dim: int,
        edge_cat_dims: t.List[int],
        edge_cont_dim: int,
        c_hidden: int,
        c_out: int,
        num_layers: int = 2,
        dp_rate: float = 0.1,
        heads: int = 1,
    ):
        super().__init__()

        if not isinstance(node_cat_dims, list) or not isinstance(edge_cat_dims, list):
            raise TypeError("node_cat_dims and edge_cat_dims must be lists")
        # Handle embeddings for categorical features
        self.node_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories, num_categories)
                for num_categories in node_cat_dims
            ]
        )
        self.edge_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories, num_categories)
                for num_categories in edge_cat_dims
            ]
        )

        layers = []
        node_in_channels = sum(node_cat_dims) + node_cont_dim
        edge_in_channels = sum(edge_cat_dims) + edge_cont_dim

        for _ in range(num_layers - 1):
            layers += [
                GINEConv(
                    Sequential(
                        Linear(node_in_channels, c_hidden),
                        # ReLU(),
                        # Linear(c_hidden, c_hidden),
                        # ReLU(),
                        # Linear(c_hidden, c_hidden),
                        # ReLU(),
                    ),
                    train_eps=True,
                    edge_dim=edge_in_channels,
                ),
                ReLU(),
                BatchNorm(c_hidden),
                nn.Dropout(dp_rate),
            ]
            node_in_channels = c_hidden
        layers += [
            GATv2Conv(
                node_in_channels,
                c_out,
                heads=heads,
                concat=False,
                edge_dim=edge_in_channels,
            )
        ]
        self.layers = nn.ModuleList(layers)
        self.final_dim = c_out
        self.attention_coefficients = []

    def forward(self, data):
        # Prepare Node Embeddings
        node_categorical = data.feature_type_map["node_categorical"]
        node_continuous = data.feature_type_map["node_continuous"]
        # Handle the case where the feature can be a tuple or a list of tuples
        node_cat_start, node_cat_end = (
            node_categorical
            if isinstance(node_categorical[0], int)
            else node_categorical[0]
        )
        node_cont_start, node_cont_end = (
            node_continuous
            if isinstance(node_continuous[0], int)
            else node_continuous[0]
        )
        x_cat = data.x[:, node_cat_start:node_cat_end].long()
        x_cont = data.x[:, node_cont_start:node_cont_end]
        # Embed categorical features
        x_embedded = [
            self.node_embeddings[i](x_cat[:, i]) for i in range(x_cat.shape[1])
        ]
        # x_embedded = torch.cat(x_embedded, dim=-1)
        # x_combined = torch.cat([x_embedded, x_cont], dim=-1)

        if len(x_embedded) == 0:
            x_embedded = x_embedded
            x_combined = x_cont
        else:
            x_embedded = torch.cat(x_embedded, dim=-1)
            x_combined = torch.cat([x_embedded, x_cont], dim=-1)

        # Prepare Edge Embeddings
        edge_categorical = data.feature_type_map["edge_categorical"]
        edge_continuous = data.feature_type_map["edge_continuous"]
        # Handle the case where the feature can be a tuple or a list of tuples
        edge_cat_start, edge_cat_end = (
            edge_categorical
            if isinstance(edge_categorical[0], int)
            else edge_categorical[0]
        )
        edge_cont_start, edge_cont_end = (
            edge_continuous
            if isinstance(edge_continuous[0], int)
            else edge_continuous[0]
        )

        edge_cont = data.edge_attr[:, edge_cont_start:edge_cont_end]
        edge_cat = data.edge_attr[:, edge_cat_start:edge_cat_end].long()

        edge_embedded = [
            self.edge_embeddings[i](edge_cat[:, i]) for i in range(edge_cat.shape[1])
        ]
        # edge_embedded = torch.cat(edge_embedded, dim=-1)
        # edge_combined = torch.cat([edge_embedded, edge_cont], dim=-1)

        if len(edge_embedded) == 0:
            edge_embedded = edge_embedded
            edge_combined = edge_cont
        else:
            edge_embedded = torch.cat(edge_embedded, dim=-1)
            edge_combined = torch.cat([edge_embedded, edge_cont], dim=-1)

        for layer in self.layers:
            if isinstance(layer, GINEConv):
                x_combined = layer(x_combined, data.edge_index, edge_combined)
            elif isinstance(layer, GATv2Conv):
                x_combined, attn_coef = layer(
                    x_combined,
                    data.edge_index,
                    edge_combined,
                    return_attention_weights=True,
                )
                self.attention_coefficients.append(attn_coef)
            else:
                x_combined = layer(x_combined)

        return x_combined


class FragmentNetGNN(nn.Module):
    def __init__(
        self,
        node_cat_dims: t.List[int],
        node_cont_dim: int,
        edge_cat_dims: t.List[int],
        edge_cont_dim: int,
        c_hidden: int,
        c_out: int,
        dp_rate_linear: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.GNN = GNNModel(
            node_cat_dims=node_cat_dims,
            node_cont_dim=node_cont_dim,
            edge_cat_dims=edge_cat_dims,
            edge_cont_dim=edge_cont_dim,
            c_hidden=c_hidden,
            c_out=c_hidden,
            **kwargs
        )
        self.final_dim = self.GNN.final_dim

        # Adding a pooling layer
        self.pool = SAGPooling(self.final_dim, ratio=0.8)

        self.head = nn.Sequential(
            nn.Linear(self.final_dim, self.final_dim // 3),
            nn.ReLU(),
            nn.Dropout(dp_rate_linear),
            nn.Linear(self.final_dim // 3, c_hidden // 2),
            nn.ReLU(),
            nn.Linear(c_hidden // 2, c_out),
        )

    def forward(self, data):
        # Pass the data through the GNN model
        x = self.GNN(data)
        # Apply pooling
        x, edge_index, _, batch, perm, _ = self.pool(
            x, data.edge_index, batch=data.batch
        )
        # Apply global pooling to summarize the node features into graph-level representation
        x = global_max_pool(
            x, batch
        )  # You can also use global_add_pool or global_max_pool
        # Pass through the fully connected layers
        x = self.head(x)
        return x

    def get_node_attention_weights(self):
        return self.GNN.attention_coefficients
