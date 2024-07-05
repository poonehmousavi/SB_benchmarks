"""
This is a simple spatio-temporal GNN model.

This model differs from TGNN in the TGNN.py file that it
    - Uses pure pytorch modules
    - Replaces pooling layers with strides, so to learn the pooling layers
    - Only uses two Temporal conv modules: Before and after the spatial GNN
"""

import torch
import torch_geometric
from torch import jit
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class PositionEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, activation=torch.nn.ReLU()):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            activation,
            torch.nn.Linear(out_dim, out_dim)
        )

    def forward(self, pos):
        return self.mlp(pos)


# Chunking approach to optimize applying GNN on every T step
class ChunkedTGnnModel(torch.nn.Module):
    def __init__(self, in_features, out_features, chunk_size, activation=torch.nn.ReLU()):
        super().__init__()
        self.chunk_size = chunk_size
        # Spatial Graph convolution
        self.spatial_gnn = torch_geometric.nn.Sequential('x, edge_index', [
            (GCNConv(in_features, out_features), 'x, edge_index -> x'),
            (activation, 'x -> x'),
            (GCNConv(out_features, out_features), 'x, edge_index -> x'),
            (activation, 'x -> x'),
        ])

    def forward(self, x, edge_index):
        N, T, D = x.shape
        outputs = []

        for t in range(0, T, self.chunk_size):
            chunk = x[:, t:t + self.chunk_size, :]
            N, T_chunk, D = chunk.shape
            chunk_reshaped = chunk.reshape(-1, D)

            edge_index_repeated = edge_index.repeat(1, T_chunk) + torch.arange(T_chunk, device=x.device).repeat_interleave(edge_index.shape[1]) * N

            chunk_processed = self.spatial_gnn(chunk_reshaped, edge_index_repeated)
            outputs.append(chunk_processed.reshape(N, T_chunk, -1))

        return torch.cat(outputs, dim=1)


# Jitted approach to optimize applying GNN on every T step
class JittedGnnModel(torch.nn.Module):
    def __init__(self, in_features, out_features, activation=torch.nn.ReLU()):
        super().__init__()
        # Spatial Graph convolution
        self.spatial_gnn = torch_geometric.nn.Sequential('x, edge_index', [
            (GCNConv(in_features, out_features), 'x, edge_index -> x'),
            (activation, 'x -> x'),
            (GCNConv(out_features, out_features), 'x, edge_index -> x'),
            (activation, 'x -> x'),
        ])

    @torch.jit.script_method
    def forward(self, x, edge_index):
        out = torch.stack([self.spatial_gnn(x[:, t, :], edge_index) for t in range(x.shape[1])], dim=1)  # (N, T, D')
        return out


class STGNN(nn.Module):
    def __init__(
        self,
        input_shape,
        cnn_kernels,
        cnn_kernelsize,
        cnn_strides,
        chunk_size=None,
        dropout=0.5,
        embed_dim=768,
        dense_n_neurons=4,
        activation_type="relu",
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")

        if activation_type == "gelu":
            self.activation = nn.GELU()
        elif activation_type == "elu":
            self.activation = nn.ELU()
        elif activation_type == "relu":
            self.activation = nn.ReLU()

        self.chunk_size = chunk_size
        self.T = input_shape[1]

        # Temporal convolution
        self.temporal_module = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=cnn_kernels[0],
                kernel_size=(cnn_kernelsize[0], 1),
                padding='same',
                padding_mode='zeros',
                bias=False,
            ),
            nn.BatchNorm2d(cnn_kernels[0], momentum=0.01, affine=True),
            self.activation
        )

        # Position encoder
        self.pos_encoder = PositionEncoder(in_dim=3, out_dim=cnn_kernels[0], activation=self.activation)

        # Spatial Graph convolution
        if chunk_size:
            self.spatial_gnn = ChunkedTGnnModel(in_features=cnn_kernels[0], out_features=embed_dim, chunk_size=chunk_size, activation=self.activation)
        else:
            self.spatial_gnn = torch.jit.script(JittedGnnModel(in_features=cnn_kernels[0], out_features=embed_dim, activation=self.activation))

        # Spatio Temporal convolutions
        self.spatio_tempora_module = nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=cnn_kernels[1],
                kernel_size=(cnn_kernelsize[1], 1),
                stride=(cnn_strides[0], 1),
                padding='valid',
                bias=False,
            ),
            nn.BatchNorm2d(cnn_kernels[1], momentum=0.01, affine=True),
            self.activation,
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=cnn_kernels[1],
                out_channels=cnn_kernels[2],
                kernel_size=(cnn_kernelsize[2], 1),
                stride=(cnn_strides[1], 1),
                padding='valid',
                bias=False,
            ),
            nn.BatchNorm2d(cnn_kernels[2], momentum=0.01, affine=True),
            self.activation,
            nn.Dropout(p=dropout),
            nn.Flatten(),
        )

        # Calculate the shape of the output after convolutions
        gnn_out = torch.ones((1, embed_dim, self.T, 1))
        conv_out = self.spatio_tempora_module(gnn_out)
        self.conv_out_size = conv_out.size(-1)

        self.dense_module = nn.Sequential(
            nn.Linear(self.conv_out_size, dense_n_neurons),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, graph):
        assert graph.x.shape[1] == self.T, f"Inconsistency, unexpected size of T: {self.T}"
        assert len(graph.x.shape) == 3, f"feature dim should be 3, got: {graph.x.shape}"
        x = graph.x.unsqueeze(1)  # (N, 1, T, 1)
        pos_encoding = self.pos_encoder(graph.pos)  # (N, D)
        pos_encoding = pos_encoding.unsqueeze(-1).repeat(1, 1, self.T)  # (N, D, T)
        x = self.temporal_module(x).squeeze()  # (N, D, T)
        x = (x + pos_encoding).permute(0, 2, 1)  # (N, T, D)
        x = self.spatial_gnn(x, graph.edge_index)  # (N, T, D'')
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (N, D'', T, 1)
        x = self.spatio_tempora_module(x)  # (N, D'', T', 1)
        # Global pooling
        x = global_mean_pool(x, graph.batch)  # (B, D''')
        # Final layer
        x = self.dense_module(x)  # (B, classes)
        return x
