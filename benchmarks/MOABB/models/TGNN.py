import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
import speechbrain as sb


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


class TGNN(torch.nn.Module):
    """TGNN.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
    dense_n_neurons: int
        Output layer shape
    activation_type: str
        Activation function of the hidden layers.

    Example
    -------
    >>> output = model(graph)
    >>> output.shape
    torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape,  # (1, T, C, 1)
        # cnn_kernels,
        # cnn_kernelsize,
        # cnn_pool,
        cnn_kernels_1,
        cnn_kernels_2,
        cnn_kernels_3,
        cnn_kernelsize_1,
        cnn_kernelsize_2,
        cnn_kernelsize_3,
        cnn_pool_1,
        cnn_pool_2,
        cnn_pool_type="avg",
        dropout=0.5,
        embed_dim=768,
        dense_n_neurons=4,
        activation_type="relu",
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        if activation_type == "gelu":
            self.activation = torch.nn.GELU()
        elif activation_type == "elu":
            self.activation = torch.nn.ELU()
        elif activation_type == "relu":
            self.activation = torch.nn.ReLU()

        self.T = input_shape[1]
        
        cnn_kernels = [cnn_kernels_1, cnn_kernels_2, cnn_kernels_3]
        cnn_kernelsize = [self.ensure_odd(cnn_kernelsize_1), self.ensure_odd(cnn_kernelsize_2), self.ensure_odd(cnn_kernelsize_3)]
        cnn_pool = [cnn_pool_1, cnn_pool_2]

        # Temporal convolutional module
        self.temporal_conv_module = nn.Sequential(
            nn.Conv2d(1, cnn_kernels[0], kernel_size=(cnn_kernelsize[0], 1), padding='same'),
            nn.BatchNorm2d(cnn_kernels[0], momentum=0.01, affine=True),
            self.activation,

            nn.Conv2d(cnn_kernels[0], cnn_kernels[1], kernel_size=(cnn_kernelsize[1], 1)),
            nn.BatchNorm2d(cnn_kernels[1], momentum=0.01, affine=True),
            nn.AvgPool2d(kernel_size=(cnn_pool[0], 1), stride=(cnn_pool[0], 1)) if cnn_pool_type == "avg" else nn.MaxPool2d(kernel_size=(cnn_pool[0], 1), stride=(cnn_pool[0], 1)),
            self.activation,
            nn.Dropout(p=dropout),

            nn.Conv2d(cnn_kernels[1], cnn_kernels[2], kernel_size=(cnn_kernelsize[2], 1)),
            nn.BatchNorm2d(cnn_kernels[2], momentum=0.01, affine=True),
            nn.AvgPool2d(kernel_size=(cnn_pool[1], 1), stride=(cnn_pool[1], 1)) if cnn_pool_type == "avg" else nn.MaxPool2d(kernel_size=(cnn_pool[1], 1), stride=(cnn_pool[1], 1)),
            self.activation,
            nn.Dropout(p=dropout),
        )
        self.flatten = nn.Flatten()

        # position encoder
        self.pos_encoder = PositionEncoder(in_dim=3, out_dim=cnn_kernels[2], activation=self.activation)

        # Shape of intermediate feature maps
        dummy_input = torch.zeros(1, 1, self.T, 1)
        node_fts = self.temporal_conv_module(dummy_input)
        in_channels = node_fts.shape[1]

        # Spatial Graph convolution
        self.spatial_gnn = GnnModel(in_features=in_channels, out_features=embed_dim, activation=self.activation)
        self.global_pool = global_mean_pool

        in_channels = embed_dim * node_fts.shape[2]

        self.dense_module = torch.nn.Sequential(
            torch.nn.Linear(in_channels, dense_n_neurons),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, graph):
        """Returns the output of the model.

        Arguments
        ---------
        graph.x : torch.Tensor (N, T)
            Input to GNN.
        edge_index : torch.Tensor
            Edge indices for GNN.
        batch : torch.Tensor
            Batch vector for GNN.
        """
        assert graph.x.shape[1] == self.T, f"Inconsistency, unexpect size of T: {self.T}"
        assert len(graph.x.shape) == 3, f"feature dim should 3, got: {graph.x.shape}"
        x = graph.x.unsqueeze(1)

        # Apply temporal convolutions
        x = self.temporal_conv_module(x).squeeze()  # (N, 1, T, 1) --> (N, D, T')

        # Add positional encoding
        pos_encoding = self.pos_encoder(graph.pos)  # (N, 3) ---> (N, D)
        pos_encoding = pos_encoding.unsqueeze(-1).repeat(1, 1, x.size(-1))  # (N, D, T')
        x = x + pos_encoding  # (N, D, T')

        # Apply GNN message passing on every T step
        x = self.spatial_gnn(x, graph.edge_index)  # (N, D', T')

        # Graph level pooling
        x = self.flatten(x)  # (N, D', T') ---> (N, F)
        x = self.global_pool(x, graph.batch)  # (B, F)

        x = self.dense_module(x)  # (B, classes)
        return x
    
    def ensure_odd(self, n):
        return n if n % 2 != 0 else n + 1


# GNN model
class GnnModel(torch.nn.Module):
    def __init__(self, in_features, out_features, activation=torch.nn.ReLU()):
        super().__init__()
        # Spatial Graph convolution
        self.spatial_gnn = torch_geometric.nn.Sequential('x, edge_index', [
            (GCNConv(in_features, out_features, node_dim=1), 'x, edge_index -> x'),
            (activation, 'x -> x'),
            (GCNConv(out_features, out_features, node_dim=1), 'x, edge_index -> x'),
            (activation, 'x -> x'),
        ])

    def forward(self, x, edge_index):
        # permute x to [T, batch_size * num_nodes, in_channels]
        x = x.permute(2, 0, 1)  # (N, D, T) --> # (T, N, D)
        x = self.spatial_gnn(x, edge_index)  # (T, N, D')
        x = x.permute(1, 2, 0)  # (N, D', T)
        return x
