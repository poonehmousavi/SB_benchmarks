import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
import speechbrain as sb
from torch import jit


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
        cnn_kernels,
        cnn_kernelsize,
        cnn_pool,
        chunk_size=None,
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

        self.chunk_size = chunk_size
        self.T = input_shape[1]
        self.temp_conv_module_1 = torch.nn.Sequential()
        # Temporal convolution
        self.temp_conv_module_1.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_kernels[0],
                kernel_size=(cnn_kernelsize[0], 1),
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.temp_conv_module_1.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_kernels[0], momentum=0.01, affine=True,
            ),
        )
        self.temp_conv_module_1.add_module("act_0", self.activation)

        # position encoder
        self.pos_encoder = PositionEncoder(3, cnn_kernels[0])

        # Shape of intermediate feature maps
        node_fts = self.temp_conv_module_1(torch.ones((1, self.T, 1, 1)))
        in_channels = node_fts.shape[-1]

        # Spatial Graph convolution
        if chunk_size:
            self.spatial_gnn = ChunkedTGnnModel(in_features=in_channels, out_features=embed_dim, chunk_size=chunk_size)
        else:
            self.spatial_gnn = torch.jit.script(JittedGnnModel(in_features=in_channels, out_features=embed_dim))

        self.temp_conv_module_2 = torch.nn.Sequential()
        self.temp_conv_module_2.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=embed_dim,
                out_channels=cnn_kernels[1],
                kernel_size=(cnn_kernelsize[1], 1),
                padding="valid",
                bias=False,
                swap=True,
            ),
        )
        self.temp_conv_module_2.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_kernels[1], momentum=0.01, affine=True,
            ),
        )

        self.temp_conv_module_2.add_module(
            "pool_0",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=(cnn_pool[0], 1),
                stride=(cnn_pool[0], 1),
                pool_axis=[1, 2],
            ),
        )
        self.temp_conv_module_2.add_module("act_0", self.activation)
        self.temp_conv_module_2.add_module("dropout_0", torch.nn.Dropout(p=dropout))

        self.global_pool = global_mean_pool

        self.rep_conv_module = torch.nn.Sequential()

        self.rep_conv_module.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_kernels[1],
                out_channels=cnn_kernels[2],
                kernel_size=(cnn_kernelsize[2], 1),
                padding="valid",
                bias=False,
                swap=True,
            ),
        )
        self.rep_conv_module.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_kernels[2],
                momentum=0.01,
                affine=True,
            ),
        )
        self.rep_conv_module.add_module(
            "pool_0",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=(cnn_pool[1], 1),
                stride=(cnn_pool[1], 1),
                pool_axis=[1, 2],
            ),
        )
        self.rep_conv_module.add_module("act_0", self.activation)
        self.rep_conv_module.add_module("dropout_0", torch.nn.Dropout(p=dropout))
        self.rep_conv_module.add_module("flatten_0", torch.nn.Flatten())

        # Shape of intermediate feature maps
        gnn_out = torch.ones((1, self.T, 1, embed_dim))
        tgnn_out = self.temp_conv_module_2(gnn_out)
        out = self.rep_conv_module(tgnn_out)
        in_channels = out.shape[-1]

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
        x = graph.x.unsqueeze(-1)
        pos_encoding = self.pos_encoder(graph.pos)  # (N, 3) ---> (N, D)
        pos_encoding = pos_encoding.unsqueeze(1).repeat(1, self.T, 1)  # (N, T, D)
        x = self.temp_conv_module_1(x)
        x = x.squeeze(2)  # (N, T, 1, D) ---> (N, T, D)
        x = x + pos_encoding  # (N, T, D)
        x = self.spatial_gnn(x, graph.edge_index)  # (N, T, D')
        x = x.unsqueeze(2)  # (N, T, 1, D')
        x = self.temp_conv_module_2(x)  # (N, T', 1, D'')
        x = x.squeeze(2)  # (N, T', 1, D) ---> (N, T', D'')
        x = torch.stack([self.global_pool(x[:, t, :], graph.batch) for t in range(x.shape[1])], dim=1)  # (B, T', D'')
        x = x.unsqueeze(2)  # (B, T, 1, D'')
        x = self.rep_conv_module(x)
        x = self.dense_module(x)
        return x


class PositionEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim, out_dim)
        )

    def forward(self, pos):
        return self.mlp(pos)


# Chunking approach to optimize applying GNN on every T step
class ChunkedTGnnModel(torch.nn.Module):
    def __init__(self, in_features, out_features, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        # Spatial Graph convolution
        self.spatial_gnn = torch_geometric.nn.Sequential('x, edge_index', [
            (GCNConv(in_features, out_features), 'x, edge_index -> x'),
            (torch.nn.ReLU(), 'x -> x'),
            (GCNConv(out_features, out_features), 'x, edge_index -> x'),
            (torch.nn.ReLU(), 'x -> x'),
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
    def __init__(self, in_features, out_features):
        super().__init__()
        # Spatial Graph convolution
        self.spatial_gnn = torch_geometric.nn.Sequential('x, edge_index', [
            (GCNConv(in_features, out_features), 'x, edge_index -> x'),
            (torch.nn.ReLU(), 'x -> x'),
            (GCNConv(out_features, out_features), 'x, edge_index -> x'),
            (torch.nn.ReLU(), 'x -> x'),
        ])

    @torch.jit.script_method
    def forward(self, x, edge_index):
        out = []
        for t in range(x.shape[1]):
            out.append(self.spatial_gnn(x[:, t, :], edge_index))
        return torch.stack(out, dim=1)
