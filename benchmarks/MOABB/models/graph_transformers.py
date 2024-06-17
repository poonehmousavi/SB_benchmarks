"""GraphTransformer

"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GIN, global_mean_pool
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch.nn import MultiheadAttention
import speechbrain as sb


class GraphSelfAttention(torch.nn.Module):
    """GraphSelfAttention.

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
    >>> inp_tensor = torch.rand([1, 200, 32, 1])
    >>> model = GraphSelfAttention(input_shape=inp_tensor.shape)
    >>> output = model(node_fts, edge_index)
    >>> output.shape
    torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape=None,  # (B, N, T)
        dense_n_neurons=4,
        activation_type="relu",
        embed_dim=768,
        num_heads=8
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

        self.gnn_conv = GCNConv(input_shape[1], embed_dim)
        self.global_pool = global_mean_pool

        self.self_attention = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.dense_module = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            self.activation,
            torch.nn.Linear(embed_dim, dense_n_neurons),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, graph):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (B, N, T)
            Input to GNN. 3D tensors are expected. 
        edge_index : torch.Tensor
            Edge indices for GNN.
        batch : torch.Tensor
            Batch vector for GNN.
        """
        x = self.gnn_conv(graph.x, graph.edge_index)
        x = self.activation(x)
        x = self.global_pool(x, graph.batch)  # (B, embed_dim)
        
        x = x.unsqueeze(1)  # (B, 1, embed_dim)
        attn_output, _ = self.self_attention(x, x, x)  # Self-attention
        
        x = attn_output.squeeze(1)  # (B, embed_dim)
        x = self.dense_module(x)
        
        return x


class STGNN(torch.nn.Module):
    """STGNN.

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
    >>> inp_tensor = torch.rand([1, 200, 32, 1])
    >>> model = GraphTransformer(input_shape=inp_tensor.shape)
    >>> output = model(node_fts, edge_index)
    >>> output.shape
    torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape,  # (1, T, C, 1)
        cnn_temporal_kernels=8,
        cnn_temporal_kernelsize=(33, 1),
        cnn_temporal_pool=(8, 1),
        cnn_septemporal_depth_multiplier=1,
        cnn_septemporal_kernelsize=(21, 1),
        cnn_septemporal_point_kernels=None,
        cnn_septemporal_pool=(8, 1),
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
        self.conv_module = torch.nn.Sequential()
        # Temporal convolution
        self.conv_module.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_temporal_kernels,
                kernel_size=cnn_temporal_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.conv_module.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_temporal_kernels, momentum=0.01, affine=True,
            ),
        )

        # Separable temporal convolution
        cnn_septemporal_kernels = (
            cnn_temporal_kernels * cnn_septemporal_depth_multiplier
        )
        self.conv_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels,
                out_channels=cnn_septemporal_kernels,
                kernel_size=cnn_septemporal_kernelsize,
                groups=cnn_temporal_kernels,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.conv_module.add_module("act_1", self.activation)

        self.conv_module.add_module(
            "conv_2",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_septemporal_kernels,
                out_channels=cnn_septemporal_point_kernels,
                kernel_size=(1, 1),
                padding="valid",
                bias=False,
                swap=True,
            ),
        )

        self.conv_module.add_module(
            "bnorm_2",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_septemporal_point_kernels,
                momentum=0.01,
                affine=True,
            ),
        )
        self.conv_module.add_module("act_2", self.activation)
        self.conv_module.add_module(
            "pool_2",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_septemporal_pool,
                stride=cnn_septemporal_pool,
                pool_axis=[1, 2],
            ),
        )
        self.conv_module.add_module("dropout_2", torch.nn.Dropout(p=dropout))

        # Shape of intermediate feature maps
        node_fts = self.conv_module(torch.ones((1, self.T, 1, 1)))
        in_channels = self._get_temp_size(node_fts)

        # Spatial Graph convolution
        self.flatten = torch.nn.Flatten()
        self.gnn_conv = GIN(in_channels, embed_dim, num_layers=2)
        self.global_pool = global_mean_pool

        self.dense_module = torch.nn.Sequential(
            # torch.nn.Linear(embed_dim, embed_dim),
            # self.activation,
            torch.nn.Linear(embed_dim, dense_n_neurons),
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
        x = graph.x.unsqueeze(-1).unsqueeze(-1)
        x = self.conv_module(x)
        x = self.flatten(x)
        x = self.gnn_conv(x, graph.edge_index)
        x = self.activation(x)
        x = self.global_pool(x, graph.batch)  # (B, embed_dim)
        x = self.dense_module(x)

        return x

    def _get_temp_size(self, node_fts):
        _, T1, _, T2 = node_fts.shape
        return T1 * T2
