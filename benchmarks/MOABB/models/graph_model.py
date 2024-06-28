"""GraphTransformer

"""
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GIN, global_mean_pool
import speechbrain as sb


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
    >>> model = STGNN(input_shape=inp_tensor.shape)
    >>> output = model(node_fts, edge_index)
    >>> output.shape
    torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape,  # (1, T, C, 1)
        cnn_kernels,
        cnn_kernelsize,
        cnn_pool, 
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
        self.temp_conv_module = torch.nn.Sequential()
        # Temporal convolution
        self.temp_conv_module.add_module(
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
        self.temp_conv_module.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_kernels[0], momentum=0.01, affine=True,
            ),
        )

        self.temp_conv_module.add_module(
            "pool_0",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=(cnn_pool[0], 1),
                stride=(cnn_pool[0], 1),
                pool_axis=[1, 2],
            ),
        )
        self.temp_conv_module.add_module("act_0", self.activation)
        self.temp_conv_module.add_module("dropout_0", torch.nn.Dropout(p=dropout))

        self.temp_conv_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_kernels[0],
                out_channels=cnn_kernels[1],
                kernel_size=(cnn_kernelsize[1], 1),
                padding="valid",
                bias=False,
                swap=True,
            ),
        )
        self.temp_conv_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_kernels[1], momentum=0.01, affine=True,
            ),
        )

        self.temp_conv_module.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=(cnn_pool[1], 1),
                stride=(cnn_pool[1], 1),
                pool_axis=[1, 2],
            ),
        )
        self.temp_conv_module.add_module("act_1", self.activation)
        self.temp_conv_module.add_module("dropout_1", torch.nn.Dropout(p=dropout))

        # Shape of intermediate feature maps
        node_fts = self.temp_conv_module(torch.ones((1, self.T, 1, 1)))
        in_channels = self._get_temp_size(node_fts)

        # Spatial Graph convolution
        self.spatial_gnn = torch_geometric.nn.Sequential('x, edge_index', [
            (torch.nn.Flatten(), 'x -> x'),
            (GCNConv(in_channels, embed_dim), 'x, edge_index -> x'),
            (self.activation, 'x -> x'),
            (GCNConv(embed_dim, embed_dim), 'x, edge_index -> x'),
            (self.activation, 'x -> x'),
        ])
        self.global_pool = global_mean_pool

        self.rep_conv_module = torch.nn.Sequential()

        self.rep_conv_module.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
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
                kernel_size=(cnn_pool[2], 1),
                stride=(cnn_pool[2], 1),
                pool_axis=[1, 2],
            ),
        )
        self.rep_conv_module.add_module("act_0", self.activation)
        self.rep_conv_module.add_module("dropout_0", torch.nn.Dropout(p=dropout))
        self.rep_conv_module.add_module("flatten_0", torch.nn.Flatten())

        # Shape of intermediate feature maps
        edge_index = torch.empty((2, 0), dtype=torch.long)
        gnn_out = self.spatial_gnn(node_fts, edge_index)
        out = self.rep_conv_module(gnn_out.unsqueeze(-1).unsqueeze(-1))
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
        x = graph.x.unsqueeze(-1).unsqueeze(-1)
        assert x.shape[1] == self.T, f"Inconsistency, unexpect size of T: {self.T} != {x.shape[1]}"
        x = self.temp_conv_module(x)
        x = self.spatial_gnn(x, graph.edge_index)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.rep_conv_module(x)
        x = self.global_pool(x, graph.batch)  # (B, embed_dim)
        x = self.dense_module(x)

        return x

    def _get_temp_size(self, node_fts):
        _, T1, _, T2 = node_fts.shape
        return T1 * T2


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
    >>> inp_tensor = torch.rand([1, 200, 32, 1])
    >>> model = TGNN(input_shape=inp_tensor.shape)
    >>> output = model(node_fts, edge_index)
    >>> output.shape
    torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape,  # (1, T, C, 1)
        cnn_kernels,
        cnn_kernelsize,
        cnn_pool,
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
        # self.temp_conv_module_1.add_module("dropout_0", torch.nn.Dropout(p=dropout))

        # Shape of intermediate feature maps
        node_fts = self.temp_conv_module_1(torch.ones((1, self.T, 1, 1)))
        in_channels = node_fts.shape[-1]
        # Spatial Graph convolution
        self.spatial_gnn = torch_geometric.nn.Sequential('x, edge_index', [
            (GCNConv(in_channels, embed_dim), 'x, edge_index -> x'),
            (self.activation, 'x -> x'),
            (GCNConv(embed_dim, embed_dim), 'x, edge_index -> x'),
            (self.activation, 'x -> x'),
        ])

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
        x = graph.x.unsqueeze(-1).unsqueeze(-1)
        assert x.shape[1] == self.T, f"Inconsistency, unexpect size of T: {self.T}"
        x = self.temp_conv_module_1(x)
        x = x.squeeze(2)  # (N, T, 1, D) ---> (N, T, D)
        x = torch.stack([self.spatial_gnn(x[:, t, :], graph.edge_index) for t in range(self.T)], dim=1)  # (N, T, D')
        x = x.unsqueeze(2)  # (N, T, 1, D')
        x = self.temp_conv_module_2(x)  # (N, T', 1, D'')
        x = x.squeeze(2)  # (N, T', 1, D) ---> (N, T', D'')
        x = torch.stack([self.global_pool(x[:, t, :], graph.batch) for t in range(x.shape[1])], dim=1)  # (B, T', D'')
        x = x.unsqueeze(2)  # (B, T, 1, D'')
        x = self.rep_conv_module(x)
        x = self.dense_module(x)
        return x
