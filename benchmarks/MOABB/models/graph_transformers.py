"""GraphTransformer

"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import MultiheadAttention


class GraphTransformer(torch.nn.Module):
    """GraphTransformer.

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
