"""MultiChannelMLP from https://doi.org/10.1088/1741-2552/aace8c.
Shallow and lightweight convolutional neural network proposed for a general decoding of single-trial EEG signals.
It was proposed for P300, error-related negativity, motor execution, motor imagery decoding.

Authors
 * Davide Borra, 2021
"""
import torch
import speechbrain as sb


class MultiChannelMLP(torch.nn.Module):
    """MultiChannelMLP.

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
    #>>> inp_tensor = torch.rand([1, 200, 32, 1])
    #>>> model = MultiChannelMLP(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape=None,  # (1, T, C, 1)
        hidden_layer=512,
        dense_n_neurons=4,
        dense_max_norm=0.25,
        activation_type="relu",
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        if activation_type == "gelu":
            activation = torch.nn.GELU()
        elif activation_type == "elu":
            activation = torch.nn.ELU()
        elif activation_type == "relu":
            activation = torch.nn.ReLU()
        self.default_sf = 128  # sampling rate of the original publication (Hz)
        T = input_shape[1]
        # DENSE MODULE
        self.dense_module_1 = torch.nn.Sequential()
        self.dense_module_2 = torch.nn.Sequential()
        self.dense_module_1.add_module(
            "fc_in",
            sb.nnet.linear.Linear(
                input_size=T,
                n_neurons=hidden_layer,
                max_norm=dense_max_norm,
            ),
        )
        self.dense_module_1.add_module('act_1', activation)
        self.dense_module_2.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=hidden_layer,
                n_neurons=dense_n_neurons,
                max_norm=dense_max_norm,
            ),
        )
        self.dense_module_2.add_module("act_out", torch.nn.LogSoftmax(dim=1))

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected. 
        """
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)
        x = self.dense_module_1(x)
        x = x.mean(1)  # global average pooling
        x = self.dense_module_2(x)
        return x