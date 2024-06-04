import torch
import speechbrain as sb


class FinetuningModel(torch.nn.Module):
    """
    Unified fine-tuning model that integrates a pretrained model 
    with adaptive average pooling and MLP for classification. It supports two input types:
    'BC' (Batch-Channel) and 'TC' (Time-Channel), which are handled differently in the forward pass.

    Parameters:
    -----------
    ssl_model : str
        The pretrained model.
    dense_n_neurons : int
        The number of classes for the final classification layer.
    input_type : str, optional
        Specifies the type of input handling. 'BC' (default) treats input as (B*C, T),
        while 'TC' treats input as (B, T*C). 'BC' stands for Batch-Channel and 'TC' for Time-Channel.

    Examples:
    ---------
    >>> model = FinetuningModel(ssl_model, input_type='BC', dense_n_neurons=10)
    >>> inputs = torch.rand(10, 640)  # Simulated input for demonstration.
    >>> outputs = model(inputs)
    """
    def __init__(self, ssl_model, dense_n_neurons, input_type='BC'):
        super(FinetuningModel, self).__init__()
        assert input_type in ['BC', 'TC'], "input_type must be either 'BC' or 'TC'"
        self.input_type = input_type

        self.base_model = ssl_model
        # MLP for classification
        ssl_output_dim = self.base_model.config.hidden_size

        self.dense_module = torch.nn.Sequential()
        self.dense_module.add_module(
            "flatten", torch.nn.Flatten(),
        )
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=ssl_output_dim, n_neurons=dense_n_neurons,
            ),
        )
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))

        # Adaptive pooling layer to match the expected MLP input size
        self.adaptive_pool = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Depending on the input type, handles and processes the input data through
        the pretrained transformer, applies adaptive pooling, and passes the output through an MLP classifier.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor (B, T, C, 1). Should be shaped as (B*C, T) for 'BC' input type or (B, T*C) for 'TC' input type.

        Returns:
        --------
        torch.Tensor
            Logits output from the classification layer.
        """
        bs, t, c, _ = x.shape
        if self.input_type == 'BC':
            # Handling for BC type input: (B*C, T)
            x = x.permute(0, 2, 1, 3)  # bring channel first index
            x = x.reshape(bs * c, t)  # Reshape to (B*C, T)
            outputs = self.base_model(x)  # (B*C, T, 768)
            outputs = outputs.reshape(bs, c, -1, outputs.shape[-1])  # Reshape to (B, C, T, 768)
            outputs = outputs.permute(0, 2, 1, 3)  # (B, T, C, 768)
            outputs = outputs.mean(dim=-2)  # Reduce (mean) along the channel dimension (B, T, 768)
        elif self.input_type == 'TC':
            # Handling for TC type input: (B, T*C)
            x = x.reshape(bs, c * t)
            outputs = self.base_model(x)  # (B, T, 768)

        # Apply adaptive pooling
        outputs = outputs.permute(0, 2, 1)  # (B, 768, T)
        pooled_output = self.adaptive_pool(outputs)  # Shape: (B, 768, 1)

        # Classifier to get the final predictions
        logits = self.dense_module(pooled_output)
        return logits
