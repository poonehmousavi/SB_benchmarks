"""ShallowConvNet from https://doi.org/10.1002/hbm.23730.
Shallow convolutional neural network proposed for motor execution and motor imagery decoding from single-trial EEG signals.
Its design is based on the filter bank common spatial pattern (FBCSP) algorithm.

Authors
 * Davide Borra, 2021
"""
import torch
import speechbrain as sb
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model


class Wav2Vec2ForFinetuning(torch.nn.Module):
    """
    Integration of HuggingFace's pretrained wav2vec2.0 models with the ability
    to use as a fixed feature extractor or for finetuning.

    Parameters
    ----------
    max_duration_in_seconds : int
        number of seconds to consider for padding (should be same as pretraining)
    model_name_or_path : str
        Local path to the EEG pretrained checkpoints.
    random_init : bool, optional
        If True, initializes the model with random weights, else uses pretrained weights (default: False).
    freeze : bool, optional
        If True, the model's weights are frozen (default: True).

    Examples
    --------
    >>> model = Wav2Vec2ForFinetuning(model_name_or_path, max_duration_in_seconds, random_init=False, freeze=True)
    >>> inputs = torch.rand([10, 600])
    >>> outputs = model(inputs)
    """

    def __init__(
        self,
        model_name_or_path,
        max_duration_in_seconds,
        random_init,
        freeze,
        C=64,
        sampling_rate=128,
        do_stable_layer_norm=True,
        feat_extract_norm="layer",
        num_feat_extract_layers=7,
        conv_dim=[512, 512, 512, 512, 512, 512, 512],
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        num_hidden_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        num_conv_pos_embeddings=128,
        num_codevectors_per_group=320,
        input_type="BC",
    ):
        super(Wav2Vec2ForFinetuning, self).__init__()

        self.max_duration_in_seconds = max_duration_in_seconds
        self.model_name_or_path = model_name_or_path
        self.freeze = freeze
        self.C = C
        self.input_type = input_type

        if random_init:
            num_feat_extract_layers = len(conv_kernel)
            print(
                f"***** Initializing the model randomly from {self.model_name_or_path} as a base *****"
            )
            assert (
                len(conv_dim)
                == len(conv_kernel)
                == len(conv_stride)
                == num_feat_extract_layers
            ), "dim mismatch, num_feat_extract_layers == len(conv_dim) == len(conv_kernel) == len(conv_stride)"
            if freeze:
                raise ValueError(
                    "freeze should be False for random initialized model"
                )
            # get the model config used when initializing random model
            self.config = Wav2Vec2Config.from_pretrained(
                self.model_name_or_path,
                do_stable_layer_norm=do_stable_layer_norm,
                feat_extract_norm=feat_extract_norm,
                num_feat_extract_layers=num_feat_extract_layers,
                conv_dim=conv_dim,
                conv_kernel=conv_kernel,
                conv_stride=conv_stride,
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_conv_pos_embeddings=num_conv_pos_embeddings,
                num_codevectors_per_group=num_codevectors_per_group,
            )
            # initialize feature extractor
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_name_or_path,
                do_normalize=True,
                sampling_rate=sampling_rate,
            )
            # Initializing a model (with random weights) from the model_name_or_path style configuration
            self.model = Wav2Vec2Model(config=self.config)
        else:
            print(
                f"***** Initializing the model from pretrained {self.model_name_or_path} checkpoint *****"
            )
            # get the model config used when initializing random model
            self.config = Wav2Vec2Config.from_pretrained(
                self.model_name_or_path
            )
            # initialize feature extractor
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_name_or_path
            )
            # Initializing a model (with pretrained weights) from the model_name_or_path style configuration
            self.model = Wav2Vec2Model.from_pretrained(self.model_name_or_path)

        if freeze:
            self.freeze_model(self.model)
            assert (
                sum(
                    p.numel()
                    for p in self.model.parameters()
                    if p.requires_grad
                )
                == 0
            ), "Model Not Frozen"

    def freeze_model(self, model):
        """
        Freezes parameters of a model.
        This should be overridden too, depending on users' needs, for example, adapters use.

        Arguments
        ---------
        model : from AutoModel.from_config
            Valid HuggingFace transformers model object.
        """
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, features, padding="max_length", pad_to_multiple_of=None):
        """
        Forward pass of the finetuning model.

        Parameters:
        -----------
        features : dict or Tensor
            Either a dictionary containing the key 'input_values', or a raw Tensor with input samples.
        padding : bool or str, optional
            Determines the padding strategy. True or 'longest' will pad to the longest in the batch.
        pad_to_multiple_of : int, optional
            If set, pads the sequence to a multiple of the provided value.

        Returns:
        --------
        torch.Tensor
            The output from the Wav2Vec2 model.
        """

        # Ensure the input is in the right format for padding
        if isinstance(features, dict) and "input_values" in features:
            input_values = features["input_values"]
        elif isinstance(features, torch.Tensor):
            input_values = features
        else:
            raise ValueError(
                "Features should be either a dict with 'input_values' or a raw Tensor."
            )
        if self.input_type == "TC":
            max_length = int(
                self.max_duration_in_seconds * self.feature_extractor.sampling_rate * self.C
            )
        else:
            max_length = int(
                self.max_duration_in_seconds * self.feature_extractor.sampling_rate
            )
        # pad the inputs to match the pretraining
        features = self.feature_extractor.pad(
            {"input_values": input_values},
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        device = self.model.device
        outputs = self.model(features["input_values"].to(device))
        return outputs.last_hidden_state


class Square(torch.nn.Module):
    """Layer for squaring activations."""

    def forward(self, x):
        return torch.square(x)


class Log(torch.nn.Module):
    """Layer to compute log of activations."""

    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class Wav2ShallowConvNet(torch.nn.Module):
    """ShallowConvNet.

    Arguments
    ---------
    input_shape : tuple
        The shape of the input.
    cnn_temporal_kernels : int
        Number of kernels in the 2d temporal convolution.
    cnn_temporal_kernelsize : tuple
        Kernel size of the 2d temporal convolution.
    cnn_spatial_kernels : int
        Number of kernels in the 2d spatial depthwise convolution.
    cnn_poolsize: tuple
        Pool size.
    cnn_poolstride: tuple
        Pool stride.
    cnn_pool_type: string
        Pooling type.
    dropout: float
        Dropout probability.
    dense_n_neurons: int
        Number of output neurons.

    Example
    -------
    >>> inp_tensor = torch.rand([1, 200, 32, 1])
    >>> model = ShallowConvNet(input_shape=inp_tensor.shape)
    >>> output = model(inp_tensor)
    >>> output.shape
    torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape,
        ssl_model,
        input_type="BC",
        cnn_temporal_kernels=40,
        cnn_temporal_kernelsize=(13, 1),
        cnn_spatial_kernels=40,
        cnn_poolsize=(38, 1),
        cnn_poolstride=(8, 1),
        cnn_pool_type="avg",
        dropout=0.5,
        dense_n_neurons=4,
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        self.input_type = input_type
        self.input_shape = input_shape
        C = input_shape[2]
        self.ssl_model = ssl_model
        # Linear layer to reduce dimensionality to 1
        ssl_output_dim = self.ssl_model.config.hidden_size
        if self.input_type == "TC":
            ssl_output_dim = 1
            C = self.ssl_model.config.hidden_size
        # CONVOLUTIONAL MODULE
        self.conv_module = torch.nn.Sequential()
        # Temporal convolution
        self.conv_module.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=ssl_output_dim,
                out_channels=cnn_temporal_kernels,
                kernel_size=cnn_temporal_kernelsize,
                padding="valid",
                bias=True,
                swap=True,
            ),
        )

        # Spatial convolution
        self.conv_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels,
                out_channels=cnn_spatial_kernels,
                kernel_size=(1, C),
                padding="valid",
                bias=False,
                swap=True,
            ),
        )
        self.conv_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels, momentum=0.1, affine=True,
            ),
        )
        # Square-pool-log-dropout
        # conv non-lin
        self.conv_module.add_module(
            "square_1", Square(),
        )
        self.conv_module.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_poolsize,
                stride=cnn_poolstride,
                pool_axis=[1, 2],
            ),
        )
        # pool non-lin
        self.conv_module.add_module(
            "log_1", Log(),
        )
        self.conv_module.add_module(
            "dropout_1", torch.nn.Dropout(p=dropout),
        )
        # Shape of intermediate feature maps
        shape = self._get_intermediate_shape()
        out = self.conv_module(torch.ones((1,) + tuple(shape[1:-1]) + (ssl_output_dim,)))
        dense_input_size = self._num_flat_features(out)
        # DENSE MODULE
        self.dense_module = torch.nn.Sequential()
        self.dense_module.add_module(
            "flatten", torch.nn.Flatten(),
        )
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=dense_input_size, n_neurons=dense_n_neurons,
            ),
        )
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))

    def _num_flat_features(self, x):
        """Returns the number of flattened features from a tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input feature map.
        """

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        # Reshape and permute input for SSL model processing
        bs, t, chann, _ = x.size()
        feats = self._prepare_for_ssl_model(x, bs, t, chann)
        # Apply SSL model to extract features
        feats = self.ssl_model(feats)
        feats = self._get_back_channel(feats, bs, chann)
        feats = self.conv_module(feats)
        feats = self.dense_module(feats)
        return feats

    def _prepare_for_ssl_model(self, x, bs, t, chann):
        """Permutes and reshapes the input tensor for processing by the SSL model."""
        if self.input_type == "BC":
            reshaped_x = x.permute(0, 2, 1, 3)
            reshaped_x = reshaped_x.reshape(bs * chann, t)
        elif self.input_type == "TC":
            reshaped_x = x.reshape(bs, chann * t)
        return reshaped_x

    def _get_back_channel(self, feats, bs, chann):
        if self.input_type == "TC":
            return feats.unsqueeze(-1)
        feats = feats.reshape(bs, chann, -1, feats.shape[-1])
        feats = feats.permute(0, 2, 1, 3)
        return feats

    def _get_intermediate_shape(self):
        # Shape of intermediate feature maps
        _, t, chann, _ = self.input_shape
        temp_sample = torch.ones((1,) + tuple(self.input_shape[1:-1]) + (1,))
        feats = self._prepare_for_ssl_model(temp_sample, 1, t, chann)
        # Apply SSL model to extract features
        feats = self.ssl_model(feats)
        feats = self._get_back_channel(feats, 1, chann)
        input_shape = feats.shape
        return input_shape

