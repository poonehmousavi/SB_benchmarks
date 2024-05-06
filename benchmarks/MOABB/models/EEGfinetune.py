import torch
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
        random_init=False,
        freeze=True,
    ):
        super(Wav2Vec2ForFinetuning, self).__init__()

        self.max_duration_in_seconds = max_duration_in_seconds
        self.model_name_or_path = model_name_or_path
        self.freeze = freeze
        # get the model config used when initializing random model
        self.config = Wav2Vec2Config.from_pretrained(self.model_name_or_path)
        # initialize feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name_or_path
        )

        if random_init:
            # Initializing a model (with random weights) from the model_name_or_path style configuration
            self.model = Wav2Vec2Model(config=self.config)
        else:
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


class AttentionMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionMLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x):
        x = self.layers(x)
        att_w = torch.nn.functional.softmax(x, dim=2)
        return att_w


class eegFinetuneModel(torch.nn.Module):
    def __init__(self, ssl_model, attention_mlp, x_vector, classifier):
        super(eegFinetuneModel, self).__init__()
        self.ssl_model = ssl_model
        self.attention_mlp = attention_mlp
        self.x_vector = x_vector
        self.classifier = classifier

    def forward(self, x):
        """Processes EEG data through a pipeline of operations including SSL feature extraction, attention, and classification.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch, time, EEG channel, channel)

        Returns:
        --------
        torch.Tensor
            The logits output from the classifier with shape (batch, num_classes)
        """
        # Reshape and permute input for SSL model processing
        bs, t, chann, _ = x.size()
        permuted_x = self._prepare_for_ssl_model(x, bs, t, chann)
        # Apply SSL model to extract features
        feats = self.ssl_model(permuted_x)
        # Process channels with attention mechanism
        attended_feats = self._apply_attention(feats, bs, chann)
        # Process sequence with x_vector and Final classification
        logits = self._classify(attended_feats)
        return logits

    def _prepare_for_ssl_model(self, x, bs, t, chann):
        """Permutes and reshapes the input tensor for processing by the SSL model."""
        permuted_x = x.permute(0, 2, 1, 3)
        return permuted_x.reshape(bs * chann, t)

    def _apply_attention(self, feats, bs, chann):
        """Applies attention mechanism to the features."""
        feats = feats.reshape(bs, chann, -1, feats.shape[-1])
        feats = feats.permute(0, 2, 1, 3)
        att_w = self.attention_mlp(feats)
        return torch.matmul(att_w.transpose(2, -1), feats).squeeze(-2)

    def _classify(self, feats):
        """Passes the features through x-vector and classifier to get final logits."""
        h = self.x_vector(feats).squeeze(1)
        return self.classifier(h)
