import speechbrain as sb
import torch


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


class eeg_xvectorFinetuneModel(torch.nn.Module):
    def __init__(self, ssl_model, attention_mlp, x_vector, classifier):
        super(eeg_xvectorFinetuneModel, self).__init__()
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


class eeg_linearFinetuneModel(torch.nn.Module):
    def __init__(
        self,
        ssl_model,
        attention_mlp,
        input_shape,
        dense_max_norm,
        out_n_neurons,
    ):
        super(eeg_linearFinetuneModel, self).__init__()
        self.ssl_model = ssl_model
        self.attention_mlp = attention_mlp

        _, t, chann, _ = input_shape
        temp_sample = torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        permuted_x = self._prepare_for_ssl_model(temp_sample, 1, t, chann)
        # Apply SSL model to extract features
        feats = self.ssl_model(permuted_x)
        # Process channels with attention mechanism
        attended_feats = self._apply_attention(feats, 1, chann)
        dense_input_size = self._num_flat_features(attended_feats)
        # DENSE MODULE
        self.dense_module = torch.nn.Sequential()
        self.dense_module.add_module(
            "flatten",
            torch.nn.Flatten(),
        )
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=dense_input_size,
                n_neurons=out_n_neurons,
                max_norm=dense_max_norm,
            ),
        )
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))

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
        logits = self.dense_module(attended_feats)
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
