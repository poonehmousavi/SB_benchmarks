import torch
import speechbrain as sb
from speechbrain.lobes.models.Xvector import Xvector,Classifier

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
    

class CustomModel(torch.nn.Module):
    def __init__(self, ssl_model,attention_mlp, x_vector,classifier):
        super(CustomModel, self).__init__()
        self.ssl_model = ssl_model
        self.attention_mlp = attention_mlp
        self.x_vector = x_vector
        self.classifier = classifier
    
    def forward(self,x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        bs, c, t, _ =x.size()
        feats = self.ssl_model(x.permute(0,2,1,3).reshape(bs*c,t))
        feats = feats.reshape(bs,c,-1, feats.shape[-1]).permute(0,2,1,3)
        att_w = self.attention_mlp(feats)
        feats = torch.matmul(att_w.transpose(2, -1), x).squeeze(-2)
        h = self.x_vector(feats).squeeze(1)
        logits = self.classifier(h)
        return logits
    
