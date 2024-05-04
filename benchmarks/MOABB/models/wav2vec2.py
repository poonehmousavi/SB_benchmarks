import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model
from typing import Optional, Tuple, Union


class wav2vec2(torch.nn.Module):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained wav2vec2.0models.

    Source paper wav2vec2.0: https://arxiv.org/abs/2006.11477
    Source paper Hubert: https://arxiv.org/abs/2106.07447
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    output_all_hiddens : bool (default: False)
        If True, the forward function outputs the hidden states from all transformer layers.
        For example wav2vec2-base has 12 transformer layers and the output is of shape (13, B, T, C),
        where a projection of the CNN output is added to the beginning.
        If False, the forward function outputs the hidden states only from the last transformer layer.

    Example
    -------
    >>> source = "facebook/wav2vec2-base-960h"
    >>> save_path= "tmp"
    >>> random_init=False
    >>> freeze=True
    >>> output_all_hiddens=False
    >>> model = wav2vec2(source,save_path ,random_init,freeze,output_all_hiddens)
    >>> inputs = torch.rand([10, 600])
    >>> outputs = model(inputs)
    torch.Size([10, 1, 768])
    """
    def __init__(self, source, save_path,random_init=False, freeze=True,output_all_hiddens=False):
        super(wav2vec2, self).__init__()
        
        self.freeze = freeze
        self.output_all_hiddens= output_all_hiddens
        
        if random_init:
            configuration = Wav2Vec2Config()
            # Initializing a model (with random weights) from the facebook/wav2vec2-base-960h style configuration
            self.model = Wav2Vec2Model(configuration)
        else:
            self.model  = Wav2Vec2Model.from_pretrained(source,cache_dir=save_path,)
        
        if freeze:
            self.freeze_model(self.model )
    
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

    def forward(self,  input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):
        with torch.set_grad_enabled(not self.freeze):
            out = self.model(input_values=input_values,
                                attention_mask=attention_mask,
                                mask_time_indices= mask_time_indices,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states)
        if self.output_all_hiddens:
            out = torch.stack(list(out.hidden_states), dim=0)
        else:
            out = out.last_hidden_state

        return out
    
source = "facebook/wav2vec2-base-960h"
save_path= "tmp"
model = wav2vec2(source,save_path ,random_init=True,freeze=True,output_all_hiddens=False)
inputs = torch.rand([10, 600])
outputs = model(inputs)
print(outputs.shape)