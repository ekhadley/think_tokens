import torch as t
from torch import nn
from torch.nn import functional as F
import datasets
import numpy as np
import plotly
import plotly.express as px
from transformers import GPT2TokenizerFast, AutoModelForCausalLM, BitsAndBytesConfig

purple = '\x1b[38;2;255;0;255m'
blue = '\x1b[38;2;0;0;255m'
brown = '\x1b[38;2;128;128;0m'
cyan = '\x1b[38;2;0;255;255m'
lime = '\x1b[38;2;0;255;0m'
yellow = '\x1b[38;2;255;255;0m'
red = '\x1b[38;2;255;0;0m'
pink = '\x1b[38;2;255;51;204m'
orange = '\x1b[38;2;255;51;0m'
green = '\x1b[38;2;0;0;128m'
gray = '\x1b[38;2;127;127;127m'
magenta = '\x1b[38;2;128;0;128m'
white = '\x1b[38;2;255;255;255m'
bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'



class ModelConfig:
    def __init__(
            self,
            d_model:int = 512,
            d_mlp:int = 2048,
            d_head:int = 64,
            n_heads:int = 8,
            n_layers:int = 6,
            d_vocab:int = 50257,
        ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_mlp = d_mlp
        self.d_head = d_head
        self.d_vocab = d_vocab

class ThinkingModelConfig:
    def __init__(
            self,
            d_model:int = 512,
            d_mlp:int = 2048,
            d_head:int = 64,
            n_heads:int = 8,
            n_layers:int = 6,
            d_normal_vocab:int = 50257,
            d_thought_vocab:int = 50257,
        ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_mlp = d_mlp
        self.d_head = d_head
        self.d_normal_vocab = d_normal_vocab
        self.d_thought_vocab = d_thought_vocab
        self.d_vocab_total = d_normal_vocab + d_thought_vocab



class TrainingConfig:
    def __init__(
            self,
            batch_size:int = 32,
            lr:float = 3e-4,
            epochs:int = 1,
            seq_len:int = 512,
            warmup_steps:int = 1000,
            weight_decay:float = 1e-1,
            adam_beta1:float = 0.9,
            adam_beta2:float = 0.95,
            gamma:float = 0.95,
        ):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.seq_len = seq_len
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.gamma = gamma

def sampleLogits(logits: t.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, ) -> t.Tensor:
    """
    Sample from the logits of a model.
    Args:
        logits: The logits to sample from.
        temperature: The temperature to use for sampling.
        top_k: The number of top logits to keep.
        top_p: The cumulative probability to use for sampling.
    Returns:
        The sampled token.
    logits shape: (batch_size, sequence_length, vocab_size)
    """
    # Apply temperature
    logits = logits.squeeze() / temperature

    # Apply top-k
    if top_k > 0:
        indices_to_remove = logits < t.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')

    # Apply top-p
    if 0 < top_p < 1:
        sorted_logits, sorted_indices = t.sort(logits, descending=True)
        cumulative_probs = t.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')

    # Sample from the distribution
    probs = F.softmax(logits, dim=-1)
    return t.multinomial(probs, num_samples=1)

def tokenizeAndSaveDataset(tokenizer: GPT2TokenizerFast, training_cfg: TrainingConfig, dataset_title, dataset_name, save_path: str, fraction: float = 1.0, pad=False):
    dataset = datasets.load_dataset(dataset_title, name=dataset_name, split="train").train_test_split(fraction)['test']
    if pad:
        dataset = dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=training_cfg.seq_len))
    else:
        dataset = dataset.filter(lambda x: len(x['input_ids']) >= training_cfg.seq_len).map(lambda x: tokenizer(x['text'], truncation=True, max_length=training_cfg.seq_len))
    dataset.save_to_disk(save_path)
    return dataset

def loadTokenizedDataset(path: str):
    dataset = datasets.load_from_disk(path)
    dataset.set_format(type='torch')
    return dataset

def loadReferenceModel(model_name: str) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model




def line(logits: t.Tensor) -> None:
    plot = plotly.graph_objects.Figure()
    plot.add_trace(plotly.graph_objects.Scatter(y=logits.squeeze().cpu().numpy(), mode='lines', name='logits'))
    plot.show()


update_layout_set = {"xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor", "showlegend", "xaxis_tickmode", "yaxis_tickmode", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap", "coloraxis_showscale", "xaxis_tickangle", "yaxis_scaleanchor", "xaxis_tickfont", "yaxis_tickfont"}

update_traces_set = {"textposition"}

def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (t.Tensor, t.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")

def reorder_list_in_plotly_way(L: list, col_wrap: int):
    '''
    Helper function, because Plotly orders figures in an annoying way when there's column wrap.
    '''
    L_new = []
    while len(L) > 0:
        L_new.extend(L[-col_wrap:])
        L = L[:-col_wrap]
    return L_new

def imshow(tensor: t.Tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if ("size" in kwargs_pre) or ("shape" in kwargs_pre):
        size = kwargs_pre.pop("size", None) or kwargs_pre.pop("shape", None)
        kwargs_pre["height"], kwargs_pre["width"] = size
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    return_fig = kwargs_pre.pop("return_fig", False)
    text = kwargs_pre.pop("text", None)
    xaxis_tickangle = kwargs_post.pop("xaxis_tickangle", None)
    # xaxis_tickfont = kwargs_post.pop("xaxis_tickangle", None)
    static = kwargs_pre.pop("static", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "color_continuous_midpoint" not in kwargs_pre:
        kwargs_pre["color_continuous_midpoint"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post)
    if facet_labels:
        # Weird thing where facet col wrap means labels are in wrong order
        if "facet_col_wrap" in kwargs_pre:
            facet_labels = reorder_list_in_plotly_way(facet_labels, kwargs_pre["facet_col_wrap"])
        for i, label in enumerate(facet_labels):
            print(fig.layout.annotations)
            fig.layout.annotations[i]['text'] = label
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    if text:
        if tensor.ndim == 2:
            # if 2D, then we assume text is a list of lists of strings
            assert isinstance(text[0], list)
            assert isinstance(text[0][0], str)
            text = [text]
        else:
            # if 3D, then text is either repeated for each facet, or different
            assert isinstance(text[0], list)
            if isinstance(text[0][0], str):
                text = [text for _ in range(len(fig.data))]
        for i, _text in enumerate(text):
            fig.data[i].update(
                text=_text, 
                texttemplate="%{text}", 
                textfont={"size": 12}
            )
    # Very hacky way of fixing the fact that updating layout with xaxis_* only applies to first facet by default
    if xaxis_tickangle is not None:
        n_facets = 1 if tensor.ndim == 2 else tensor.shape[0]
        for i in range(1, 1+n_facets):
            xaxis_name = "xaxis" if i == 1 else f"xaxis{i}"
            fig.layout[xaxis_name]["tickangle"] = xaxis_tickangle
    return fig if return_fig else fig.show(renderer=renderer, config={"staticPlot": static})