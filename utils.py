from dataclasses import dataclass
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

t.backends.cuda.enable_flash_sdp(enabled=True)

@dataclass
class ModelConfig:
    d_model: int = 512
    seq_len: int = 512
    d_mlp: int = 2048
    d_head: int = 64
    n_heads: int = 8
    n_layers: int = 6
    d_vocab: int = 50257
    seq_len: int = 512

@dataclass
class ThinkingModelConfig:
    d_model: int = 512
    seq_len: int = 512
    d_mlp: int = 2048
    d_head: int = 64
    n_heads: int = 8
    n_layers: int = 8
    d_normal_vocab: int = 50257
    d_thought_vocab: int = 2048
    
    def __post_init__(self):
        self.d_vocab_total = self.d_normal_vocab + self.d_thought_vocab

@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    gamma: float = 0.95

    think_len: int = 8
    group_size: int = 16
    think_reward_weight: float = 0.0
    entropy_reward_weight: float = 0.0
    prob_force_end_thought: float = 1.0
    eps_decay: float = 0.999995
    eps_min: float = 0.05


def sampleLogits(logits: t.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, ) -> t.Tensor:
    logits = logits.squeeze() / temperature
    if top_k > 0:
        indices_to_remove = logits < t.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
    if 0 < top_p < 1:
        sorted_logits, sorted_indices = t.sort(logits, descending=True)
        cumulative_probs = t.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    return t.multinomial(probs, num_samples=1)
def sampleLogprobs(logprobs: t.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, ) -> t.Tensor:
    logprobs = logprobs.squeeze() / temperature
    if top_k > 0:
        indices_to_remove = logprobs < t.topk(logprobs, top_k)[0][..., -1, None]
        logprobs[indices_to_remove] = -float('Inf')
    if 0 < top_p < 1:
        sorted_logits, sorted_indices = t.sort(logprobs, descending=True)
        cumulative_probs = t.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logprobs[indices_to_remove] = -float('Inf')
    probs = F.softmax(logprobs, dim=-1)
    return t.multinomial(probs, num_samples=1)

def tokenizeAndSaveDataset(tokenizer: GPT2TokenizerFast, cfg: ModelConfig, dataset_title, dataset_name, save_name: str, fraction: float = 1.0, pad=False):
    dataset = datasets.load_dataset(dataset_title, name=dataset_name, split="train").train_test_split(fraction)['test']
    if pad:
        dataset = dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=cfg.seq_len))
    else:
        dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, max_length=cfg.seq_len)).filter(lambda x: len(x['input_ids']) >= cfg.seq_len)
    dataset.set_format(type='torch')
    dataset.save_to_disk(f"datasets/{save_name}")
    return dataset

def loadModel(model_path: str, model_class, cfg):
    model = model_class(cfg)
    model.load_state_dict(t.load(model_path, weights_only=True))
    return model

def loadTokenizedDataset(name: str):
    dataset = datasets.load_from_disk(f"datasets/{name}")
    dataset.set_format(type='torch')
    return dataset

def LoadNormalModelAsThinking(normal_model: nn.Module, thinking_model_class, thinking_cfg: ThinkingModelConfig):
    model = thinking_model_class(thinking_cfg)

    # Copy transformer blocks and ln_f
    model.blocks.load_state_dict(normal_model.blocks.state_dict())
    model.ln_f.load_state_dict(normal_model.ln_f.state_dict())

    # Resize embeddings and unembedding
    resize_embedding(model.pos_embed, normal_model.pos_embed)
    resize_embedding(model.embed, normal_model.embed)
    resize_embedding(model.unembed, normal_model.unembed)

    # Optionally copy tokenizer if needed
    if hasattr(normal_model, 'tokenizer'):
        model.tokenizer = normal_model.tokenizer

    return model







# plotting stuff
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

def resize_embedding(target_embed, source_embed):
    """
    Resize (expand or shrink) the target embedding weights to match the source as much as possible.
    Copies overlapping rows, initializes new rows if needed.
    """
    with t.no_grad():
        old_weight = source_embed.weight.data
        new_weight = target_embed.weight.data
        min_rows = min(old_weight.shape[0], new_weight.shape[0])
        new_weight[:min_rows] = old_weight[:min_rows]
        if new_weight.shape[0] > old_weight.shape[0]:
            nn.init.normal_(new_weight[old_weight.shape[0]:], mean=0.0, std=0.02)
        target_embed.weight.data = new_weight