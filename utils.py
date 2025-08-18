import random
import itertools
import os
from tqdm import trange
import tqdm
import torch as t
from torch import nn
from torch.nn import functional as F
import datasets
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
from transformers import GPT2TokenizerFast

def cfg_to_dict(cfg_obj):
    """
    Convert a dataclass-like config object to a dict, including dynamically
    added attributes not declared in the dataclass fields. Excludes private
    attributes (starting with '_') and callables.
    """
    # Start with declared dataclass fields (if present)
    base = {field.name: getattr(cfg_obj, field.name) for field in getattr(cfg_obj, "__dataclass_fields__", {}).values()}
    # Merge in any dynamically-added attributes
    for key, value in getattr(cfg_obj, "__dict__", {}).items():
        if key not in base and not key.startswith('_') and not callable(value):
            base[key] = value
    return base

purple = '\x1b[38;2;255;0;255m'
blue = '\x1b[38;2;0;0;255m'
brown = '\x1b[38;2;128;128;0m'
cyan = '\x1b[38;2;0;255;255m'
lime = '\x1b[38;2;0;255;0m'
yellow = '\x1b[38;2;255;255;0m'
red = '\x1b[38;2;255;0;0m'
pink = '\x1b[38;2;255;51;204m'
orange = '\x1b[38;2;255;51;0m'
green = '\x1b[38;2;5;170;20m'
gray = '\x1b[38;2;127;127;127m'
magenta = '\x1b[38;2;128;0;128m'
white = '\x1b[38;2;255;255;255m'
bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

t.backends.cuda.enable_flash_sdp(enabled=True)
t.set_printoptions(sci_mode=False, linewidth=200, edgeitems=4)

def tokenizeAndSaveDataset(tokenizer: GPT2TokenizerFast, min_seq_len: int, dataset_title: str, dataset_name: str, save_name: str, fraction: float = 1.0, pad=False):
    dataset = datasets.load_dataset(dataset_title, name=dataset_name, split="train").train_test_split(fraction)['test']
    if pad:
        dataset = dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=min_seq_len))
    else:
        dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, max_length=min_seq_len)).filter(lambda x: len(x['input_ids']) >= min_seq_len)
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

# plotting stuff
def line(logits: t.Tensor, ymin: float|None = None, ymax: float|None = None) -> None:
    if ymax is not None:
        assert ymin is not None, "If ymax is specified, ymin must also be specified."
        plot = plotly.graph_objects.Figure(layout_yaxis_range=[ymin, ymax])
    else:
        assert ymin is None, "If ymin is specified, ymax must also be specified."
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


def chess_coord_to_int(coord):
    """
    Convert chess coordinate (like 'a1', 'h8') to integer.
    a1=0, b1=1, c1=2, ..., h1=7, a2=8, b2=9, ..., h8=63
    """
    if len(coord) != 2:
        raise ValueError(f"Invalid coordinate: {coord}")
    
    file_char = coord[0].lower()
    rank_char = coord[1]
    
    if file_char < 'a' or file_char > 'h':
        raise ValueError(f"Invalid file: {file_char}")
    if rank_char < '1' or rank_char > '8':
        raise ValueError(f"Invalid rank: {rank_char}")
    
    file_idx = ord(file_char) - ord('a')  # a=0, b=1, ..., h=7
    rank_idx = int(rank_char) - 1         # 1=0, 2=1, ..., 8=7
    
    return rank_idx * 8 + file_idx

def int_to_chess_coord(coord_int):
    """
    Convert integer back to chess coordinate.
    0=a1, 1=b1, 2=c1, ..., 7=h1, 8=a2, 9=b2, ..., 63=h8
    """
    if coord_int < 0 or coord_int > 63:
        raise ValueError(f"Invalid coordinate integer: {coord_int}")
    
    rank_idx = coord_int // 8
    file_idx = coord_int % 8
    
    file_char = chr(ord('a') + file_idx)
    rank_char = str(rank_idx + 1)
    
    return file_char + rank_char

def lan_move_to_tokens(lan_move):
    """
    Convert a LAN move (like 'e2e4') to two tokens [start_square, end_square].
    """
    if len(lan_move) != 4:
        raise ValueError(f"Invalid LAN move: {lan_move}")
    
    start_coord = lan_move[:2]
    end_coord = lan_move[2:]
    
    start_token = chess_coord_to_int(start_coord)
    end_token = chess_coord_to_int(end_coord)
    
    return [start_token, end_token]

def tokenize_chess_games(chess_df):
    """
    Tokenize chess games into coordinate integers.
    Each move becomes 2 tokens: [start_square, end_square]
    """
    print("Tokenizing chess games...")
    tokenized_games = []
    failed_games = 0
    
    for idx, row in tqdm.tqdm(chess_df.iterrows(), total=len(chess_df), desc="Tokenizing games"):
        try:
            if 'lan' in chess_df.columns and pd.notna(row['lan']):
                # Use LAN notation
                moves = row['lan'].split()
                game_tokens = []
                
                for move in moves:
                    try:
                        tokens = lan_move_to_tokens(move)
                        game_tokens.extend(tokens)
                    except ValueError:
                        # Skip invalid moves
                        continue
                
                tokenized_games.append(np.array(game_tokens, dtype=np.int8))
            else:
                # No LAN data available
                tokenized_games.append(np.array([], dtype=np.int8))
                failed_games += 1
                
        except Exception as e:
            # Failed to tokenize this game
            tokenized_games.append(np.array([], dtype=np.int8))
            failed_games += 1
    
    print(f"Successfully tokenized {len(tokenized_games) - failed_games} games")
    print(f"Failed to tokenize {failed_games} games")
    
    return tokenized_games