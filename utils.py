import torch as t
from torch import nn
from torch.nn import functional as F

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
        ):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.seq_len = seq_len
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2

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