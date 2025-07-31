from dataclasses import dataclass
import pandas as pd
import torch as t
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from transformers import GPT2TokenizerFast

from utils import *



@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 3e-4
    think_lr: float = None
    answer_lr: float = None
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

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}


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
    
    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.linear1 = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2: x = x.unsqueeze(0)
        seq_len = x.shape[1]
        attn_mask = t.triu(t.ones((seq_len, seq_len)), diagonal=1).bool()
        attn_output, _ = self.attn(x, x, x, is_causal=True, attn_mask=attn_mask)
        x = self.norm1(x + attn_output)
        ff_output = self.linear2(self.act(self.linear1(x)))
        x = self.norm2(x + ff_output)
        return x

class GPT2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(GPT2, self).__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)

        self.tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
    
    def encode(self, text):
        return self.tokenizer.tokenize(text)
    def decode(self, tokens):
        return self.tokenizer.batch_decode(tokens)
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1: x = x.unsqueeze(0)
        x = self.embed(x) + self.pos_embed(t.arange(x.shape[1], device=x.device)).unsqueeze(0)
        for i, block in enumerate(self.blocks):
            x = block(x)
        x = self.ln_f(x)
        x = self.unembed(x) # untied
        #x = F.linear(x, self.embed.weight) # tied
        return x
    def yap(self, prompt: str, max_length: int = 50):
        with t.no_grad():
            tokens = t.tensor(self.tokenizer(prompt).input_ids).squeeze()
            for _ in range(max_length):
                logits = self(tokens).squeeze()
                next_token = sampleLogits(logits[-1], 0.9)
                tokens = t.cat([tokens, next_token], dim=-1)
        return "".join(self.tokenizer.batch_decode(tokens))


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
    
    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
        d['d_vocab_total'] = self.d_vocab_total
        return d

class GPT2Thinking(nn.Module):
    def __init__(self, cfg: ThinkingModelConfig):
        super(GPT2Thinking, self).__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.embed = nn.Embedding(cfg.d_vocab_total, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab_total, bias=False)
        
        self.end_thought = cfg.d_vocab_total - 1
        self.tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")

    def encode(self, text):
        return t.tensor(self.tokenizer(text).input_ids)
    def decode(self, tokens):
        return self.tokenizer.batch_decode(tokens)
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1: x = x.unsqueeze(0)
        x = self.embed(x) + self.pos_embed(t.arange(x.shape[1])).unsqueeze(0)
        for i, block in enumerate(self.blocks):
            x = block(x)
        x = self.ln_f(x)
        x = self.unembed(x)
        return x
    def yap(self, prompt: str, max_length: int = 50) -> Tensor:
        with t.inference_mode():
            tokens = t.tensor(self.tokenizer(prompt).input_ids)
            for _ in range(max_length):
                next_token = 0
                while next_token != self.end_thought:
                    logits = self.forward(tokens)
                    next_token = logits[0, -1, self.cfg.d_normal_vocab:].argmax(-1).item() + self.cfg.d_normal_vocab
                    tokens = t.cat((tokens, t.tensor([next_token], device=tokens.device)), dim=0)
                    if tokens.shape[-1] >= self.cfg.seq_len: return tokens
                next_token = logits[0, -1, :self.cfg.d_normal_vocab].argmax(-1).item()
                tokens = t.cat((tokens, t.tensor([next_token], device=tokens.device)), dim=0)
        return tokens.squeeze()
    def seqStr(self, seq: Tensor) -> str:
        out = ""
        seq = seq.squeeze()
        for token in seq:
            if token <= self.cfg.d_normal_vocab:
                out += f"{blue}{self.tokenizer.decode(token)}{endc}"
            elif token == self.end_thought:
                out += f"{magenta}<end_thought>{endc}"
            else:
                out += f"{cyan}<think{token.item() - self.cfg.d_normal_vocab}>{endc}"
        return out
    def printSeq(self, seq: Tensor) -> None:
        print("\n", self.seqStr(seq))


@dataclass
class SplitModelConfig:
    d_model: int = 512
    seq_len: int = 512
    d_mlp: int = 2048
    d_head: int = 64
    n_heads: int = 8
    n_layers: int = 6
    seq_len: int = 512

    d_vocab_in: int = 50257
    d_vocab_out: int = 50257
    d_thought_vocab: int = 2048
    
    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}

class GPT2SplitModel(nn.Module):
    def __init__(self, cfg: SplitModelConfig):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.embed = nn.Embedding(cfg.d_vocab_in, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab_out, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1: x = x.unsqueeze(0)
        x = self.embed(x) + self.pos_embed(t.arange(x.shape[1], device=x.device)).unsqueeze(0)
        for i, block in enumerate(self.blocks):
            x = block(x)
        x = self.ln_f(x)
        x = self.unembed(x)
        return x

    def _forward_embeddings(self, embedding: Tensor) -> Tensor:
        if embedding.ndim == 1: embedding = embedding.unsqueeze(0)
        x = embedding + self.pos_embed(t.arange(embedding.shape[1], device=embedding.device)).unsqueeze(0)
        for i, block in enumerate(self.blocks):
            x = block(x)
        x = self.ln_f(x)
        x = self.unembed(x)
        return x
    def forward_one_hot(self, toks_one_hot: Tensor) -> Tensor:
        assert toks_one_hot.ndim == 3, "Input should be (batch, seq_len, d_vocab_in)"
        embedding = toks_one_hot @ self.embed.weight
        x = embedding + self.pos_embed(t.arange(embedding.shape[1], device=embedding.device)).unsqueeze(0)
        for i, block in enumerate(self.blocks):
            x = block(x)
        x = self.ln_f(x)
        x = self.unembed(x)
        return x

@dataclass
class RecycleModelConfig:
    d_model: int = 512
    seq_len: int = 512
    d_mlp: int = 2048
    d_head: int = 64
    n_heads: int = 8
    n_layers: int = 6
    d_vocab: int = 50257
    seq_len: int = 512
    recycle_layer: int = None

    def __post_init__(self):
        if self.recycle_layer is None:
            self.recycle_layer = self.n_layers - 1
    
    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}

class Recycler(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(Recycler, self).__init__()
        assert cfg.recycle_layer < cfg.n_layers, "Recycle layer must be less than total layers"
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)
        self.tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
        
    # forward passes like an rnn. Takes a continuous 2d context of previous text and a single new token, outputs the new context vector and a distn for next token prediction
    # the context vector is one of the later layer hidden states (residual stream vectors) for the last token position. Context is combined by simple concatenation.
    def forward(self, token: Tensor = None, context: Tensor = None, need_distn: bool = True) -> tuple[Tensor, Tensor] | Tensor: 
        assert token is not None or context is not None, "Either a first token or an context state must be provided"
        if token.ndim == 1: token = token.unsqueeze(0)
        assert token.ndim == 2, "Token should be single item or 1D tensor"

        token_embed = self.embed(token) if token is not None else None
        if context is not None:
            if context.ndim == 2: context = context.unsqueeze(0)  # Ensure context is 3D
            assert context.ndim == 3, "Context should be (batch, seq, d_model) or (seq, d_model)"

        if context is not None and token_embed is not None:
            x = t.cat([context, token_embed], dim=1)  # Concatenate context with the new token embedding
        elif context is None:
            x = token_embed
        else:
            x = context

        seq_len = x.shape[1]

        x += self.pos_embed(t.arange(seq_len, device=x.device)).unsqueeze(0) # Add positional embeddings
        for i, block in enumerate(self.blocks):
            x = block(x)
            if seq_len >= 10 and seq_len < 16 and i ==3:
                imshow(x[0], title=f"seq_len {seq_len}, layer {i}")
            if i == self.cfg.recycle_layer:
                new_context = x[:, -1, :]  # Store the context vector from the specified layer
                if not need_distn: return new_context # if we don't need the distribution, return the context vector immediately
        x = self.ln_f(x[:, -1, :]) # Toss unecessary context.
        distn = self.unembed(x) # unembed the last position residual stream to get next token distn
        return new_context, distn

    def forward2(self, token: Tensor = None, context: Tensor = None, need_distn: bool = True) -> tuple[Tensor, Tensor] | Tensor: 
        assert token is not None or context is not None, "Either a first token or an context state must be provided"
        if token.ndim == 1: token = token.unsqueeze(0)
        assert token.ndim == 2, "Token should be single item or 1D tensor"

        token_embed = self.embed(token) if token is not None else None
        if context is not None:
            if context.ndim == 2: context = context.unsqueeze(0)  # Ensure context is 3D
            assert context.ndim == 3, "Context should be (batch, seq, d_model) or (seq, d_model)"
        if context is not None and token_embed is not None:
            x = t.cat([context, token_embed], dim=1)  # Concatenate context with the new token embedding
        elif context is None: x = token_embed
        else: x = context
        seq_len = x.shape[1]

        x += self.pos_embed(t.arange(seq_len, device=x.device)).unsqueeze(0)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.cfg.recycle_layer:
                if not need_distn: return x
        x = self.ln_f(x)
        distn = self.unembed(x[:, -1, :])
        return x, distn

    # forward pass for a single string of tokens.
    # Has to sequentially process  each token to accumulate hidden state context.
    # Returns full context for the sequence.
    def process_seq(self, tokens: Tensor) -> Tensor:
        if tokens.ndim == 1: tokens = tokens.unsqueeze(0)  # Ensure tokens is 2D
        bsize = tokens.shape[0]
        seq_len = tokens.shape[1]
        ctx = tokens.new_zeros((bsize, seq_len, self.cfg.d_model))
        new_ctx = self.forward(token = tokens[:, 0], context = None, need_distn=False) # Get initial context for the first token
        for i in range(1, seq_len):
            new_ctx = self.forward(token = tokens[:, i], context = new_ctx, need_distn=False) # Process each
            ctx[:, i] = new_ctx
        return ctx



@dataclass
class ContThinkingModelConfig:
    d_model: int = 512
    seq_len: int = 512
    d_mlp: int = 2048
    d_head: int = 64
    n_heads: int = 8
    n_layers: int = 6
    d_vocab: int = 50257
    seq_len: int = 512
    recycle_layer: int = None

    def __post_init__(self):
        if self.recycle_layer is None:
            self.recycle_layer = self.n_layers - 1
    
    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}

class ContThinkingModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(ContThinkingModel, self).__init__()
        assert cfg.recycle_layer < cfg.n_layers, "Recycle layer must be less than total layers"
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)
        self.tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
        
    # forward passes like an rnn. Takes a continuous 2d context of previous text and a single new token, outputs the new context vector and a distn for next token prediction
    # the context vector is one of the later layer hidden states (residual stream vectors) for the last token position. Context is combined by simple concatenation.
    def forward(self, token: Tensor = None, context: Tensor = None, need_distn: bool = True) -> tuple[Tensor, Tensor] | Tensor: 
        assert token is not None or context is not None, "Either a first token or an context state must be provided"
        if token.ndim == 1: token = token.unsqueeze(0)
        assert token.ndim == 2, "Token should be single item or 1D tensor"

        token_embed = self.embed(token) if token is not None else None
        if context is not None:
            if context.ndim == 2: context = context.unsqueeze(0)  # Ensure context is 3D
            assert context.ndim == 3, "Context should be (batch, seq, d_model) or (seq, d_model)"

        if context is not None and token_embed is not None:
            x = t.cat([context, token_embed], dim=1)  # Concatenate context with the new token embedding
        elif context is None:
            x = token_embed
        else:
            x = context

        seq_len = x.shape[1]

        x += self.pos_embed(t.arange(seq_len, device=x.device)).unsqueeze(0) # Add positional embeddings
        for i, block in enumerate(self.blocks):
            x = block(x)
            if seq_len >= 10 and seq_len < 16 and i ==3:
                imshow(x[0], title=f"seq_len {seq_len}, layer {i}")
            if i == self.cfg.recycle_layer:
                new_context = x[:, -1, :]  # Store the context vector from the specified layer
                if not need_distn: return new_context # if we don't need the distribution, return the context vector immediately
        x = self.ln_f(x[:, -1, :]) # Toss unecessary context.
        distn = self.unembed(x) # unembed the last position residual stream to get next token distn
        return new_context, distn

    def forward2(self, token: Tensor = None, context: Tensor = None, need_distn: bool = True) -> tuple[Tensor, Tensor] | Tensor: 
        assert token is not None or context is not None, "Either a first token or an context state must be provided"
        if token.ndim == 1: token = token.unsqueeze(0)
        assert token.ndim == 2, "Token should be single item or 1D tensor"

        token_embed = self.embed(token) if token is not None else None
        if context is not None:
            if context.ndim == 2: context = context.unsqueeze(0)  # Ensure context is 3D
            assert context.ndim == 3, "Context should be (batch, seq, d_model) or (seq, d_model)"
        if context is not None and token_embed is not None:
            x = t.cat([context, token_embed], dim=1)  # Concatenate context with the new token embedding
        elif context is None: x = token_embed
        else: x = context
        seq_len = x.shape[1]

        x += self.pos_embed(t.arange(seq_len, device=x.device)).unsqueeze(0)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.cfg.recycle_layer:
                if not need_distn: return x
        x = self.ln_f(x)
        distn = self.unembed(x[:, -1, :])
        return x, distn

    # forward pass for a single string of tokens.
    # Has to sequentially process  each token to accumulate hidden state context.
    # Returns full context for the sequence.
    def process_seq(self, tokens: Tensor) -> Tensor:
        if tokens.ndim == 1: tokens = tokens.unsqueeze(0)  # Ensure tokens is 2D
        bsize = tokens.shape[0]
        seq_len = tokens.shape[1]
        ctx = tokens.new_zeros((bsize, seq_len, self.cfg.d_model))
        new_ctx = self.forward(token = tokens[:, 0], context = None, need_distn=False) # Get initial context for the first token
        for i in range(1, seq_len):
            new_ctx = self.forward(token = tokens[:, i], context = new_ctx, need_distn=False) # Process each
            ctx[:, i] = new_ctx
        return ctx