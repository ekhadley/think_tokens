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
        return cfg_to_dict(self)


@dataclass
class ModelConfig:
    d_model: int = 512
    seq_len: int = 512
    d_mlp: int = 2048
    n_heads: int = 8
    n_layers: int = 6
    d_vocab: int = 50257
    seq_len: int = 512
    
    def to_dict(self):
        return cfg_to_dict(self)

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
        # Explicit causal mask for compatibility across PyTorch versions
        seq_len = x.shape[1]
        attn_mask = t.triu(t.ones((seq_len, seq_len), device=x.device, dtype=t.bool), diagonal=1)
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
        return x

@dataclass
class ThinkingModelConfig:
    d_model: int = 512
    seq_len: int = 512
    d_mlp: int = 2048
    n_heads: int = 8
    n_layers: int = 8
    d_normal_vocab: int = 50257
    d_thought_vocab: int = 2048
    
    def __post_init__(self):
        self.d_vocab_total = self.d_normal_vocab + self.d_thought_vocab
    
    def to_dict(self):
        return cfg_to_dict(self)

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
    def forward_one_hot(self, toks_one_hot: Tensor) -> Tensor:
        assert toks_one_hot.ndim == 3, "Input should be (batch, seq_len, d_vocab_in)"
        embedding = toks_one_hot @ self.embed.weight
        x = embedding + self.pos_embed(t.arange(embedding.shape[1], device=embedding.device)).unsqueeze(0)
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
    n_heads: int = 8
    n_layers: int = 6
    seq_len: int = 512

    d_vocab_in: int = 50257
    d_vocab_out: int = 50257
    d_thought_vocab: int = 2048
    
    def to_dict(self):
        return cfg_to_dict(self)

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
    n_heads: int = 4
    n_layers: int = 6
    d_vocab: int = 50257
    seq_len: int = 512
    recycle_layer: int = None

    def __post_init__(self):
        if self.recycle_layer is None:
            self.recycle_layer = self.n_layers - 1
    
    def to_dict(self):
        return cfg_to_dict(self)

class Recycler(nn.Module):
    def __init__(self, cfg: RecycleModelConfig):
        super(Recycler, self).__init__()
        assert cfg.recycle_layer < cfg.n_layers, "Recycle layer must be less than total layers"
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)
        self.tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
        self.mixing_attn = nn.MultiheadAttention(cfg.d_model, 1, batch_first=True)
        self.recycler_block = TransformerBlock(cfg)
        
    # forward passes like an rnn. Takes a continuous 2d context of previous text and a single new token, outputs the new context vector and a distn for next token prediction
    # the context vector is one of the later layer hidden states (residual stream vectors) for the last token position. Context is combined by simple concatenation.
    def forward_replace_embed(self, next_tokens: Tensor = None, context: Tensor = None, need_distn: bool = True) -> tuple[Tensor, Tensor] | Tensor: 
        self.cfg.forward_type = "replace_embed"
        assert next_tokens is not None or context is not None, "Either a first token or an context state must be provided"
        if next_tokens is not None:
            assert next_tokens.ndim == 1, "Token should be single item or 1D tensor"
            next_tokens = next_tokens.reshape(-1, 1)

        token_embed = self.embed(next_tokens) if next_tokens is not None else None
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
        # Add positional embeddings only to the new token position when context exists,
        # to avoid re-adding position to recycled hidden states.
        x[:, -1, :] = x[:, -1, :] + self.pos_embed(t.tensor([seq_len - 1], device=x.device)).squeeze()
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.cfg.recycle_layer:
                new_context = x[:, -1, :]  # Store the context vector from the specified layer
                if not need_distn: return new_context # if we don't need the distribution, return the context vector immediately
        x = self.ln_f(x[:, -1, :]) # Toss unecessary context.
        distn = self.unembed(x) # unembed the last position residual stream to get next token distn
        return new_context, distn

    # fully recurrent version. Instead of returning a single context vector for each forward pass, it returns an entirely different hidden state
    def forward_full_context_replace(self, next_tokens: Tensor = None, context: Tensor = None, need_distn: bool = True) -> tuple[Tensor, Tensor] | Tensor: 
        self.cfg.forward_type = "full_context_replace"
        assert next_tokens is not None or context is not None, "Either a first token or an context state must be provided"
        if next_tokens is not None:
            assert next_tokens.ndim == 1, "Token should be single item or 1D tensor"
            next_tokens = next_tokens.reshape(-1, 1)

        token_embed = self.embed(next_tokens) if next_tokens is not None else None
        if context is not None:
            if context.ndim == 2: context = context.unsqueeze(0)  # Ensure context is 3D
            assert context.ndim == 3, "Context should be (batch, seq, d_model) or (seq, d_model)"
        if context is not None and token_embed is not None:
            x = t.cat([context, token_embed], dim=1)  # Concatenate context with the new token embedding
        elif context is None: x = token_embed
        else: x = context
        seq_len = x.shape[1]
        # Avoid re-adding positional embeddings to recycled hidden states
        x[:, -1, :] = x[:, -1, :] + self.pos_embed(t.tensor([seq_len - 1], device=x.device)).squeeze()
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.cfg.recycle_layer:
                if not need_distn: return x
        x = self.ln_f(x)
        distn = self.unembed(x[:, -1, :])
        return x, distn

    # like forward_replace_embed but uses an attention layer like a gate, selectively crossing over the recycled context and the normal token embeddings.
    # basically pre-cats all the recycled context to the current token embeddings
    def forward_attn_gate(self, tokens: Tensor, context: Tensor = None, need_distn: bool = True, show_pattern: bool = False) -> tuple[Tensor, Tensor] | Tensor: 
        self.cfg.forward_type = "attn_gate"
        assert tokens.ndim == 2, "Tokens should be (batch, seq_len)"
        
        token_seq_len = tokens.shape[1]
        token_embeds = self.embed(tokens)  # (batch, token_seq_len, d_model)
        token_embeds = token_embeds + self.pos_embed(t.arange(token_seq_len, device=token_embeds.device)).unsqueeze(0)  # Add positional embeddings to the token embeddings

        if context is not None:
            if context.ndim == 2: context = context.unsqueeze(0)  # Ensure context is 3D
            assert context.ndim == 3, "Context should be (batch, seq, d_model)"
            context_seq_len = context.shape[1]
            combined = t.cat([context, token_embeds], dim=1)  # (batch, total_seq_len, d_model)
            total_seq_len = combined.shape[1]
            attn_mask = t.triu(t.ones((total_seq_len, total_seq_len), device=combined.device, dtype=t.bool), diagonal=1)
            mixed_embeds, weights = self.mixing_attn(combined, combined, combined, is_causal=True, attn_mask=attn_mask, need_weights=show_pattern)
            if show_pattern:
                imshow(weights[0])
                print(pink, self.mixing_attn.in_proj_weight[0, :10], endc)
                print(purple, self.blocks[0].attn.in_proj_weight[0, :10], endc)
            
            x = mixed_embeds[:, context_seq_len:, :]  # (batch, token_seq_len, d_model)
            #x = mixed_embeds[:, :context_seq_len:, :]  # (batch, token_seq_len, d_model)
        else:
            x = token_embeds
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.cfg.recycle_layer:
                new_context = x[:, -1, :]  # Get the last position's residual stream vector
                if not need_distn: 
                    return new_context
        
        x = self.ln_f(x[:, -1, :])  # Only process the last position
        distn = self.unembed(x)  # Get next token distribution
        return new_context, distn

    # the input context is simply the embedding for token 0, recycled vector for token 0, embedding for token 1, recycled vector for token 1, etc.
    # each sequence position is now two sequence positions. 
    def forward_interleaved_embeddings(self, next_tokens: Tensor = None, context: Tensor = None, need_distn: bool = True) -> tuple[Tensor, Tensor] | Tensor: 
        self.cfg.forward_type = "interleaved_embeddings"
        assert next_tokens is not None or context is not None, "Either a first token or an context state must be provided"
        if next_tokens is not None:
            assert next_tokens.ndim == 1, "Token should be single item or 1D tensor"
            next_tokens = next_tokens.reshape(-1, 1)

        token_embed = self.embed(next_tokens) if next_tokens is not None else None

        if context is not None and token_embed is not None:
            x = t.cat([context, token_embed], dim=1)  # Concatenate context with the new token embedding
        elif context is None:
            x = token_embed
        elif token_embed is None:
            if context.ndim == 2: context = context.unsqueeze(0)  # Ensure context is 3D
            assert context.ndim == 3, "Context should be (batch, seq, d_model) or (seq, d_model)"
            x = context
        else:
            raise ValueError("Either token or context must be provided")

        seq_len = x.shape[1]
        seq_indices = t.arange(seq_len//2, device=x.device)
        x[:, seq_indices*2, :] = x[:, seq_indices*2, :] + self.pos_embed(seq_indices).unsqueeze(0)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.cfg.recycle_layer:
                new_context = x[:, -1, :]  # Store the context vector from the specified layer
                if not need_distn: return new_context # if we don't need the distribution, return the context vector immediately
        x = self.ln_f(x[:, -1, :]) # Toss unecessary context.
        distn = self.unembed(x) # unembed the last position residual stream to get next token distn
        return new_context, distn

    # like attn_gate but with interleaved embeddings.
    def forward_attn_gate_interleaved(self, tokens: Tensor, context: Tensor = None, need_distn: bool = True) -> tuple[Tensor, Tensor] | Tensor: 
        self.cfg.forward_type = "attn_gate_interleaved"
        assert tokens.ndim == 2, "Tokens should be (batch, seq_len)"

        token_seq_len = tokens.shape[1]
        token_embeds = self.embed(tokens)  # (batch, token_seq_len, d_model)
        token_embeds = token_embeds + self.pos_embed(t.arange(token_seq_len, device=token_embeds.device)).unsqueeze(0)

        if context is not None:
            if context.ndim == 2: context = context.unsqueeze(0)
            assert context.ndim == 3, "Context should be (batch, seq, d_model)"
            context_seq_len = context.shape[1]
            #print(red, context.shape, blue, token_embeds.shape, endc)
            combined = t.cat([context, token_embeds], dim=1)  # (batch, total_seq_len, d_model)
            #print(green, combined.shape, endc)
            total_seq_len = combined.shape[1]
            attn_mask = t.triu(t.ones((total_seq_len, total_seq_len), device=combined.device, dtype=t.bool), diagonal=1)
            mixed_embeds, _ = self.mixing_attn(combined, combined, combined, is_causal=True, attn_mask=attn_mask)
            x = mixed_embeds[:, context_seq_len:, :]  # (batch, token_seq_len, d_model)
        else:
            x = token_embeds

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.cfg.recycle_layer - 1:
                recycler_stream = self.recycler_block(x)
                new_context = recycler_stream[:, -1, :]  # Use the separate recycler block's residual stream
                if not need_distn:
                    return new_context

        x = self.ln_f(x[:, -1, :])  # Only process the last position for logits
        distn = self.unembed(x)
        return new_context, distn
    
    # like attn_gate_interleaved but with a separate, alternative recycler block at recycle_layer whose output we actually recycle.
    def forward_recycler_block_interleaved(self, next_tokens: Tensor = None, context: Tensor = None, need_distn: bool = True) -> tuple[Tensor, Tensor] | Tensor: 
        self.cfg.forward_type = "recycler_block_interleaved"
        assert next_tokens is not None or context is not None, "Either a first token or an context state must be provided"
        if next_tokens is not None:
            assert next_tokens.ndim == 1, "Token should be single item or 1D tensor"
            next_tokens = next_tokens.reshape(-1, 1)

        token_embed = self.embed(next_tokens) if next_tokens is not None else None

        if context is not None and token_embed is not None:
            x = t.cat([context, token_embed], dim=1)
        elif context is None:
            x = token_embed
        elif token_embed is None:
            if context.ndim == 2: context = context.unsqueeze(0)
            assert context.ndim == 3, "Context should be (batch, seq, d_model) or (seq, d_model)"
            x = context
        else:
            raise ValueError("Either token or context must be provided")

        seq_len = x.shape[1]
        seq_indices = t.arange(seq_len//2, device=x.device)
        x[:, seq_indices*2, :] = x[:, seq_indices*2, :] + self.pos_embed(seq_indices).unsqueeze(0)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.cfg.recycle_layer - 1:
                recycler_stream = self.recycler_block(x)
                #new_context = recycler_stream[:, -1, :]
                new_context = recycler_stream[:, -1, :] - x[:, -1, :]
                if not need_distn:
                    return new_context

        x = self.ln_f(x[:, -1, :])
        distn = self.unembed(x)
        return new_context, distn