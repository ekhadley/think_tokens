import tqdm
import datasets
import wandb
import torch as t
from torch import nn
from transformers import GPT2TokenizerFast, AutoTokenizer
import random
import pandas as pd
from eindex import eindex

from normal import TransformerBlock
from utils import *
from add_normal import makeAdditionDataset




class GPT2Thinking(nn.Module):
    def __init__(self, cfg: ThinkingModelConfig):
        super(GPT2Thinking, self).__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.embed = nn.Embedding(cfg.d_vocab_total, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab_total, bias=False)
        self.eot = 50256
        self.end_thought = cfg.d_vocab_total - 2
        self.tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")

    def encode(self, text):
        return t.tensor(self.tokenizer(text).input_ids)
    def decode(self, tokens):
        return self.tokenizer.batch_decode(tokens)
    def forward(self, x: t.Tensor) -> t.Tensor:
        if x.ndim == 1: x = x.unsqueeze(0)
        x = self.embed(x) + self.pos_embed(t.arange(x.shape[1], device=x.device)).unsqueeze(0)
        for i, block in enumerate(self.blocks):
            x = block(x)
        x = self.ln_f(x)
        x = self.unembed(x)
        return x


t.backends.cuda.enable_flash_sdp(enabled=True)
t.set_default_device(t.device("cuda"))

def train(model: GPT2Thinking, cfg: TrainingConfig, dataset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)







if __name__ == "__main__":
    model_cfg = ThinkingModelConfig(d_model=64, seq_len=128, d_mlp=256, d_head=16, n_heads=4, n_layers=4, d_normal_vocab=50_257, d_thought_vocab=2048)
    model = GPT2Thinking(model_cfg)
    training_cfg = TrainingConfig(gamma=0.95, batch_size=16, lr=3e-4, weight_decay=1e-3, adam_beta1=0.9, adam_beta2=0.95)

    dataset = makeAdditionDataset(model.tokenizer, int_max=10_000)
    #dataset = pd.read_pickle("datasets/additions.pkl")

    print(dataset)
    print(dataset.iloc[0])