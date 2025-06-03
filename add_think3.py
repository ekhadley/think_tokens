import math
import tqdm
import datasets
import wandb
import torch as t
from torch import nn
from transformers import GPT2TokenizerFast, AutoTokenizer
import random
import pandas as pd
import numpy as np
from eindex import eindex

from normal import GPT2
from supervised_rollout_think import GPT2Thinking
from add_normal2 import SimpleTokenizer, makeAdditionDataset
from utils import *

def train(model: GPT2Thinking, cfg: TrainingConfig, dataset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)

    input_max = dataset.attrs["input_max"]
    wandb.init(project="add_thoughtful_think", name=f"think_nothink_{input_max}", config=cfg)
    wandb.watch(model, log="all")

    for b in (tr:=tqdm.trange(len(dataset), ncols=200)):
        row = dataset.iloc[b]
        q_toks = t.tensor(row["question_toks"])
        ans_tok = row["answer_tok"]  # Single token, not tensor
        q_len = row["question_len"]

        # concatenate end thgought token to the question:
        q_toks = t.cat([q_toks, t.tensor([model.end_thought])])  # End thought token
        logits = model.forward(q_toks).squeeze()
        logprobs = t.log_softmax(logits[-1], dim=-1)
        loss = logprobs[ans_tok]
        loss.backward()

        if b != 0 and b % cfg.batch_size == 0:
            opt.step()
            opt.zero_grad()

            wandb.log({
                "pred_reward": loss,
            })
            #printSeq(rollouts[0], simple_tokenizer, model.cfg)
            tr.set_description(f"{magenta}loss: {loss:.3f}")

INPUT_MAX = 100
NUM_EXAMPLES = 1_000_000

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))

    model_cfg = ThinkingModelConfig(d_model=32, seq_len=32, d_mlp=128, d_head=16, n_heads=4, n_layers=2, d_normal_vocab=INPUT_MAX + 2, d_thought_vocab=100)
    training_cfg = TrainingConfig(batch_size=16, lr=3e-3, weight_decay=1e-6, adam_beta1=0.9, adam_beta2=0.95)
    model = GPT2Thinking(model_cfg)

    simple_tokenizer = SimpleTokenizer(max_int=INPUT_MAX)
    trainset, testset = makeAdditionDataset(simple_tokenizer, INPUT_MAX, NUM_EXAMPLES, train_split=0.99)

    train(model, training_cfg, trainset)