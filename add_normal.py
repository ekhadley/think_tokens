import tqdm
import wandb
import torch as t
from torch import nn
import pandas as pd
import numpy as np

from models import GPT2, ModelConfig, TrainingConfig
from utils import *


def benchmark_addition(model: GPT2, dataset: pd.DataFrame, max_answer_len: int = 10):
    """
    Benchmarks the model's addition ability on a dataset.
    Returns:
        - mean_logprob: mean logprob over correct answer tokens
        - accuracy: fraction of exact matches using argmax sampling
    """
    model.eval()
    total_logprob = 0.0
    total_tokens = 0
    correct = 0
    n = len(dataset)
    q_len = dataset.attrs['question_len']
    for i, row in tqdm.tqdm(enumerate(dataset.itertuples()), total=n, desc="Benchmark", ncols=100):
        q_toks = t.tensor(row.question_toks)
        ans_tok = row.answer_tok  # Single token
        
        # Create full sequence by concatenating question and answer
        full_seq = t.cat([q_toks, t.tensor([ans_tok], device=q_toks.device)])
        
        # Get logits for the full sequence
        with t.no_grad():
            logits = model(full_seq).squeeze(0)  # [seq_len, vocab]
            logprobs = t.log_softmax(logits, dim=-1)
        
        # Get logprob for the single answer token
        ans_logprob = logprobs[q_len - 1, ans_tok]  # logprob of answer token conditioned on question
        total_logprob += ans_logprob.item()
        total_tokens += 1
        
        # Argmax sampling - predict single answer token
        with t.no_grad():
            logits = model(q_toks).squeeze(0)  # [q_len, vocab]
        next_token = logits[-1].argmax().item()
        if next_token == ans_tok:
            correct += 1
            
    mean_logprob = total_logprob / total_tokens if total_tokens > 0 else float('nan')
    accuracy = correct / n if n > 0 else float('nan')
    print(f"Mean logprob: {mean_logprob:.4f}, Accuracy: {accuracy:.4f}")
    return mean_logprob, accuracy

def train(model: GPT2, cfg: TrainingConfig, dataset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dataset)//cfg.batch_size)

    input_max = dataset.attrs["input_max"]
    wandb.init(project="add_thoughtful", name=f"normal_{input_max}", config=cfg)
    wandb.watch(model, log="all")
    wandb.config.update(model.cfg.to_dict())
    wandb.config.update(cfg.to_dict())

    for b in (tr:=tqdm.trange(0, len(dataset), cfg.batch_size, ncols=100)):
        q_toks = t.tensor(np.stack(dataset.iloc[b:b+cfg.batch_size]['question_toks']))
        ans_toks = t.tensor(dataset.iloc[b:b+cfg.batch_size]['answer_tok'].to_numpy())
        
        batch_indices = t.arange(len(ans_toks), requires_grad=False)

        logits = model.forward(q_toks).squeeze()
        logprobs = t.log_softmax(logits, dim=-1)

        pred_logprobs = logprobs[batch_indices, -1, ans_toks]
        loss = -pred_logprobs.mean()
        loss.backward()
        opt.step()
        scheduler.step()
        opt.zero_grad()

        if b != 0 and b % 10_000 == 0:
            wandb.log({"loss": loss.detach().item()})
            tr.set_description(f"{magenta}loss: {loss.detach().item():.3f}")
            t.save(model.state_dict(), f"saves/add_normal{b}.pth")

INPUT_MAX = 100
NUM_EXAMPLES = 1_000_000

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    
    model_cfg = ModelConfig(d_model=32, seq_len=32, d_mlp=128, d_head=16, n_heads=4, n_layers=2, d_vocab=INPUT_MAX)
    training_cfg = TrainingConfig(batch_size=16, lr=1e-3, weight_decay=1e-3, adam_beta1=0.9, adam_beta2=0.95)
    model = GPT2(model_cfg)

    simple_tokenizer = SimpleTokenizer(max_int=INPUT_MAX)
    trainset, testset = makeAdditionDataset(simple_tokenizer, INPUT_MAX, NUM_EXAMPLES, train_split=0.99)

    train(model, training_cfg, trainset)
    benchmark_addition(model, testset)