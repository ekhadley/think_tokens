import tqdm
import wandb
import torch as t
from torch import nn
import pandas as pd
import numpy as np

from models import GPT2, ModelConfig, TrainingConfig
from utils import *

@t.inference_mode()
def benchmark_addition_normal(model: GPT2, dataset: pd.DataFrame):
    model.eval()
    q_toks = t.tensor(np.stack(dataset['question_toks']))
    ans_toks = t.tensor(dataset['answer_tok'].to_numpy())

    logits = model(q_toks).squeeze(0)  # [seq_len, vocab]
    logprobs = t.log_softmax(logits[:, -1], dim=-1)
    
    pred_logprobs = logprobs[t.arange(ans_toks.shape[0]), ans_toks]
    mean_logprob = pred_logprobs.mean().item()
    pred_guesses = logprobs.argmax(dim=-1)
    accuracy = (pred_guesses == ans_toks).float().mean().item()
    return mean_logprob, accuracy

def train(model: GPT2, cfg: TrainingConfig, dataset: pd.DataFrame, testset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    sched = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dataset)//cfg.batch_size, eta_min=1e-6)
    model.train()

    batch_indices = t.arange(cfg.batch_size, requires_grad=False)

    acc = 0.0

    for b in (tr:=tqdm.trange(0, len(dataset) - len(dataset)%cfg.batch_size, cfg.batch_size, ncols=100)):
        q_toks = t.tensor(np.stack(dataset.iloc[b:b+cfg.batch_size]['question_toks']))
        ans_toks = t.tensor(dataset.iloc[b:b+cfg.batch_size]['answer_tok'].to_numpy())

        logits = model.forward(q_toks).squeeze()
        logprobs = t.log_softmax(logits, dim=-1)

        pred_logprobs = logprobs[batch_indices, -1, ans_toks]
        loss = -pred_logprobs.mean()
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()

        wandb.log({"loss": loss.detach().item()}, step=b)
        tr.set_description(f"{magenta}loss: {loss.detach().item():.3f}, test acc: {acc:.4f}")

        if b*cfg.batch_size % 64_000 == 0:
            _, acc = benchmark_addition_normal(model, testset)
            wandb.log({"test_acc": acc}, step=b)
            #t.save(model.state_dict(), f"saves/multiadd_normal{b}.pth")

def run_training():
    wandb.init()
    w_config = wandb.config

    t.set_default_device(t.device("cuda"))

    model_cfg = ModelConfig(d_model=32, seq_len=NUM_ADDS, d_mlp=256, d_head=16, n_heads=4, n_layers=2, d_vocab=INPUT_MAX) # narrow, short
    
    training_cfg = TrainingConfig(
        batch_size=w_config.batch_size,
        lr=w_config.lr,
        weight_decay=w_config.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95
    )
    model = GPT2(model_cfg)

    wandb.config.update(model_cfg.to_dict())
    wandb.watch(model, log="all")

    simple_tokenizer = SimpleTokenizer(max_int=INPUT_MAX)
    trainset, testset = makeMultiAdditionDataset(INPUT_MAX, NUM_ADDS, NUM_EXAMPLES, train_split=0.999)

    train(model, training_cfg, trainset, testset)

INPUT_MAX = 10
NUM_EXAMPLES = 10_000_000
NUM_ADDS = 5

if __name__ == "__main__":
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'test_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'batch_size': {
                'values': [16, 32, 64, 128, 256]
            },
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-10,
                'max': 1e-3
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="multiadd-sweep")
    wandb.agent(sweep_id, function=run_training, count=100)