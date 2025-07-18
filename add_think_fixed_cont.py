import math
import tqdm
import wandb
import random
import pandas as pd
import torch as t

from models import TrainingConfig, ContThinkingModel, ContThinkingModelConfig
from utils import *

def benchmark_addition_think_fixed_blind_split(answer_model: ContThinkingModel, think_model: ContThinkingModel, dataset: pd.DataFrame, think_len: int, cat_end_thought: bool = False):
    answer_model.eval()
    think_model.eval()
    q_len = dataset.attrs["question_len"]
    d_normal_vocab = dataset.attrs["input_max"]
    n_examples = dataset.attrs["n_examples"]
    end_thought_token = think_model.cfg.d_thought_vocab - 1

    with t.no_grad():
        q_toks = t.tensor(np.stack(dataset['question_toks']))
        ans_toks = t.tensor(dataset['answer_tok'].to_numpy())
        if cat_end_thought: end_thoughts = t.tensor([end_thought_token] * len(dataset), dtype=t.int64)

        rollouts = q_toks.clone()
        for i in range(think_len):
            think_logits = think_model(rollouts)
            think_toks = think_logits[:, -1].argmax(dim=-1) + d_normal_vocab
            rollouts = t.cat([rollouts, think_toks.unsqueeze(-1)], dim=1)
        rollout_no_question = rollouts[:, q_len:] - d_normal_vocab
        if cat_end_thought: rollout_no_question = t.cat([rollout_no_question, end_thoughts], dim=0)
        ans_logits = answer_model(rollout_no_question)
        logprobs = t.log_softmax(ans_logits[:, -1], dim=-1)
        ans_logprobs = logprobs[t.arange(n_examples), ans_toks]
        ans_guesses = logprobs.argmax(dim=-1)
        mean_logprob = ans_logprobs.mean().item()
        accuracy = (ans_guesses == ans_toks).float().mean().item()
    return mean_logprob, accuracy

# this version is parallelized over the batch and the group. so keep an eye out.
def train(model: ContThinkingModel, cfg: TrainingConfig, dataset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    model.train()

    input_max = dataset.attrs["input_max"]
    q_len = dataset.attrs["question_len"]

    wandb.init(project="add_thoughtful_think", name=f"think_fixed_cont_{input_max}x{q_len}", config=cfg)
    wandb.config.update(cfg.to_dict())

    epsilon = 1.0 # prob of choosing random think token

    d_model = model.cfg.d_model
    batch_indices = t.arange(cfg.batch_size, requires_grad=False)

    benchmark_accuracy = 0.0

    n_batches = len(dataset) // cfg.batch_size
    for b in (tr:=tqdm.trange(n_batches, ncols=150)):
        b_i = b * cfg.batch_size
        q_toks = t.tensor(np.stack(dataset.iloc[b_i:b_i+cfg.batch_size]['question_toks']))
        ans_toks = t.tensor(dataset.iloc[b_i:b_i+cfg.batch_size]['answer_tok'].to_numpy())
        print(red, q_toks, cyan, ans_toks, endc)

        ctx = t.zeros((cfg.batch_size, q_len + cfg.think_len, d_model))
        ctx[:, :q_len] = model.make_context(q_toks).squeeze()
        print(green, ctx.shape, endc)
        for i_t in range(cfg.think_len): # for each think token in the rollout.
            thought_vec = model.generate_thought(ctx)
            print(lime, i_t, blue, ctx.shape, purple, thought_vec.shape, endc)
            ctx[:, q_len + i_t, :] = thought_vec

        logits = model.generate_logits(ctx).squeeze()
        print(yellow, logits.shape, endc)
        logprobs = t.log_softmax(logits, dim=-1)
        loss = logprobs[batch_indices, ans_toks]
        loss.backward()
        opt.step()

        exit()
        
        wandb.log({"pred_reward": loss})
        tr.set_description(f"{magenta}pred loss: {loss.detach().item():.3f} bench acc: {benchmark_accuracy:.4f}")
        if b*cfg.batch_size % 32_000 == 0:
            _, benchmark_accuracy = benchmark_addition_think_fixed_cont(model, testset, cfg.think_len)

            wandb.log({"benchmark_accuracy": benchmark_accuracy})
            #t.save(model.state_dict(), f"saves/add_think_fixed_cont{b}.pth")


INPUT_MAX = 100
NUM_EXAMPLES = 10_000_000
NUM_ADDS = 2

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    think_len = 2
    model_cfg = ContThinkingModelConfig(d_model=32, seq_len=8, d_mlp=128, d_head=16, n_heads=4, n_layers=4, n_think_layers=3, d_vocab=INPUT_MAX)
    training_cfg = TrainingConfig(
        think_len=think_len,
        lr=1e-3,
        batch_size=8,
        weight_decay=1e-9,
    )
    model = ContThinkingModel(model_cfg)

    trainset, testset = makeMultiAdditionDataset(INPUT_MAX, NUM_ADDS, NUM_EXAMPLES, train_split=0.99)

    train(model, training_cfg, trainset)
    benchmark_addition_think_fixed_blind_split(model, testset, training_cfg.think_len)
