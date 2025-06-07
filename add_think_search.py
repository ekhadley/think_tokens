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
import itertools
from eindex import eindex
from collections import deque

from normal import GPT2
from supervised_rollout_think import GPT2Thinking
from add_normal import SimpleTokenizer, makeAdditionDataset
from utils import *

def evalRollout(model: GPT2Thinking, rollout: t.Tensor, ans_tok: int) -> float: # concatenates an end_thought to the rollout and  evaluates the logprob of the answer token.
    with t.no_grad():
        rollout = t.cat([rollout, t.tensor([model.end_thought])])
        logits = model(rollout).squeeze()
        logprobs = t.log_softmax(logits[..., :model.cfg.d_normal_vocab], dim=-1)
        ans_logprob = logprobs[-1, ans_tok]
    return ans_logprob.item()

def expandThought(rollout: t.Tensor, model: GPT2Thinking, ans_tok: int) -> t.Tensor:
    with t.no_grad():
        rollouts = rollout.repeat(model.cfg.d_thought_vocab - 1, 1)
        rollouts = t.cat([rollouts, t.arange(model.cfg.d_normal_vocab, model.cfg.d_vocab_total - 1).unsqueeze(-1)], dim=1)
        #rollouts = t.cat([rollouts, t.tensor([model.end_thought]*rollout.shape[0]).unsqueeze(-1)], dim=1)
        logits = model(rollouts).squeeze()
        pred_logprobs = t.log_softmax(logits[:, -1, :model.cfg.d_normal_vocab], dim=-1)
        ans_logprobs = pred_logprobs[:, ans_tok]
    return rollouts, ans_logprobs

def greedyThoughtSearch(rollout: t.Tensor, model: GPT2Thinking, ans_tok: int, max_steps: int) -> t.Tensor:

    with t.no_grad():
        print()
        print(green, rollout, endc)
        for _ in range(max_steps):
            next_thought_scores = expandThought(rollout, model, ans_tok)
            best_thought_idx = next_thought_scores.argmax().item() + model.cfg.d_normal_vocab
            rollout = t.cat([rollout, t.tensor([best_thought_idx])], dim=0)

def allPossibleRollouts(low: int, high: int, size: int) -> t.Tensor:
    base = t.arange(low, high, dtype=t.int64)
    if size == 1:
        return base.view(1, -1)
    grids = t.meshgrid(*([base] * size), indexing='ij')  # list length `size`
    return t.stack(grids).reshape(size, -1).T

def bruteForceThoughtSearch(rollout: t.Tensor, model: GPT2Thinking, ans_tok: int, max_steps: int) -> t.Tensor:
    """
    Exhaustively searches all possible thinking token combinations up to max_steps.
    Returns the rollout with the highest logprob of the answer token.
    """
    # create a single 2d tensor containing all possible rollouts. This is every permutation of thinking tokens up to max_steps
    with t.no_grad():
        print()
        perms = allPossibleRollouts(model.cfg.d_normal_vocab, model.cfg.d_vocab_total, max_steps)
        rollouts = t.cat([rollout.repeat(perms.shape[0], 1), perms], dim=1)
        rollouts = t.cat([rollouts, t.tensor([model.end_thought] * rollouts.shape[0]).unsqueeze(-1)], dim=1)
        logits = model(rollouts).squeeze()
        pred_logprobs = t.log_softmax(logits[:, -1, :model.cfg.d_normal_vocab], dim=-1)
        ans_logprobs = pred_logprobs[:, ans_tok]
        print(purple, ans_logprobs, endc)
        print(orange, ans_logprobs.min().item(), ans_logprobs.max().item(), ans_logprobs.mean().item(), endc)
    return ans_logprobs

def train(model: GPT2Thinking, cfg: TrainingConfig, dataset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dataset)//cfg.batch_size)

    input_max = dataset.attrs["input_max"]
    wandb.init(project="add_thoughtful_think", name=f"think_nothink_{input_max}", config=cfg)
    wandb.watch(model, log="all")

    q_len = dataset.attrs["question_len"]

    epsilon = 1.0 # prob of choosing random think token

    for b in (tr:=tqdm.trange(len(dataset), ncols=200)):
        row = dataset.iloc[b]
        q_toks = t.tensor(row["question_toks"])
        ans_tok = row["answer_tok"]  # Single token, not tensor


        rollouts, pred_rewards = [], []
        with t.inference_mode(): # do inference without gradients to generate rollouts
            for g in range(cfg.group_size): # for each rollout in the group
                rollout = q_toks.clone()
                max_think_idx = model.cfg.seq_len - 1  # Reserve 1 position for answer prediction

                for i in range(q_len, max_think_idx): # for each think token in the rollout
                    if random.random() < epsilon:
                        think_tok = t.tensor([random.randint(model.cfg.d_normal_vocab, model.cfg.d_vocab_total - 1)])
                    else:
                        logits = model(rollout).squeeze()
                        sample_probs = t.softmax((logits[-1, model.cfg.d_normal_vocab:]), dim=-1) # get logpprob distn over thinking token
                        think_tok = t.multinomial(sample_probs, num_samples=1) + model.cfg.d_normal_vocab # sample a thinking token
                    if i == max_think_idx - 1 or random.random() < cfg.prob_force_end_thought: 
                        think_tok = t.tensor([model.end_thought])
                    rollout = t.cat([rollout, think_tok])
                    if think_tok.item() == model.end_thought: # end of thinking. get logprobs on answer (reward)
                        break

                logits = model(rollout).squeeze()
                logprobs = t.log_softmax(logits[-1, :model.cfg.d_normal_vocab], dim=-1) # get the logprobs of the answer tokens
                ans_logprob = logprobs[ans_tok]  # ans_tok is the single token ID
                reward = ans_logprob # reward is the logprob on correct answer token

                pred_rewards.append(reward.detach().item())
                rollouts.append(rollout)

            epsilon = max(epsilon * cfg.eps_decay, cfg.eps_min)
        
        pred_rewards = t.tensor(pred_rewards, requires_grad=False)
        pred_reward_mean = pred_rewards.mean().item() # mean logprob of correct answer token
        mc_pred_rewards = pred_rewards - pred_reward_mean # mean centered rewards
        normed_pred_rewards = mc_pred_rewards / (pred_rewards.std() + 1e-8)
        pred_reward_var = pred_rewards.var().item() # variance of the mean centered rewards
        pred_probs = t.exp(pred_rewards) # convert logprobs to probabilities
        pred_prob_var = pred_probs.var().item() # variance of the probabilities of the correct answer token

        mean_num_think = sum([len(r) for r in rollouts]) / cfg.group_size - q_len
        
        total_rewards, think_rewards, entropy_rewards = [], [], []
        for g in range(cfg.group_size): # we run the rollouts back through with gradients to get a differentiable reward
            rollout = rollouts[g].clone()
            logits = model(rollout).squeeze()
            logprobs = t.log_softmax(logits, dim=-1)
            rollout_len = rollout.shape[0]

            pred_reward = logprobs[-1, ans_tok] # logprob of correct answer token at last position (the end_thought token)

            think_indices = list(range(q_len - 1, rollout_len - 1)) # thinking tokens go from after question to end of rollout

            think_logprobs_sel = logprobs[think_indices, rollout[q_len:rollout_len]] # get the logprob values of the thinking tokens that were outputted
            weighted_think_logprobs = think_logprobs_sel * normed_pred_rewards[g] # weight the logprobs by the mean centered reward
            think_reward = weighted_think_logprobs.mean()
            think_rewards.append(think_reward.detach().item())

            think_logprobs_all = logprobs[think_indices, model.cfg.d_normal_vocab:] # shape: (num_think_steps, d_thought_vocab)
            entropy = -(think_logprobs_all.exp() * think_logprobs_all).sum(dim=-1) # shape: (num_think_steps,)
            entropy_reward = entropy.mean() # average over think steps
            entropy_rewards.append(entropy_reward.detach().item())

            total_reward = (1 - cfg.think_reward_weight)*pred_reward + cfg.think_reward_weight*think_reward + cfg.entropy_reward_weight*entropy_reward
            total_rewards.append(total_reward)

        think_reward_mean = sum(think_rewards) / len(think_rewards)
        entropy_reward_mean = sum(entropy_rewards) / len(entropy_rewards)

        total_reward = sum(total_rewards) / len(total_rewards)
        total_reward.backward()

        if b != 0 and b % cfg.batch_size == 0:
            opt.step()
            scheduler.step()
            opt.zero_grad()

            wandb.log({
                "pred_reward": pred_reward_mean,
                "think_reward": think_reward_mean,
                "total_reward": total_reward.item(),
                "num_think": mean_num_think,
                "entropy_reward": entropy_reward_mean,
                "pred_reward_var": pred_reward_var,
                "pred_prob_var": pred_prob_var,
                "prob_force_end_thought": cfg.prob_force_end_thought,
                "epsilon": epsilon,
                "think_logprobs": think_logprobs_all,
            })
            #printSeq(rollouts[0], simple_tokenizer, model.cfg)
            tr.set_description(f"{magenta}pred reward mean: {pred_reward_mean:.3f}, total reward: {total_reward.item():.3f}, think reward: {think_reward_mean:.3f}, entropy: {entropy_reward_mean:.3f}, epsilon: {epsilon:.3f}, num_think: {mean_num_think:.3f}")

        if b % 1000 == 0:
            #t.save(model.state_dict(), f"saves/add_think2_{b}.pt")
            #greedyThoughtSearch(rollout, model, ans_tok, max_steps=16) # greedy search to expand the rollout with thinking tokens
            #dfsThoughtSearch(rollout, model, ans_tok, max_steps=16) # greedy search to expand the rollout with thinking tokens
            bruteForceThoughtSearch(rollout, model, ans_tok, max_steps=4) # greedy search to expand the rollout with thinking tokens

INPUT_MAX = 100
NUM_EXAMPLES = 1_000_000

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))

    model_cfg = ThinkingModelConfig(d_model=32, seq_len=32, d_mlp=128, d_head=16, n_heads=4, n_layers=2, d_normal_vocab=INPUT_MAX, d_thought_vocab=16)
    training_cfg = TrainingConfig(
        group_size=16,
        think_reward_weight=0.5,
        entropy_reward_weight=0.0,
        prob_force_end_thought=0.05,
        eps_decay=0.999995,
        eps_min=0.05,
        batch_size=16,
        lr=1e-3,
        weight_decay=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.95
    )
    model = GPT2Thinking(model_cfg)

    simple_tokenizer = SimpleTokenizer(max_int=INPUT_MAX)
    trainset, testset = makeAdditionDataset(simple_tokenizer, INPUT_MAX, NUM_EXAMPLES, train_split=0.99)

    train(model, training_cfg, trainset)
    #benchmark_addition_think(model, testset)
