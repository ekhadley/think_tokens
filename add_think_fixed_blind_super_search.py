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

from supervised_rollout_think import GPT2Thinking
from add_normal import SimpleTokenizer, makeAdditionDataset
from utils import *

from add_think_search import allPossibleRollouts
from add_think_fixed_blind import benchmark_addition_think_fixed_blind

def bruteForceThoughtSearch(model: GPT2Thinking, ans_tok: int, max_steps: int) -> t.Tensor:
    with t.no_grad():
        print()
        perms = allPossibleRollouts(model.cfg.d_normal_vocab, model.cfg.d_vocab_total - 1, max_steps)
        rollouts = t.cat([perms, t.tensor([model.end_thought] * perms.shape[0]).unsqueeze(-1)], dim=1)
        logits = model(rollouts).squeeze()
        pred_logprobs = t.log_softmax(logits[:, -1, :model.cfg.d_normal_vocab], dim=-1)
        ans_logprobs = pred_logprobs[:, ans_tok]
        print(purple, ans_logprobs, endc)
        print(orange, ans_logprobs.min().item(), ans_logprobs.max().item(), ans_logprobs.mean().item(), endc)

    return ans_logprobs

def train(model: GPT2Thinking, cfg: TrainingConfig, dataset: pd.DataFrame, testset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dataset)//cfg.batch_size)

    input_max = dataset.attrs["input_max"]
    wandb.init(project="add_thoughtful_think", name=f"think_fixed_blind_super_search_clean{input_max}", config=cfg)
    wandb.watch(model, log="all")
    wandb.config.update(model.cfg.to_dict())
    wandb.config.update(cfg.to_dict())

    q_len = dataset.attrs["question_len"]

    perms = allPossibleRollouts(model.cfg.d_normal_vocab, model.cfg.d_vocab_total - 1, cfg.think_len)
    group_size = perms.shape[0]
    end_thoughts = t.tensor([model.end_thought] * group_size, requires_grad=False).unsqueeze(-1) # end_thought token for each group
    perms = t.cat([perms, end_thoughts], dim=1)  # add end_thought token to each permutation

    group_indices = t.arange(group_size, requires_grad=False).unsqueeze(-1)
    think_indices = t.arange(q_len, q_len + cfg.think_len, requires_grad=False)

    ndig = int(math.log10(input_max))

    for b in (tr:=tqdm.trange(len(dataset), ncols=200)):
        row = dataset.iloc[b]
        q_toks = t.tensor(row["question_toks"])
        ans_tok = row["answer_tok"]  # Single token, not tensor

        ans_str = [model.cfg.d_normal_vocab + int(c) for c in str(row["answer"])] # manually creating the 'correct' chain of thought tokens
        while len(ans_str) < ndig: ans_str.insert(0, model.cfg.d_normal_vocab)
        ans_str.append(model.end_thought)
        correct_thoughts = t.tensor(ans_str)

        with t.inference_mode(): # do inference without gradients to generate rollouts
            rollouts = t.cat([q_toks.unsqueeze(0).repeat(group_size, 1), perms], dim=1)

            rollouts_no_question = rollouts[:, q_len:]
            logits = model(rollouts_no_question).squeeze()
            logprobs = t.log_softmax(logits[:, -1, :model.cfg.d_normal_vocab], dim=-1)
            #pred_rewards = logprobs[:, ans_tok]
            #pred_reward_mean = pred_rewards.mean().item() # mean of the predicted rewards
            #normed_pred_rewards = (pred_rewards - pred_reward_mean) / (pred_rewards.std() + 1e-8) # normalize the rewards
            pred_rewards = (rollouts[:, q_len:q_len + cfg.think_len] == correct_thoughts[:cfg.think_len]).float().sum(dim=-1) * 50
            pred_reward_mean = pred_rewards.mean().item()
            normed_pred_rewards = pred_rewards

        rollouts = rollouts.clone()
        normed_pred_rewards = normed_pred_rewards.clone()
        logits = model(rollouts).squeeze()

        pred_logits = model(correct_thoughts).squeeze()
        pred_logprobs = t.log_softmax(pred_logits[-1, :model.cfg.d_normal_vocab], dim=-1) # real token logprob distn on the end_thought token
        pred_reward = pred_logprobs[ans_tok]
        pred_reward_mean = pred_reward

        think_logprobs = t.log_softmax(logits[group_indices, (think_indices - 1).unsqueeze(0), model.cfg.d_normal_vocab:-1], dim=-1) # logprob distns for each thinking token position
        action_logprobs = think_logprobs[group_indices, think_indices - q_len, rollouts[:, think_indices] - model.cfg.d_normal_vocab] # logprob of the thinking tokens that were outputted
        weighted_think_logprobs = action_logprobs * normed_pred_rewards.unsqueeze(-1) # logprobs times rewards
        think_reward = weighted_think_logprobs.mean(dim=0) # mean over the group size
        think_reward_mean = think_reward.mean() # mean of the think rewards

        entropy = -(think_logprobs * t.exp(think_logprobs)).sum(dim=-1).mean()

        total_reward = (1 - cfg.think_reward_weight) * pred_reward_mean + cfg.think_reward_weight * think_reward_mean + cfg.entropy_reward_weight * entropy
        total_reward.backward()

        if b != 0 and b % cfg.batch_size == 0:
            opt.step()
            scheduler.step()
            opt.zero_grad()

            pred_reward_var = pred_rewards.var().item() # variance of the predicted rewards for logging
            pred_prob_var = t.exp(pred_rewards).var().item() # answer prob variance for logging
            rollout_mean_logprob = action_logprobs.mean(dim=-1)

            if b%1024 == 0:
                correct_rollout_idx = normed_pred_rewards.argmax().item()
                print()
                print(orange, ans_tok, correct_thoughts, endc)
                print(red, think_logprobs[correct_rollout_idx].T, endc)
                policy_first_thought = think_logprobs[0, 0].argmax().item()
                policy_second_thought = think_logprobs[policy_first_thought, 1].argmax().item()
                policy_third_thought = think_logprobs[policy_first_thought, 2].argmax().item()
                guess = policy_first_thought * 100 + policy_second_thought * 10 + policy_third_thought
                print(f"{blue}policy guess: {guess}{endc}")
                print(green, action_logprobs[correct_rollout_idx].T, endc)

            wandb.log({
                "pred_reward": pred_reward_mean,
                "think_reward": think_reward_mean,
                "total_reward": total_reward,
                "num_think": cfg.think_len,
                "pred_reward_var": pred_reward_var,
                "pred_prob_var": pred_prob_var,
                "prob_force_end_thought": 0.0,
                "epsilon": 0,
                "think_logprobs": think_logprobs[0],
                "entropy_reward": entropy,
            })
            #printSeq(rollouts[0], simple_tokenizer, model.cfg)
            tr.set_description(f"{magenta}pred reward mean: {pred_reward_mean:.3f}, total reward: {total_reward.item():.3f}, think reward: {think_reward_mean:.3f}")

        if b != 0 and b % 32_000 == 0:
            print()
            print(red, correct_thoughts, endc)
            for row in range(rollouts.shape[0]):
                print(f"{blue}{rollouts[row].tolist()} {magenta}{rollout_mean_logprob[row].item():.3f} : {cyan}{pred_rewards[row].item():.3f} {green}({normed_pred_rewards[row].item():.3f}){endc}")
            #bruteForceThoughtSearch(model, ans_tok, cfg.think_len)
            _, benchmark_accuracy = benchmark_addition_think_fixed_blind(model, testset, cfg.think_len)
            wandb.log({"benchmark_accuracy": benchmark_accuracy})

INPUT_MAX = 1_000
NUM_EXAMPLES = 1_000_000

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.set_printoptions(sci_mode=False)

    model_cfg = ThinkingModelConfig(d_model=32, seq_len=32, d_mlp=128, d_head=16, n_heads=4, n_layers=2, d_normal_vocab=INPUT_MAX, d_thought_vocab=11)
    training_cfg = TrainingConfig(
        think_len=3,
        think_reward_weight=0.5,
        entropy_reward_weight=0.03,
        batch_size=16,
        lr=1e-3,
        weight_decay=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.95
    )
    model = GPT2Thinking(model_cfg)

    simple_tokenizer = SimpleTokenizer(max_int=INPUT_MAX)
    trainset, testset = makeAdditionDataset(simple_tokenizer, INPUT_MAX, NUM_EXAMPLES, train_split=0.99)

    train(model, training_cfg, trainset, testset)
    benchmark_addition_think_fixed_blind(model, testset, training_cfg.think_len)
