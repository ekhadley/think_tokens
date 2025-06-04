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
from add_normal import SimpleTokenizer, makeAdditionDataset
from utils import *

def evalRollout(model: GPT2Thinking, rollout: t.Tensor, ans_tok: int) -> float: # concatenates an end_thought to the rollout and  evaluates the logprob of the answer token.
    with t.no_grad():
        rollout = t.cat([rollout, t.tensor([model.end_thought], device=rollout.device)])
        logits = model(rollout).squeeze()
        logprobs = t.log_softmax(logits[..., :model.cfg.d_normal_vocab], dim=-1)
        ans_logprob = logprobs[-1, ans_tok]
    return ans_logprob.item()

# creates a batch where each continueation concatenates 1 of all the possible thinking tokens
# returns a tensor of correct answer logprobs for each continuation
def expandSteps(rollout: t.Tensor, model: GPT2Thinking, ans_tok: int) -> t.Tensor:
    with t.no_grad():
        rollout = rollout.repeat(model.cfg.d_thought_vocab, 1)
        rollout = t.cat([rollout, t.arange(model.cfg.d_normal_vocab, model.cfg.d_vocab_total - 1).unsqueeze(0)], dim=1)
        print(rollout)


def train(model: GPT2Thinking, cfg: TrainingConfig, dataset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dataset)//cfg.batch_size)

    input_max = dataset.attrs["input_max"]
    wandb.init(project="add_thoughtful_think", name=f"think_fixed_{input_max}", config=cfg)
    wandb.watch(model, log="all")

    q_len = dataset.attrs["question_len"]

    think_len = 8

    group_size = 32
    think_reward_weight = 0.5

    epsilon = 1 # prob of choosing random think token
    eps_decay = 0.999995
    eps_min = 0.05

    end_thoughts = t.tensor([model.end_thought] * group_size, requires_grad=False).unsqueeze(-1)  # end_thought token for each group
    group_indices = t.arange(group_size, requires_grad=False).unsqueeze(-1)
    think_indices = t.arange(q_len, q_len + think_len)

    for b in (tr:=tqdm.trange(len(dataset), ncols=200)):
        row = dataset.iloc[b]
        q_toks = t.tensor(row["question_toks"])
        ans_tok = row["answer_tok"]  # Single token, not tensor

        with t.inference_mode(): # do inference without gradients to generate rollouts
            rollouts = q_toks.unsqueeze(0).repeat(group_size, 1)
            for i in range(think_len): # for each think token in the rollout. end_thought counts as a thought.
                if random.random() < epsilon:
                    think_toks = t.randint(model.cfg.d_normal_vocab, model.cfg.d_vocab_total - 1, (group_size, 1))
                else:
                    logits = model(rollouts).squeeze()
                    sample_probs = t.softmax((logits[:, -1, model.cfg.d_normal_vocab:]), dim=-1) # get logpprob distn over thinking token
                    think_toks = t.multinomial(sample_probs, num_samples=1) + model.cfg.d_normal_vocab # sample a thinking token

                rollouts = t.cat([rollouts, think_toks], dim=1)

            rollouts = t.cat([rollouts, end_thoughts], dim=1)
            logits = model(rollouts).squeeze()
            logprobs = t.log_softmax(logits[:, -1, :model.cfg.d_normal_vocab], dim=-1) # get the logprobs of the answer tokens
            pred_rewards = logprobs[:, ans_tok]  # ans_tok is the single token ID

            pred_reward_mean = pred_rewards.mean().item() # mean logprob of correct answer token
            mc_pred_rewards = pred_rewards - pred_reward_mean # mean centered rewards
            normed_pred_rewards = (mc_pred_rewards / (pred_rewards.std() + 1e-8))
            pred_reward_var = pred_rewards.var().item() # variance of the mean centered rewards
            pred_probs = t.exp(pred_rewards) # convert logprobs to probabilities
            pred_prob_var = pred_probs.var().item() # variance of the probabilities of the correct answer token
            
            epsilon = max(epsilon * eps_decay, eps_min)
        
        rollouts = rollouts.clone()
        normed_pred_rewards = normed_pred_rewards.clone()
        logits = model(rollouts).squeeze()

        pred_logprobs = t.log_softmax(logits[:, -1, :model.cfg.d_normal_vocab], dim=-1) # real token logprob distn on the end_thought token
        pred_rewards = pred_logprobs[:, ans_tok]
        pred_reward_mean = pred_rewards.mean()

        think_logprobs = t.log_softmax(logits[group_indices, (think_indices - 1).unsqueeze(0), model.cfg.d_normal_vocab:], dim=-1) # logprob distns for each thinking token position
        action_logprobs = think_logprobs[group_indices, think_indices - q_len, rollouts[:, think_indices] - model.cfg.d_normal_vocab] # logprob of the thinking tokens that were outputted
        weighted_think_logprobs = action_logprobs * normed_pred_rewards.unsqueeze(-1) # logprobs times rewards
        think_reward = weighted_think_logprobs.mean(dim=0) # mean over the group size
        think_reward_mean = think_reward.mean() # mean of the think rewards

        total_reward = (1 - think_reward_weight) * pred_reward_mean + think_reward_weight * think_reward_mean
        total_reward.backward()

        if b != 0 and b % cfg.batch_size == 0:
            opt.step()
            scheduler.step()
            opt.zero_grad()

            wandb.log({
                "pred_reward": pred_reward_mean,
                "think_reward": think_reward_mean,
                "total_reward": total_reward,
                "num_think": think_len,
                "pred_reward_var": pred_reward_var,
                "pred_prob_var": pred_prob_var,
                "prob_force_end_thought": 0.0,
                "epsilon": epsilon,
                "think_logprobs": action_logprobs,
            })
            #printSeq(rollouts[0], simple_tokenizer, model.cfg)
            tr.set_description(f"{magenta}pred reward mean: {pred_reward_mean:.3f}, total reward: {total_reward.item():.3f}, think reward: {think_reward_mean:.3f}, epsilon: {epsilon:.3f}")

        if b != 0 and b % 1000 == 0:
            t.save(model.state_dict(), f"saves/add_think2_{b}.pt")

INPUT_MAX = 100
NUM_EXAMPLES = 1_000_000

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))

    model_cfg = ThinkingModelConfig(d_model=32, seq_len=32, d_mlp=128, d_head=16, n_heads=4, n_layers=2, d_normal_vocab=INPUT_MAX + 2, d_thought_vocab=100)
    training_cfg = TrainingConfig(batch_size=16, lr=3e-4, weight_decay=1e-6, adam_beta1=0.9, adam_beta2=0.95)
    model = GPT2Thinking(model_cfg)

    simple_tokenizer = SimpleTokenizer(max_int=INPUT_MAX)
    trainset, testset = makeAdditionDataset(simple_tokenizer, INPUT_MAX, NUM_EXAMPLES, train_split=0.99)

    train(model, training_cfg, trainset)
    #benchmark_addition_think_fixed(model, testset)
