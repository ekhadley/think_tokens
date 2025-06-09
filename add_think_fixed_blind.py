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

def benchmark_addition_think_fixed_blind(model: GPT2Thinking, dataset: pd.DataFrame, think_len: int):
    model.eval()
    total_logprob = 0.0
    total_tokens = 0
    correct = 0
    q_len = dataset.attrs["question_len"]
    for i, row in tqdm.tqdm(enumerate(dataset.itertuples()), total=len(dataset), desc="BenchmarkThinkFixed", ncols=100):
        q_toks = t.tensor(row.question_toks)
        ans_tok = row.answer_tok  # Single token

        rollout = q_toks.clone()
        with t.no_grad():
            for i in range(think_len):
                logits = model(rollout).squeeze()
                think_tok = logits[-1, model.cfg.d_normal_vocab:-1].argmax() + model.cfg.d_normal_vocab
                rollout = t.cat([rollout, think_tok.unsqueeze(0)])
            
            rollout_no_question = rollout[q_len:]
            rollout_no_question = t.cat([rollout_no_question, t.tensor([model.end_thought], device=rollout.device)])
            logits = model(rollout_no_question).squeeze()
            logprobs = t.log_softmax(logits[-1, :model.cfg.d_normal_vocab], dim=-1)
            ans_logprob = logprobs[ans_tok]
            total_logprob += ans_logprob.item()
            total_tokens += 1
            
            answer_logit = logits[-1, :model.cfg.d_normal_vocab]
            generated = answer_logit.argmax().item()
            if generated == ans_tok:
                correct += 1
    mean_logprob = total_logprob / total_tokens if total_tokens > 0 else float('nan')
    accuracy = correct / len(dataset)
    print(yellow, f"[ThinkFixed] Mean logprob: {mean_logprob:.4f}, Accuracy: {accuracy:.4f}", endc)
    return mean_logprob, accuracy

def train(model: GPT2Thinking, cfg: TrainingConfig, dataset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dataset)//cfg.batch_size)

    input_max = dataset.attrs["input_max"]
    wandb.init(project="add_thoughtful_think", name=f"think_fixed_blind{input_max}", config=cfg)
    wandb.watch(model, log="all")
    wandb.config.update(model.cfg.to_dict())
    wandb.config.update(cfg.to_dict())

    q_len = dataset.attrs["question_len"]

    epsilon = .25 # prob of choosing random think token

    end_thoughts = t.tensor([model.end_thought] * cfg.group_size, requires_grad=False).unsqueeze(-1)  # end_thought token for each group
    group_indices = t.arange(cfg.group_size, requires_grad=False).unsqueeze(-1)
    think_indices = t.arange(q_len, q_len + cfg.think_len, requires_grad=False)

    for b in (tr:=tqdm.trange(len(dataset), ncols=200)):
        row = dataset.iloc[b]
        q_toks = t.tensor(row["question_toks"])
        ans_tok = row["answer_tok"]  # Single token, not tensor

        with t.inference_mode(): # do inference without gradients to generate rollouts
            rollouts = q_toks.unsqueeze(0).repeat(cfg.group_size, 1)
            for i in range(cfg.think_len): # for each think token in the rollout. end_thought counts as a thought.
                if random.random() < epsilon:
                    think_toks = t.randint(model.cfg.d_normal_vocab, model.cfg.d_vocab_total - 1, (cfg.group_size, 1))
                else:
                    logits = model(rollouts).squeeze()
                    sample_probs = t.softmax((logits[:, -1, model.cfg.d_normal_vocab:]), dim=-1) # get logpprob distn over thinking token
                    think_toks = t.multinomial(sample_probs, num_samples=1) + model.cfg.d_normal_vocab # sample a thinking token

                rollouts = t.cat([rollouts, think_toks], dim=1)

            rollouts = t.cat([rollouts, end_thoughts], dim=1)
            rollouts_no_question = rollouts[:, q_len:]
            logits = model(rollouts_no_question).squeeze()
            logprobs = t.log_softmax(logits[:, -1, :model.cfg.d_normal_vocab], dim=-1) # get the logprobs of the answer tokens
            pred_rewards = logprobs[:, ans_tok]  # ans_tok is the single token ID

            pred_reward_mean = pred_rewards.mean().item() # mean logprob of correct answer token
            mc_pred_rewards = pred_rewards - pred_reward_mean # mean centered rewards
            normed_pred_rewards = (mc_pred_rewards / (pred_rewards.std() + 1e-8))
            pred_reward_var = pred_rewards.var().item() # variance of the mean centered rewards
            pred_probs = t.exp(pred_rewards) # convert logprobs to probabilities
            pred_prob_var = pred_probs.var().item() # variance of the probabilities of the correct answer token
            
            epsilon = max(epsilon * cfg.eps_decay, cfg.eps_min)
        
        rollouts = rollouts.clone()
        rollouts_no_question = rollouts[:, q_len:].clone()
        normed_pred_rewards = normed_pred_rewards.clone()
        logits = model(rollouts).squeeze()

        pred_logits = model(rollouts_no_question).squeeze()
        pred_logprobs = t.log_softmax(pred_logits[:, -1, :model.cfg.d_normal_vocab], dim=-1) # real token logprob distn on the end_thought token
        pred_rewards = pred_logprobs[:, ans_tok]
        pred_reward_mean = pred_rewards.mean()

        think_logprobs = t.log_softmax(logits[group_indices, (think_indices - 1).unsqueeze(0), model.cfg.d_normal_vocab:], dim=-1) # logprob distns for each thinking token position
        action_logprobs = think_logprobs[group_indices, think_indices - q_len, rollouts[:, think_indices] - model.cfg.d_normal_vocab] # logprob of the thinking tokens that were outputted
        weighted_think_logprobs = action_logprobs * normed_pred_rewards.unsqueeze(-1) # logprobs times rewards
        think_reward = weighted_think_logprobs.mean(dim=0) # mean over the group size
        think_reward_mean = think_reward.mean() # mean of the think rewards

        total_reward = (1 - cfg.think_reward_weight) * pred_reward_mean + cfg.think_reward_weight * think_reward_mean
        total_reward.backward()

        if b != 0 and b % cfg.batch_size == 0:
            opt.step()
            scheduler.step()
            opt.zero_grad()

            wandb.log({
                "pred_reward": pred_reward_mean,
                "think_reward": think_reward_mean,
                "total_reward": total_reward,
                "num_think": cfg.think_len,
                "pred_reward_var": pred_reward_var,
                "pred_prob_var": pred_prob_var,
                "prob_force_end_thought": 0.0,
                "epsilon": epsilon,
            })
            #printSeq(rollouts[0], simple_tokenizer, model.cfg)
            tr.set_description(f"{magenta}pred reward mean: {pred_reward_mean:.3f}, total reward: {total_reward.item():.3f}, think reward: {think_reward_mean:.3f}, epsilon: {epsilon:.3f}")

        if b % 32_000 == 0:
            t.save(model.state_dict(), f"saves/add_think_fixed_blind{b}.pt")
            _, benchmark_accuracy = benchmark_addition_think_fixed_blind(model, testset, cfg.think_len)
            wandb.log({"benchmark_accuracy": benchmark_accuracy})

INPUT_MAX = 100
NUM_EXAMPLES = 1_000_000

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))

    model_cfg = ThinkingModelConfig(d_model=32, seq_len=32, d_mlp=128, d_head=16, n_heads=4, n_layers=2, d_normal_vocab=INPUT_MAX, d_thought_vocab=16)
    training_cfg = TrainingConfig(
        think_len=16,
        group_size=32,
        think_reward_weight=0.5,
        eps_decay=0.999995,
        eps_min=0.05,
        batch_size=16,
        lr=3e-4,
        weight_decay=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.95
    )
    model = GPT2Thinking(model_cfg)

    simple_tokenizer = SimpleTokenizer(max_int=INPUT_MAX)
    trainset, testset = makeAdditionDataset(simple_tokenizer, INPUT_MAX, NUM_EXAMPLES, train_split=0.99)

    train(model, training_cfg, trainset)
    benchmark_addition_think_fixed_blind(model, testset, training_cfg.think_len)