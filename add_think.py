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

def printSeq(seq: t.Tensor, tokenizer, cfg: ThinkingModelConfig) -> None:
    seq = seq.squeeze()
    print()
    for token in seq:
        if token < cfg.d_normal_vocab:
            print(blue, tokenizer.id_to_token[token.item()], endc, end="", sep="")
        elif token == cfg.d_vocab_total - 1:
            print(magenta, "<end_thought>", endc, end="", sep="")
        else:
            print(cyan, f"<think{token.item() - cfg.d_normal_vocab}>", endc, end="", sep="")
    print()

def benchmark_addition_think(model: GPT2Thinking, dataset: pd.DataFrame, max_answer_len: int = 10, group_size: int = 1):
    """
    Benchmarks the thinking model's addition ability on a dataset.
    For each question, generates a rollout by sampling thinking tokens, then predicts the single answer token.
    Returns:
        - mean_logprob: mean logprob over correct answer tokens
        - accuracy: fraction of exact matches using argmax sampling
    """
    model.eval()
    prob_force_end_thought = 0.1
    total_logprob = 0.0
    total_tokens = 0
    correct = 0
    n = len(dataset)
    for i, row in tqdm.tqdm(enumerate(dataset.itertuples()), total=n, desc="BenchmarkThink", ncols=100):
        q_toks = t.tensor(row.question_toks)
        ans_tok = row.answer_tok  # Single token
        q_len = row.question_len

        rollout = q_toks.clone()
        with t.no_grad():
            for i in range(q_len, model.cfg.seq_len - 1):  # Reserve 1 position for answer
                logits = model(rollout).squeeze()
                logprobs = t.log_softmax(logits[..., model.cfg.d_normal_vocab:], dim=-1)
                if i == model.cfg.seq_len - 2 or random.random() < prob_force_end_thought: 
                    think_tok = model.end_thought
                else: 
                    think_tok = logprobs[-1].argmax().item() + model.cfg.d_normal_vocab
                rollout = t.cat([rollout, t.tensor([think_tok], device=rollout.device)])
                if think_tok == model.end_thought: break
            
            # Predict answer token
            logits = model(rollout).squeeze()
            logprobs = t.log_softmax(logits[..., :model.cfg.d_normal_vocab], dim=-1)
            ans_logprob = logprobs[-1, ans_tok]
            total_logprob += ans_logprob.item()
            total_tokens += 1
            
            answer_logit = logits[-1, :model.cfg.d_normal_vocab]
            generated = answer_logit.argmax().item()
            if generated == ans_tok:
                correct += 1
    mean_logprob = total_logprob / total_tokens if total_tokens > 0 else float('nan')
    accuracy = correct / n if n > 0 else float('nan')
    print(f"[Think] Mean logprob: {mean_logprob:.4f}, Accuracy: {accuracy:.4f}")
    return mean_logprob, accuracy


def train(model: GPT2Thinking, cfg: TrainingConfig, dataset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dataset)//cfg.batch_size)

    input_max = dataset.attrs["input_max"]
    wandb.init(project="add_thoughtful_think", name=f"think_nothink_{input_max}", config=cfg)
    wandb.watch(model, log="all")

    q_len = dataset.attrs["question_len"]

    group_size = 16
    think_reward_weight = 0.0
    entropy_reward_weight = 0.0

    prob_force_end_thought = 1.0
    epsilon = 1.0 # prob of choosing random think token
    eps_decay = 0.999995
    eps_min = 0.05

    for b in (tr:=tqdm.trange(len(dataset), ncols=200)):
        row = dataset.iloc[b]
        q_toks = t.tensor(row["question_toks"])
        ans_tok = row["answer_tok"]  # Single token, not tensor


        rollouts, pred_rewards = [], []
        with t.inference_mode(): # do inference without gradients to generate rollouts
            for g in range(group_size): # for each rollout in the group
                rollout = q_toks.clone()
                max_think_idx = model.cfg.seq_len - 1  # Reserve 1 position for answer prediction
                for i in range(q_len, max_think_idx): # for each think token in the rollout
                    if random.random() < epsilon:
                        think_tok = t.tensor([random.randint(model.cfg.d_normal_vocab, model.cfg.d_vocab_total - 1)])
                    else:
                        logits = model(rollout).squeeze()
                        sample_probs = t.softmax((logits[-1, model.cfg.d_normal_vocab:]), dim=-1) # get logpprob distn over thinking token
                        think_tok = t.multinomial(sample_probs, num_samples=1) + model.cfg.d_normal_vocab # sample a thinking token
                    if i == max_think_idx - 1 or random.random() < prob_force_end_thought: 
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

            epsilon = max(epsilon * eps_decay, eps_min)
        
        pred_rewards = t.tensor(pred_rewards, requires_grad=False)
        pred_reward_mean = pred_rewards.mean().item() # mean logprob of correct answer token
        mc_pred_rewards = pred_rewards - pred_reward_mean # mean centered rewards
        normed_pred_rewards = mc_pred_rewards / (pred_rewards.std() + 1e-8)
        pred_reward_var = pred_rewards.var().item() # variance of the mean centered rewards
        pred_probs = t.exp(pred_rewards) # convert logprobs to probabilities
        pred_prob_var = pred_probs.var().item() # variance of the probabilities of the correct answer token

        mean_num_think = sum([len(r) for r in rollouts]) / group_size - 4
        
        total_rewards, think_rewards, entropy_rewards = [], [], []
        for g in range(group_size): # we run the rollouts back through with gradients to get a differentiable reward
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

            total_reward = (1 - think_reward_weight)*pred_reward + think_reward_weight*think_reward + entropy_reward_weight*entropy_reward
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
                "prob_force_end_thought": prob_force_end_thought,
                "epsilon": epsilon,
                "think_logprobs": think_logprobs_all,
            })
            #printSeq(rollouts[0], simple_tokenizer, model.cfg)
            tr.set_description(f"{magenta}pred reward mean: {pred_reward_mean:.3f}, total reward: {total_reward.item():.3f}, think reward: {think_reward_mean:.3f}, entropy: {entropy_reward_mean:.3f}, epsilon: {epsilon:.3f}, num_think: {mean_num_think:.3f}")

        if b != 0 and b % 1000 == 0:
            t.save(model.state_dict(), f"saves/add_think2_{b}.pt")

INPUT_MAX = 100
NUM_EXAMPLES = 1_000_000

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))

    model_cfg = ThinkingModelConfig(d_model=32, seq_len=32, d_mlp=128, d_head=16, n_heads=4, n_layers=2, d_normal_vocab=INPUT_MAX + 2, d_thought_vocab=100)
    training_cfg = TrainingConfig(batch_size=16, lr=1e-3, weight_decay=1e-6, adam_beta1=0.9, adam_beta2=0.95)
    model = GPT2Thinking(model_cfg)

    simple_tokenizer = SimpleTokenizer(max_int=INPUT_MAX)
    trainset, testset = makeAdditionDataset(simple_tokenizer, INPUT_MAX, NUM_EXAMPLES, train_split=0.99)

    train(model, training_cfg, trainset)
    benchmark_addition_think(model, testset)
