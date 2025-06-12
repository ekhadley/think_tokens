import math
import tqdm
import wandb
import random
import pandas as pd
import torch as t

from models import GPT2Thinking, TrainingConfig, ThinkingModelConfig, SimpleTokenizer, makeAdditionDataset
from utils import *

from add_think_fixed_blind import benchmark_addition_think_fixed_blind
from add_think_fixed_blind_super_search import bruteForceThoughtSearch

def train(model: GPT2Thinking, cfg: TrainingConfig, dataset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dataset)//cfg.batch_size)

    input_max = dataset.attrs["input_max"]
    wandb.init(project="add_thoughtful_think", name=f"think_fixed_blind_super_clean{input_max}", config=cfg)
    wandb.watch(model, log="all")
    wandb.config.update(model.cfg.to_dict())
    wandb.config.update(cfg.to_dict())

    q_len = dataset.attrs["question_len"]

    epsilon = 1.0 # prob of choosing random think token

    end_thoughts = t.tensor([model.end_thought] * cfg.group_size, requires_grad=False).unsqueeze(-1)  # end_thought token for each group
    group_indices = t.arange(cfg.group_size, requires_grad=False).unsqueeze(-1)
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
            rollouts = q_toks.unsqueeze(0).repeat(cfg.group_size, 1)
            for i in range(cfg.think_len): # for each think token in the rollout. end_thought counts as a thought.
                if random.random() < epsilon: # epsilon-greedy exploration sampling
                    think_toks = t.randint(model.cfg.d_normal_vocab, model.cfg.d_vocab_total - 1, (cfg.group_size, 1))
                else:
                    logits = model(rollouts).squeeze()
                    sample_probs = t.softmax((logits[:, -1, model.cfg.d_normal_vocab:-1]), dim=-1) # get logpprob distn over thinking token
                    think_toks = t.multinomial(sample_probs, num_samples=1) + model.cfg.d_normal_vocab # sample a thinking token

                rollouts = t.cat([rollouts, think_toks], dim=1)

            
            rollouts = t.cat([rollouts, end_thoughts], dim=1) # generating the rewards for the thinking token rl
            #rollouts_no_question = rollouts[:, q_len:]
            #logits = model(rollouts_no_question).squeeze()
            #logprobs = t.log_softmax(logits[:, -1, :model.cfg.d_normal_vocab], dim=-1) # get the logprobs of the answer tokens
            #pred_rewards = logprobs[:, ans_tok]  # ans_tok is the single token ID
            #pred_reward_mean = pred_rewards.mean().item() # mean of the predicted rewards
            #normed_pred_rewards = (pred_rewards - pred_reward_mean) / (pred_rewards.std() + 1e-8) # normalize the rewards
            
            #pred_rewards = (rollouts[:, q_len:q_len + cfg.think_len] == correct_thoughts[:cfg.think_len]).float().sum(dim=-1) * 50 ################ giving perfect reward signals. This is the 'clean' part.
            pred_rewards = (rollouts[:, q_len:q_len + cfg.think_len] == correct_thoughts[:cfg.think_len]).all(dim=-1).float() * 100 ################ giving perfect reward signals. This is the 'clean' part.
            pred_reward_mean = pred_rewards.mean().item()
            normed_pred_rewards = pred_rewards
            
            epsilon = max(epsilon * cfg.eps_decay, cfg.eps_min)
        
        rollouts = rollouts.clone() # sampled rollouts but with gradients on
        normed_pred_rewards = normed_pred_rewards.clone()
        logits = model(rollouts).squeeze()

        pred_logits = model(correct_thoughts).squeeze() ######################## just using the correct rollouts, not what we sampled. this is is 'super' part. Also the rollouts do not contain the original question, which is the 'blind' part.
        pred_logprobs = t.log_softmax(pred_logits[-1, :model.cfg.d_normal_vocab], dim=-1) # real token logprob distn on the end_thought token
        pred_reward = pred_logprobs[ans_tok] # logprob value on the correct answer token
        pred_reward_mean = pred_reward

        think_logprobs = t.log_softmax(logits[group_indices, (think_indices - 1).unsqueeze(0), model.cfg.d_normal_vocab:-1], dim=-1) # logprob distns for each thinking token position
        action_logprobs = think_logprobs[group_indices, think_indices - q_len, rollouts[:, think_indices] - model.cfg.d_normal_vocab] # logprob of the thinking tokens that were outputted
        weighted_think_logprobs = action_logprobs * normed_pred_rewards.unsqueeze(-1) # logprobs times rewards
        think_reward_mean = weighted_think_logprobs.mean()

        entropy = -(think_logprobs * t.exp(think_logprobs)).sum(dim=-1).mean()

        total_reward = (1 - cfg.think_reward_weight) * pred_reward_mean + cfg.think_reward_weight * think_reward_mean + cfg.entropy_reward_weight * entropy
        total_reward.backward()

        if b != 0 and b % cfg.batch_size == 0:
            opt.step()
            scheduler.step()
            opt.zero_grad()

            pred_prob_var = t.exp(pred_rewards).var().item() # answer prob variance for logging
            pred_reward_var = pred_rewards.var().item() # variance of the predicted rewards for logging

            think_loss = action_logprobs[(pred_rewards > 0)].mean()

            wandb.log({
                "pred_reward": pred_reward_mean,
                "think_reward": think_reward_mean,
                "total_reward": total_reward,
                "num_think": cfg.think_len,
                "pred_reward_var": pred_reward_var,
                "pred_prob_var": pred_prob_var,
                "prob_force_end_thought": 0.0,
                "epsilon": epsilon,
                "think_logprobs": think_logprobs[0],
                "entropy_reward": entropy,
                "think_loss": think_loss,
            })
            #printSeq(rollouts[0], simple_tokenizer, model.cfg)
            tr.set_description(f"{magenta}pred reward mean: {pred_reward_mean:.3f}, total reward: {total_reward.item():.3f}, think reward: {think_reward_mean:.3f}, epsilon: {epsilon:.3f}")

        if b % 32_000 == 0:
            print()
            print(red, ans_tok, endc)
            print(red, correct_thoughts, endc)
            rollout_mean_logprob = action_logprobs.mean(dim=-1)
            for row in range(rollouts.shape[0]):
                print(f"{blue}{rollouts[row].tolist()} {magenta}{rollout_mean_logprob[row].item():.3f} : {cyan}{pred_rewards[row].item():.3f} {green}({normed_pred_rewards[row].item():.3f}){endc}")
            #bruteForceThoughtSearch(model, ans_tok, cfg.think_len)
            _, benchmark_accuracy = benchmark_addition_think_fixed_blind(model, testset, cfg.think_len)
            wandb.log({"benchmark_accuracy": benchmark_accuracy})
            t.save(model.state_dict(), f"saves/add_think_fixed_blind_super_clean{b}.pth")



INPUT_MAX = 100
NUM_EXAMPLES = 1_000_000

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))

    model_cfg = ThinkingModelConfig(d_model=32, seq_len=32, d_mlp=128, d_head=16, n_heads=4, n_layers=2, d_normal_vocab=INPUT_MAX, d_thought_vocab=11)
    training_cfg = TrainingConfig(
        think_len=2,
        group_size=32,
        think_reward_weight=0.5,
        entropy_reward_weight=0.01,
        eps_decay=0.999995,
        eps_min=0.01,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-3,
        adam_beta1=0.9,
        adam_beta2=0.95
    )
    model = GPT2Thinking(model_cfg)

    simple_tokenizer = SimpleTokenizer(max_int=INPUT_MAX)
    trainset, testset = makeAdditionDataset(simple_tokenizer, INPUT_MAX, NUM_EXAMPLES, train_split=0.9999)

    train(model, training_cfg, trainset)
    benchmark_addition_think_fixed_blind(model, testset, training_cfg.think_len)