import math
import tqdm
import wandb
import random
import pandas as pd
import torch as t

from models import GPT2SplitModel, TrainingConfig, SplitModelConfig
from utils import *

def benchmark_addition_think_fixed_blind_split(answer_model: GPT2SplitModel, think_model: GPT2SplitModel, dataset: pd.DataFrame, think_len: int, cat_end_thought: bool = False):
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
def train(answer_model: GPT2SplitModel, think_model: GPT2SplitModel, cfg: TrainingConfig, dataset: pd.DataFrame):
    answer_opt = t.optim.AdamW(answer_model.parameters(), lr=cfg.answer_lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    think_opt = t.optim.AdamW(think_model.parameters(), lr=cfg.think_lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    answer_model.train()
    think_model.train()

    input_max = dataset.attrs["input_max"]
    q_len = dataset.attrs["question_len"]
    d_normal_vocab = input_max
    d_vocab_total = input_max + think_model.cfg.d_thought_vocab

    wandb.init(project="add_thoughtful_think", name=f"think_fixed_blind_split_{input_max}x{q_len}", config=cfg)
    #wandb.config.update(answer_model.cfg.to_dict())
    #wandb.config.update(think_model.cfg.to_dict())
    wandb.config.update(cfg.to_dict())

    epsilon = 1.0 # prob of choosing random think token

    batch_indices = t.arange(cfg.batch_size, requires_grad=False)
    group_indices = t.arange(cfg.group_size, requires_grad=False).unsqueeze(-1)
    full_batch_size = cfg.group_size * cfg.batch_size
    full_batch_indices = t.arange(full_batch_size, requires_grad=False)
    think_indices = t.arange(q_len, q_len + cfg.think_len, requires_grad=False)
    #end_thoughts = t.tensor([end_thought] * cfg.group_size, requires_grad=False).unsqueeze(-1) # end_thought token for each group

    answer_train_stop = 1e9
    benchmark_accuracy = 0.0

    n_batches = len(dataset) // cfg.batch_size
    for b in (tr:=tqdm.trange(n_batches, ncols=150)):
        b_i = b * cfg.batch_size
        q_toks = t.tensor(np.stack(dataset.iloc[b_i:b_i+cfg.batch_size]['question_toks']))
        ans_toks = t.tensor(dataset.iloc[b_i:b_i+cfg.batch_size]['answer_tok'].to_numpy())
        #print(red, q_toks, endc)
        q_toks = q_toks.unsqueeze(0).repeat(1, 1, cfg.group_size).reshape(full_batch_size, -1) # repeat each seq in the batch group_size times
        ans_toks = ans_toks.reshape(-1, 1).repeat(1, cfg.group_size).flatten() # correct answer token for each sequence in the batch
        #print(orange, q_toks, cyan, ans_toks, endc)

        with t.inference_mode(): # do inference without gradients to generate rollouts
            rollouts = q_toks.clone()
            for i in range(cfg.think_len): # for each think token in the rollout.
                if random.random() < epsilon: # epsilon-greedy exploration sampling
                    think_toks = t.randint(d_normal_vocab, d_vocab_total - 1, (full_batch_size, 1))
                else:
                    logits = think_model(rollouts).squeeze()
                    sample_probs = t.softmax((logits[:, -1]), dim=-1) # get logpprob distn over thinking token
                    think_toks = t.multinomial(sample_probs, num_samples=1) + d_normal_vocab # sample a thinking token. thinking model input expects thinking tokens in the range [d_normal_vocab, d_normal_vocab + d_thought_vocab)

                rollouts = t.cat([rollouts, think_toks], dim=1)
            
            #rollouts_no_question = t.cat([rollouts[:, q_len:] - d_normal_vocab, end_thoughts], dim=-1) # add end_thought token and shift token ids
            rollouts_no_question = rollouts[:, q_len:] - d_normal_vocab
            logits = answer_model(rollouts_no_question).squeeze()
            logprobs = t.log_softmax(logits[:, -1], dim=-1)
            pred_rewards = logprobs[full_batch_indices, ans_toks]
            
            pred_reward_mean = pred_rewards.mean().item() # mean of the predicted rewards
            normed_pred_rewards = t.clamp_min((pred_rewards - pred_reward_mean) / (pred_rewards.var() + 1e-8), 0) # normalize the rewards

            epsilon = max(epsilon * cfg.eps_decay, cfg.eps_min)
        
        rollouts = rollouts.clone() # sampled rollouts but with gradients on
        rollouts_no_question = rollouts_no_question.clone()
        normed_pred_rewards = normed_pred_rewards.clone()

        if b < answer_train_stop:
            pred_logits = answer_model(rollouts_no_question).squeeze()
            pred_logprobs = t.log_softmax(pred_logits[:, -1], dim=-1) # real token logprob distn on the end_thought token
            pred_reward = pred_logprobs[full_batch_indices, ans_toks].sum() # logprob value on the correct answer token
            answer_opt.zero_grad()
            pred_reward.backward()
            answer_opt.step()

        think_logits = think_model(rollouts)
        think_logprobs = t.log_softmax(think_logits[full_batch_indices, q_len-1:-1], dim=-1) # logprob distns for each thinking token position
        action_logprobs = think_logprobs[full_batch_indices.unsqueeze(-1), t.arange(think_len).unsqueeze(0), rollouts_no_question] # logprob of the thinking tokens that were outputted
        weighted_action_logprobs = action_logprobs * normed_pred_rewards.unsqueeze(-1) # logprobs times rewards
        think_reward = weighted_action_logprobs.sum() # sum of the think rewards
        think_reward.backward()
        think_opt.step()
        think_opt.zero_grad()

        with t.inference_mode():
            pred_prob_var = t.exp(pred_rewards).var().item() # answer prob variance for logging
            pred_reward_var = pred_rewards.var().item() # variance of the predicted rewards for logging

            wandb.log({
                "pred_reward": pred_reward_mean,
                "think_reward": think_reward,
                "num_think": cfg.think_len,
                "pred_reward_var": pred_reward_var,
                "pred_prob_var": pred_prob_var,
                "epsilon": epsilon,
            })
            #printSeq(rollouts[0], simple_tokenizer, model.cfg)
            tr.set_description(f"{magenta}pred loss: {pred_reward/full_batch_size:.3f}, pred reward: {pred_reward_mean:.3f}, epsilon: {epsilon:.3f} bench acc: {benchmark_accuracy:.4f}")

            if b*cfg.batch_size % 32_000 == 0:
                print()
                rollout_mean_logprob = action_logprobs.mean(dim=-1)
                for row in range(rollouts.shape[0]):
                    print(f"{blue}{rollouts[row].tolist()} {magenta}{rollout_mean_logprob[row].item():.3f} : {cyan}{pred_rewards[row].item():.3f} {green}({normed_pred_rewards[row].item():.3f}){endc}")
                _, benchmark_accuracy = benchmark_addition_think_fixed_blind_split(answer_model, think_model, testset, cfg.think_len)

                wandb.log({"benchmark_accuracy": benchmark_accuracy})
                #t.save(answer_model.state_dict(), f"saves/add_think_fixed_blind_super_clean_split_answer{b}.pth")
                #t.save(think_model.state_dict(), f"saves/add_think_fixed_blind_super_clean_split_think{b}.pth")


INPUT_MAX = 100
NUM_EXAMPLES = 10_000_000
NUM_ADDS = 2

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    d_thought_vocab = 10
    think_len = 6
    answer_model_cfg = SplitModelConfig(d_model=32, seq_len=8, d_mlp=128, d_head=16, n_heads=4, n_layers=2, d_vocab_in=d_thought_vocab, d_vocab_out=INPUT_MAX, d_thought_vocab=d_thought_vocab)
    think_model_cfg =  SplitModelConfig(d_model=32, seq_len=8, d_mlp=128, d_head=16, n_heads=4, n_layers=2, d_vocab_in=INPUT_MAX + d_thought_vocab, d_vocab_out=d_thought_vocab, d_thought_vocab=d_thought_vocab)
    training_cfg = TrainingConfig(
        think_len=think_len,
        answer_lr=1e-3,
        think_lr=1e-3,
        batch_size=64,
        weight_decay=1e-9,
        entropy_reward_weight=0.0,
        group_size=32,
        eps_decay=0.9999,
        eps_min=0.0,
    )
    answer_model = GPT2SplitModel(answer_model_cfg)
    think_model = GPT2SplitModel(think_model_cfg)

    trainset, testset = makeMultiAdditionDataset(INPUT_MAX, NUM_ADDS, NUM_EXAMPLES, train_split=0.99)

    train(answer_model, think_model, training_cfg, trainset)
    benchmark_addition_think_fixed_blind_split(answer_model, think_model, testset, training_cfg.think_len)
