import math
import tqdm
import wandb
import random
import pandas as pd
import torch as t

from models import GPT2SplitModel, TrainingConfig, SplitModelConfig
from utils import *

def benchmark_addition_think_fixed_blind_split(answer_model: GPT2SplitModel, think_model: GPT2SplitModel, dataset: pd.DataFrame, think_len: int):
    answer_model.eval()
    think_model.eval()
    total_logprob = 0.0
    total_tokens = 0
    correct = 0
    q_len = dataset.attrs["question_len"]
    d_normal_vocab = dataset.attrs["input_max"]
    for i, row in tqdm.tqdm(enumerate(dataset.itertuples()), total=len(dataset), desc="BenchmarkThinkFixed", ncols=100):
        q_toks = t.tensor(row.question_toks)
        ans_tok = row.answer_tok

        rollout = q_toks.clone()
        with t.no_grad():
            for i in range(think_len):
                logits = think_model(rollout).squeeze()
                think_tok = logits[-1].argmax() + d_normal_vocab
                rollout = t.cat([rollout, think_tok.unsqueeze(0)])
            
            rollout_no_question = rollout[q_len:] - d_normal_vocab
            logits = answer_model(rollout_no_question).squeeze()
            logprobs = t.log_softmax(logits[-1], dim=-1)
            ans_logprob = logprobs[ans_tok]
            total_logprob += ans_logprob.item()
            total_tokens += 1
            generated = logprobs.argmax().item()

            if generated == ans_tok:
                correct += 1

    mean_logprob = total_logprob / total_tokens if total_tokens > 0 else float('nan')
    accuracy = correct / len(dataset)
    print(yellow, f"[ThinkFixed] Mean logprob: {mean_logprob:.4f}, Accuracy: {accuracy:.4f}", endc)
    return mean_logprob, accuracy

def train(answer_model: GPT2SplitModel, think_model: GPT2SplitModel, cfg: TrainingConfig, dataset: pd.DataFrame):
    answer_opt = t.optim.AdamW(answer_model.parameters(), lr=cfg.answer_lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    #answer_scheduler = t.optim.lr_scheduler.CosineAnnealingLR(answer_opt, T_max=len(dataset)//cfg.batch_size)
    answer_scheduler = t.optim.lr_scheduler.CosineAnnealingLR(answer_opt, T_max=2000)
    
    think_opt = t.optim.AdamW(think_model.parameters(), lr=cfg.think_lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    think_scheduler = t.optim.lr_scheduler.CosineAnnealingLR(think_opt, T_max=len(dataset)//cfg.batch_size)

    input_max = dataset.attrs["input_max"]
    ndig = int(math.log10(input_max))
    q_len = dataset.attrs["question_len"]
    d_normal_vocab = input_max
    d_vocab_total = input_max + think_model.cfg.d_thought_vocab

    wandb.init(project="add_thoughtful_think", name=f"think_fixed_blind_super_split{input_max}", config=cfg)
    #wandb.config.update(answer_model.cfg.to_dict())
    #wandb.config.update(think_model.cfg.to_dict())
    #wandb.config.update(cfg.to_dict())

    epsilon = 1.0 # prob of choosing random think token

    group_indices = t.arange(cfg.group_size, requires_grad=False).unsqueeze(-1)
    think_indices = t.arange(q_len, q_len + cfg.think_len, requires_grad=False)

    for b in (tr:=tqdm.trange(len(dataset), ncols=200)):
        row = dataset.iloc[b]
        q_toks = t.tensor(row["question_toks"])
        ans_tok = row["answer_tok"]  # Single token, not tensor


        ans_digits = [int(c) for c in str(ans_tok.item())] # manually creating the 'correct' chain of thought tokens
        while len(ans_digits) < ndig: ans_digits.insert(0, 0)
        correct_thoughts = t.tensor(ans_digits)

        with t.inference_mode(): # do inference without gradients to generate rollouts
            rollouts = q_toks.unsqueeze(0).repeat(cfg.group_size, 1)
            for i in range(cfg.think_len): # for each think token in the rollout.
                if random.random() < epsilon: # epsilon-greedy exploration sampling
                    think_toks = t.randint(d_normal_vocab, d_vocab_total, (cfg.group_size, 1))
                else:
                    logits = think_model(rollouts).squeeze()
                    sample_probs = t.softmax((logits[:, -1]), dim=-1) # get logpprob distn over thinking token
                    think_toks = t.multinomial(sample_probs, num_samples=1) + d_normal_vocab # sample a thinking token. thinking model input expects thinking tokens in the range [d_normal_vocab, d_normal_vocab + d_thought_vocab)

                rollouts = t.cat([rollouts, think_toks], dim=1)
            
            rollouts_no_question = rollouts[:, q_len:] - d_normal_vocab # answering model input expects thinking tokens in the range [0, d_thought_vocab)
            #logits = answer_model(rollouts_no_question).squeeze()
            #logprobs = t.log_softmax(logits[:, -1], dim=-1)
            #pred_rewards = logprobs[:, ans_tok]  # ans_tok is the single token ID
            #pred_reward_mean = pred_rewards.mean().item() # mean of the predicted rewards
            #normed_pred_rewards = (pred_rewards - pred_reward_mean) / (pred_rewards.std() + 1e-8) # normalize the rewards
            #pred_rewards = (rollouts_no_question == correct_thoughts[:cfg.think_len]).float().sum(dim=-1) * 50
            pred_rewards = (rollouts_no_question == correct_thoughts[:cfg.think_len]).all(dim=-1).float() * 50
            pred_reward_mean = pred_rewards.mean().item()
            normed_pred_rewards = pred_rewards
            
            epsilon = max(epsilon * cfg.eps_decay, cfg.eps_min)
        
        rollouts = rollouts.clone() # sampled rollouts but with gradients on
        normed_pred_rewards = normed_pred_rewards.clone()

        pred_logits = answer_model(correct_thoughts).squeeze()
        pred_logprobs = t.log_softmax(pred_logits[-1], dim=-1) # real token logprob distn on the end_thought token
        pred_reward = pred_logprobs[ans_tok] # logprob value on the correct answer token
        pred_reward_mean = pred_reward
        pred_reward.backward()

        think_logits = think_model(rollouts).squeeze()
        think_logprobs = t.log_softmax(think_logits[group_indices, (think_indices - 1).unsqueeze(0)], dim=-1) # logprob distns for each thinking token position
        action_logprobs = think_logprobs[group_indices, think_indices - q_len, rollouts[:, think_indices] - d_normal_vocab] # logprob of the thinking tokens that were outputted
        weighted_action_logprobs = action_logprobs * normed_pred_rewards.unsqueeze(-1) # logprobs times rewards
        think_reward = weighted_action_logprobs.mean(dim=0) # mean over the group size
        think_reward_mean = think_reward.mean() # mean of the think rewards
        entropy = -(think_logprobs * t.exp(think_logprobs)).sum(dim=-1).mean()
        think_reward_total = entropy * cfg.entropy_reward_weight + think_reward_mean
        think_reward_total.backward()

        if b != 0 and b % cfg.batch_size == 0:
            answer_opt.step()
            answer_scheduler.step()
            think_opt.step()
            think_scheduler.step()

            pred_prob_var = t.exp(pred_rewards).var().item() # answer prob variance for logging
            pred_reward_var = pred_rewards.var().item() # variance of the predicted rewards for logging
            
            total_reward = think_reward_total + pred_reward_mean

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
            _, benchmark_accuracy = benchmark_addition_think_fixed_blind_split(answer_model, think_model, testset, cfg.think_len)
            wandb.log({"benchmark_accuracy": benchmark_accuracy})


INPUT_MAX = 100
NUM_EXAMPLES = 1_000_000

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))

    d_thought_vocab = 10
    answer_model_cfg = SplitModelConfig(d_model=32, seq_len=32, d_mlp=128, d_head=16, n_heads=4, n_layers=2, d_thought_vocab=d_thought_vocab, d_vocab_in=d_thought_vocab, d_vocab_out=INPUT_MAX)
    think_model_cfg =  SplitModelConfig(d_model=32, seq_len=32, d_mlp=128, d_head=16, n_heads=4, n_layers=2, d_thought_vocab=d_thought_vocab, d_vocab_in=INPUT_MAX + d_thought_vocab, d_vocab_out=d_thought_vocab)
    training_cfg = TrainingConfig(
        think_lr=5e-4,
        answer_lr=1e-4,
        weight_decay=1e-6,
        entropy_reward_weight=0.01,
        think_len=2,
        group_size=32,
        eps_decay=0.999995,
        eps_min=0.01,
        batch_size=16,
        adam_beta1=0.9,
        adam_beta2=0.95
    )
    answer_model = GPT2SplitModel(answer_model_cfg)
    think_model = GPT2SplitModel(think_model_cfg)

    simple_tokenizer = SimpleTokenizer(max_int=INPUT_MAX)
    trainset, testset = makeAdditionDataset(simple_tokenizer, INPUT_MAX, NUM_EXAMPLES, train_split=0.9999)

    train(answer_model, think_model, training_cfg, trainset)
    benchmark_addition_think_fixed_blind_split(answer_model, think_model, testset, training_cfg.think_len)