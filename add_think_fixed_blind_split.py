import math
import tqdm
import wandb
import random
import pandas as pd
import torch as t

from models import GPT2SplitModel, TrainingConfig, SplitModelConfig
from utils import *

def __benchmark_addition_think_fixed_blind_split(answer_model: GPT2SplitModel, think_model: GPT2SplitModel, dataset: pd.DataFrame, think_len: int):
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
                think_tok = logits[-1, :-1].argmax() + d_normal_vocab
                rollout = t.cat([rollout, think_tok.unsqueeze(0)])
            
            rollout_no_question = t.cat([rollout[q_len:] - d_normal_vocab, t.tensor([think_model.cfg.d_thought_vocab - 1])], dim=0) # add end_thought token
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

def _benchmark_addition_think_fixed_blind_split(answer_model: GPT2SplitModel, think_model: GPT2SplitModel, dataset: pd.DataFrame, think_len: int, cat_end_thought: bool = False, display: bool = False):
    answer_model.eval()
    think_model.eval()
    total_logprob = 0.0
    total_tokens = 0
    correct = 0
    q_len = dataset.attrs["question_len"]
    d_normal_vocab = dataset.attrs["input_max"]
    end_thought_token = think_model.cfg.d_thought_vocab - 1
    dataset_iter = enumerate(dataset.itertuples())
    if display: dataset_iter = tqdm.tqdm(dataset_iter, total=len(dataset), desc="BenchmarkThinkFixedBlindSplit", ncols=100)
    for i, row in dataset_iter:
        q_toks = t.tensor(row.question_toks)
        ans_tok = row.answer_tok

        rollout = q_toks.clone()
        with t.no_grad():
            for i in range(think_len):
                think_logits = think_model(rollout).squeeze()
                think_tok = think_logits[-1].argmax() + d_normal_vocab
                rollout = t.cat([rollout, think_tok.unsqueeze(0)])
            #rollout_no_question = t.cat([rollout[q_len:] - d_normal_vocab, t.tensor([think_model.cfg.d_thought_vocab - 1])], dim=0) # add end_thought token
            rollout_no_question = rollout[q_len:] - d_normal_vocab
            if cat_end_thought: rollout_no_question = t.cat([rollout_no_question, t.tensor([end_thought_token])], dim=0)
            ans_logits = answer_model(rollout_no_question).squeeze()
            logprobs = t.log_softmax(ans_logits[-1], dim=-1)
            ans_logprob = logprobs[ans_tok]
            total_logprob += ans_logprob.item()
            total_tokens += 1
            generated = logprobs.argmax().item()

            if generated == ans_tok:
                correct += 1
    
    mean_logprob = total_logprob / total_tokens if total_tokens > 0 else float('nan')
    accuracy = correct / len(dataset)
    return mean_logprob, accuracy

def benchmark_addition_think_fixed_blind_split(answer_model: GPT2SplitModel, think_model: GPT2SplitModel, dataset: pd.DataFrame, think_len: int, cat_end_thought: bool = False, display: bool = False):
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
            think_logits = think_model(rollouts).squeeze()
            think_toks = think_logits[:, -1].argmax(dim=-1) + d_normal_vocab
            rollouts = t.cat([rollouts, think_toks.unsqueeze(-1)], dim=1)
        rollout_no_question = rollouts[:, q_len:] - d_normal_vocab
        if cat_end_thought: rollout_no_question = t.cat([rollout_no_question, end_thoughts], dim=0)
        ans_logits = answer_model(rollout_no_question).squeeze()
        logprobs = t.log_softmax(ans_logits[:, -1], dim=-1)
        ans_logprobs = logprobs[t.arange(n_examples), ans_toks]
        ans_guesses = logprobs.argmax(dim=-1)
        mean_logprob = ans_logprobs.mean().item()
        accuracy = (ans_guesses == ans_toks).float().mean().item()
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
    end_thought = input_max + think_model.cfg.d_thought_vocab - 1

    wandb.init(project="add_thoughtful_think", name=f"think_fixed_blind_super_split{input_max}", config=cfg)
    #wandb.config.update(answer_model.cfg.to_dict())
    #wandb.config.update(think_model.cfg.to_dict())
    #wandb.config.update(cfg.to_dict())

    epsilon = 1.0 # prob of choosing random think token
    answer_train_stop = 1e9

    group_indices = t.arange(cfg.group_size, requires_grad=False).unsqueeze(-1)
    think_indices = t.arange(q_len, q_len + cfg.think_len, requires_grad=False)
    end_thoughts = t.tensor([end_thought] * cfg.group_size, requires_grad=False).unsqueeze(-1) # end_thought token for each group

    for b in (tr:=tqdm.trange(len(dataset), ncols=200)):
        row = dataset.iloc[b]
        q_toks = t.tensor(row["question_toks"])
        ans_tok = row["answer_tok"]  # Single token, not tensor

        ans_digits = [int(c) for c in str(ans_tok.item())] # manually creating the 'correct' chain of thought tokens
        while len(ans_digits) < ndig: ans_digits.insert(0, 0)
        ans_digits.append(think_model.cfg.d_thought_vocab - 1)
        correct_thoughts = t.tensor(ans_digits, requires_grad=False)

        with t.inference_mode(): # do inference without gradients to generate rollouts
            rollouts = q_toks.unsqueeze(0).repeat(cfg.group_size, 1)
            for i in range(cfg.think_len): # for each think token in the rollout.
                if random.random() < epsilon: # epsilon-greedy exploration sampling
                    think_toks = t.randint(d_normal_vocab, d_vocab_total - 1, (cfg.group_size, 1))
                else:
                    logits = think_model(rollouts).squeeze()
                    sample_probs = t.softmax((logits[:, -1, :-1]), dim=-1) # get logpprob distn over thinking token
                    think_toks = t.multinomial(sample_probs, num_samples=1) + d_normal_vocab # sample a thinking token. thinking model input expects thinking tokens in the range [d_normal_vocab, d_normal_vocab + d_thought_vocab)

                rollouts = t.cat([rollouts, think_toks], dim=1)
            
            rollouts = t.cat([rollouts, end_thoughts], dim=1)
            rollouts_no_question = rollouts[:, q_len:] - d_normal_vocab # add end_thought token and shift token ids
            logits = answer_model(rollouts_no_question).squeeze()
            logprobs = t.log_softmax(logits[:, -1], dim=-1)
            pred_rewards = logprobs[:, ans_tok]  # ans_tok is the single token ID
            #pred_reward_mean = pred_rewards.mean().item() # mean of the predicted rewards
            #normed_pred_rewards = (pred_rewards - pred_reward_mean) / (pred_rewards.std() + 1e-8) # normalize the rewards
            normed_pred_rewards = pred_rewards.softmax(dim=0)
            pred_reward_mean = pred_rewards.mean().item()
            
            epsilon = max(epsilon * cfg.eps_decay, cfg.eps_min)
        
        rollouts = rollouts.clone() # sampled rollouts but with gradients on
        rollouts_no_question = rollouts_no_question.clone()
        normed_pred_rewards = normed_pred_rewards.clone()

        if b < answer_train_stop:
            pred_logits = answer_model(rollouts_no_question).squeeze()
            pred_logprobs = t.log_softmax(pred_logits[:, -1], dim=-1) # real token logprob distn on the end_thought token
            pred_reward = pred_logprobs[:, ans_tok].sum() # logprob value on the correct answer token
            pred_reward.backward()

        think_logits = think_model(rollouts).squeeze()
        think_logprobs = t.log_softmax(think_logits[group_indices, (think_indices - 1).unsqueeze(0), :-1], dim=-1) # logprob distns for each thinking token position
        action_logprobs = think_logprobs[group_indices, think_indices - q_len, rollouts[:, think_indices] - d_normal_vocab] # logprob of the thinking tokens that were outputted
        weighted_action_logprobs = action_logprobs * normed_pred_rewards.unsqueeze(-1) # logprobs times rewards
        think_reward = weighted_action_logprobs.sum() # mean over the group size
        think_reward.backward()

        #entropy = -(think_logprobs * t.exp(think_logprobs)).sum(dim=-1).mean()
        #think_reward_total = entropy * cfg.entropy_reward_weight + think_reward_mean
        #think_reward_total.backward()

        if b != 0 and b % cfg.batch_size == 0:
            if b < answer_train_stop:
                answer_opt.step()
                answer_scheduler.step()
                answer_opt.zero_grad()
            think_opt.step()
            think_scheduler.step()
            think_opt.zero_grad()

            pred_prob_var = t.exp(pred_rewards).var().item() # answer prob variance for logging
            pred_reward_var = pred_rewards.var().item() # variance of the predicted rewards for logging

            think_loss = action_logprobs[(pred_rewards > 0)].mean()
            
            wandb.log({
                "pred_reward": pred_reward_mean,
                "think_reward": think_reward,
                "num_think": cfg.think_len,
                "pred_reward_var": pred_reward_var,
                "pred_prob_var": pred_prob_var,
                "prob_force_end_thought": 0.0,
                "epsilon": epsilon,
                #"think_logprobs": think_logprobs[0],
                #"entropy_reward": entropy,
                "think_loss": think_loss,
            })
            #printSeq(rollouts[0], simple_tokenizer, model.cfg)
            tr.set_description(f"{magenta}pred reward mean: {pred_reward_mean:.3f}, think reward: {think_reward:.3f}, epsilon: {epsilon:.3f}")

        if b % 32_000 == 0:
            print()
            rollout_mean_logprob = action_logprobs.mean(dim=-1)
            for row in range(rollouts.shape[0]):
                print(f"{blue}{rollouts[row].tolist()} {magenta}{rollout_mean_logprob[row].item():.3f} : {cyan}{pred_rewards[row].item():.3f} {green}({normed_pred_rewards[row].item():.3f}){endc}")
            _, benchmark_accuracy = benchmark_addition_think_fixed_blind_split(answer_model, think_model, testset, cfg.think_len)
            best_rollout_idx = pred_rewards.argmax().item()
            print(red, correct_thoughts, yellow, rollouts[best_rollout_idx], endc)
            print(magenta, think_logprobs[best_rollout_idx].T, endc)

            wandb.log({"benchmark_accuracy": benchmark_accuracy})
            t.save(answer_model.state_dict(), f"saves/add_think_fixed_blind_super_clean_split_answer{b}.pth")
            t.save(think_model.state_dict(), f"saves/add_think_fixed_blind_super_clean_split_think{b}.pth")


INPUT_MAX = 100
NUM_EXAMPLES = 1_000_000

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))

    d_thought_vocab = 11
    answer_model_cfg = SplitModelConfig(d_model=32, seq_len=32, d_mlp=128, d_head=16, n_heads=4, n_layers=2, d_vocab_in=d_thought_vocab, d_vocab_out=INPUT_MAX, d_thought_vocab=d_thought_vocab)
    think_model_cfg =  SplitModelConfig(d_model=32, seq_len=32, d_mlp=128, d_head=16, n_heads=4, n_layers=2, d_vocab_in=INPUT_MAX + d_thought_vocab, d_vocab_out=d_thought_vocab, d_thought_vocab=d_thought_vocab)
    training_cfg = TrainingConfig(
        think_lr=1e-3,
        answer_lr=1e-3,
        weight_decay=1e-3,
        entropy_reward_weight=0.0,
        think_len=2,
        group_size=64,
        eps_decay=0.999998,
        eps_min=0.01,
        batch_size=16,
        adam_beta1=0.9,
        adam_beta2=0.95
    )
    answer_model = GPT2SplitModel(answer_model_cfg)
    think_model = GPT2SplitModel(think_model_cfg)

    simple_tokenizer = SimpleTokenizer(max_int=INPUT_MAX)
    trainset, testset = makeAdditionDataset(INPUT_MAX, NUM_EXAMPLES, train_split=0.99)

    train(answer_model, think_model, training_cfg, trainset)
    benchmark_addition_think_fixed_blind_split(answer_model, think_model, testset, training_cfg.think_len)
