import math
import tqdm
import wandb
import pandas as pd
import torch as t

from utils import *
from models import GPT2SplitModel, TrainingConfig, SplitModelConfig
from add_think_fixed_blind_super_split import benchmark_addition_think_fixed_blind_split
from add_think_search import allPossibleRollouts

def train(answer_model: GPT2SplitModel, think_model: GPT2SplitModel, cfg: TrainingConfig, dataset: pd.DataFrame):
    answer_opt = t.optim.AdamW(answer_model.parameters(), lr=cfg.answer_lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    think_opt = t.optim.AdamW(think_model.parameters(), lr=cfg.think_lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)

    input_max = dataset.attrs["input_max"]
    ndig = int(math.log10(input_max))
    q_len = dataset.attrs["question_len"]
    d_normal_vocab = input_max
    d_vocab_total = think_model.cfg.d_vocab_in

    wandb.init(project="add_thoughtful_think", name=f"think_fixed_blind_super_split_search{input_max}", config=cfg.to_dict())
    #wandb.config.update(answer_model.cfg.to_dict())
    #wandb.config.update(think_model.cfg.to_dict())
    #wandb.config.update(cfg.to_dict())

    perms = allPossibleRollouts(d_normal_vocab, d_vocab_total, cfg.think_len)
    group_size = perms.shape[0]
    #end_thoughts = t.tensor([end_thought] * cfg.group_size, requires_grad=False).unsqueeze(-1) # end_thought token for each group
    #group_indices = t.arange(group_size, requires_grad=False).unsqueeze(-1)
    #think_indices = t.arange(q_len, q_len + cfg.think_len, requires_grad=False)

    answer_train_stop = 1e12

    for b in (tr:=tqdm.trange(len(dataset), ncols=200)):

        row = dataset.iloc[b]
        q_toks = t.tensor(row["question_toks"])
        ans_tok = row["answer_tok"]  # Single token, not tensor

        ans_digits = [int(c) for c in str(ans_tok.item())] # manually creating the 'correct' chain of thought tokens
        ans_digits = [0 for i in range(cfg.think_len - len(ans_digits))] + ans_digits
        correct_thoughts = t.tensor(ans_digits, requires_grad=False)

        with t.inference_mode(): # do inference without gradients to generate rollouts
            rollouts = t.cat([q_toks.unsqueeze(0).repeat(group_size, 1), perms], dim=1)

            #rollouts_no_question = rollouts[:, q_len:] - d_normal_vocab # add end_thought token and shift token ids
            #logits = answer_model(rollouts_no_question).squeeze()
            #logprobs = t.log_softmax(logits[:, -1], dim=-1)
            #pred_rewards = logprobs[:, ans_tok]  # ans_tok is the single token ID
            #pred_reward_mean = pred_rewards.mean().item() # mean of the predicted rewards
            #normed_pred_rewards = (pred_rewards - pred_reward_mean) / (pred_rewards.std() + 1e-8) # normalize the rewards
            #pred_rewards = pred_rewards.softmax(dim=0)
            
            pred_rewards = ((rollouts[:, q_len:q_len + cfg.think_len] - d_normal_vocab) == correct_thoughts[:cfg.think_len]).all(dim=-1).float()
            normed_pred_rewards = pred_rewards
            
        rollouts = rollouts.clone() # sampled rollouts but with gradients on
        normed_pred_rewards = normed_pred_rewards.clone()

        if b < answer_train_stop:
            pred_logits = answer_model(correct_thoughts).squeeze()
            pred_logprobs = t.log_softmax(pred_logits[-1], dim=-1) # real token logprob distn on the end_thought token
            pred_reward = pred_logprobs[ans_tok] # logprob value on the correct answer token
            pred_reward_mean = pred_reward
            pred_reward.backward()

        #think_logits = think_model(rollouts).squeeze()
        #think_logprobs = t.log_softmax(think_logits[group_indices, (think_indices - 1).unsqueeze(0)], dim=-1) # logprob distns for each thinking token position
        #action_logprobs = think_logprobs[group_indices, think_indices - q_len, rollouts[:, think_indices] - d_normal_vocab] # logprob of the thinking tokens that were outputted
        #weighted_action_logprobs = action_logprobs * normed_pred_rewards.unsqueeze(-1) # logprobs times rewards
        #think_reward = weighted_action_logprobs.sum() # sum of the think rewards
        #think_reward.backward()
        qwe = t.cat([q_toks, correct_thoughts], dim=-1)
        think_logits = think_model(qwe).squeeze()
        think_logprobs = t.log_softmax(think_logits[1:-1], dim=-1)
        think_rewards = think_logprobs[t.arange(ndig), correct_thoughts]
        #think_rewards = think_logprobs[0, ans_digits[-1]%2]
        think_reward = think_rewards.sum()
        think_reward.backward()
        
        #entropy = -(think_logprobs * t.exp(think_logprobs)).sum(dim=-1).mean()
        #think_reward_total = entropy * cfg.entropy_reward_weight + think_reward
        #think_reward_total.backward()

        if b != 0 and b % cfg.batch_size == 0:
            if b < answer_train_stop:
                answer_opt.step()
                answer_opt.zero_grad()
            
            think_opt.step()
            think_opt.zero_grad()
            

            pred_prob_var = t.exp(pred_rewards).var().item() # answer prob variance for logging
            pred_reward_var = pred_rewards.var().item() # variance of the predicted rewards for logging
            
            #think_loss = action_logprobs[(pred_rewards > 0)].mean()
            
            wandb.log({
                "pred_reward": pred_reward_mean,
                "think_reward": think_reward,
                "num_think": cfg.think_len,
                "pred_reward_var": pred_reward_var,
                "pred_prob_var": pred_prob_var,
                "think_logprobs": think_logprobs[0].tolist(),
                #"entropy_reward": entropy,
                "think_loss": think_reward.item(),
            })
            #printSeq(rollouts[0], simple_tokenizer, model.cfg)
            tr.set_description(f"{magenta}pred reward mean: {pred_reward_mean:.3f}, think reward: {think_reward.item():.3f}")

        if b % 32_000 == 0:
            print()
            print(lime, rollouts.shape, endc)
            print(red, rollouts[ans_tok], endc)
            _, benchmark_accuracy = benchmark_addition_think_fixed_blind_split(answer_model, think_model, testset, cfg.think_len)
            wandb.log({"benchmark_accuracy": benchmark_accuracy})
            t.save(answer_model.state_dict(), f"saves/add_think_fixed_blind_super_split_search_answer{b}.pth")
            t.save(think_model.state_dict(), f"saves/add_think_fixed_blind_super_split_search_think{b}.pth")


INPUT_MAX = 1000
NUM_EXAMPLES = 1_000_000

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))

    d_thought_vocab = 10
    answer_model_cfg = SplitModelConfig(d_model=64, seq_len=32, d_mlp=128, d_head=16, n_heads=4, n_layers=1, d_vocab_in=d_thought_vocab, d_vocab_out=INPUT_MAX, d_thought_vocab=d_thought_vocab)
    think_model_cfg =  SplitModelConfig(d_model=256, seq_len=32, d_mlp=1024, d_head=32, n_heads=4, n_layers=6, d_vocab_in=INPUT_MAX + d_thought_vocab, d_vocab_out=d_thought_vocab, d_thought_vocab=d_thought_vocab)
    training_cfg = TrainingConfig(
        think_len=3,
        think_lr=1e-4,
        answer_lr=1e-4,
        entropy_reward_weight=0.01,
        batch_size=16,
        weight_decay=1e-3,
        adam_beta1=0.9,
        adam_beta2=0.95
    )
    answer_model = GPT2SplitModel(answer_model_cfg)
    think_model = GPT2SplitModel(think_model_cfg)

    simple_tokenizer = SimpleTokenizer(max_int=INPUT_MAX)
    trainset, testset = makeAdditionDataset(simple_tokenizer, INPUT_MAX, NUM_EXAMPLES, train_split=0.9995)

    try:
        train(answer_model, think_model, training_cfg, trainset)
    except KeyboardInterrupt:
        pass
    benchmark_addition_think_fixed_blind_split(answer_model, think_model, testset, training_cfg.think_len)
