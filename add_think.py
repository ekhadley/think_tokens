import tqdm
import datasets
import wandb
import torch as t
from torch import nn
from transformers import GPT2TokenizerFast, AutoTokenizer
import random
import pandas as pd
from eindex import eindex

from supervised_rollout_think import GPT2Thinking
from add_normal import makeAdditionDataset, SimpleTokenizer
from utils import *

simple_tokenizer = SimpleTokenizer()

def printSeq(seq: t.Tensor, cfg: ThinkingModelConfig, end_thought: int) -> None:
    seq = seq.squeeze()
    print()
    for token in seq:
        if token < cfg.d_normal_vocab:
            print(blue, simple_tokenizer.vocab[token], endc, end="", sep="")
        elif token == end_thought:
            print(magenta, "<end_thought>", endc, end="", sep="")
        else:
            print(cyan, f"<think{token.item() - cfg.d_normal_vocab}>", endc, end="", sep="")
    print()


def benchmark_addition_think(model: GPT2Thinking, dataset: pd.DataFrame, max_answer_len: int = 10, group_size: int = 1):
    """
    Benchmarks the thinking model's addition ability on a dataset.
    For each question, generates a rollout by sampling thinking tokens, then appends the answer tokens.
    Returns:
        - mean_logprob: mean aggregate logprob over correct answer tokens
        - accuracy: fraction of exact matches using argmax sampling
    """
    model.eval()
    total_logprob = 0.0
    total_tokens = 0
    correct = 0
    n = len(dataset)
    for i, row in tqdm.tqdm(enumerate(dataset.itertuples()), total=n, desc="BenchmarkThink", ncols=100):
        q_toks = t.tensor(row.question_toks)
        ans_toks = t.tensor(row.answer_toks)
        q_len = row.question_len
        a_len = row.answer_len

        # Generate rollout (simulate thinking)
        rollout = q_toks.clone()
        with t.no_grad():
            for i in range(q_len, model.cfg.seq_len - a_len):
                logits = model(rollout).squeeze()
                logprobs = t.log_softmax(logits[..., model.cfg.d_normal_vocab:], dim=-1)
                if i == model.cfg.seq_len - a_len - 1: think_tok = model.end_thought
                else: think_tok = logprobs[-1].argmax().item() + model.cfg.d_normal_vocab
                rollout = t.cat([rollout, t.tensor([think_tok], device=rollout.device)])
                if think_tok == model.end_thought: break
            rollout = t.cat([rollout, ans_toks], dim=0)
            logits = model(rollout).squeeze()
            logprobs = t.log_softmax(logits[..., :model.cfg.d_normal_vocab], dim=-1)
            rollout_len = rollout.shape[0]
            ans_logprobs = logprobs[range(rollout_len - a_len - 1, rollout_len - 1), rollout[rollout_len - a_len:]]
            total_logprob += ans_logprobs.sum().item()
            total_tokens += a_len
            answer_logits = logits[rollout_len - a_len - 1: rollout_len - 1, :model.cfg.d_normal_vocab]
            generated = answer_logits.argmax(dim=-1).cpu().tolist()
            if generated == ans_toks.cpu().tolist():
                correct += 1
    mean_logprob = total_logprob / total_tokens if total_tokens > 0 else float('nan')
    accuracy = correct / n if n > 0 else float('nan')
    print(f"[Think] Mean logprob: {mean_logprob:.4f}, Accuracy: {accuracy:.4f}")
    return mean_logprob, accuracy


def train(model: GPT2Thinking, cfg: TrainingConfig, dataset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)

    wandb.init(project="add_thoughtful_think", name="think", config=cfg)
    wandb.watch(model, log="all")

    group_size = 16
    beta = 1.0

    for b in (tr:=tqdm.trange(len(dataset), ncols=200)):
        row = dataset.iloc[b]
        q_toks = t.tensor(row["question_toks"])
        ans_toks = t.tensor(row["answer_toks"])
        q_len = row["question_len"]
        a_len = row["answer_len"]

        rollouts = []
        rewards = []
        with t.inference_mode(): # do inference without gradients to generate rollouts
            for g in range(group_size): # for each rollout in the group
                rollout = q_toks.clone()
                for _ in range(model.cfg.seq_len - a_len): # for each think token in the rollout
                    logits = model(rollout).squeeze()
                    logprobs = t.log_softmax(logits[..., model.cfg.d_normal_vocab:], dim=-1) # get logpprob distn over thinking tokens
                    think_tok = sampleLogprobs(logprobs[..., -1, :], temperature=0.7).reshape(1, 1) + model.cfg.d_normal_vocab # sample a thinking token
                    if random.random() > 0.7: think_tok = t.tensor(model.end_thought).reshape(1, 1)
                    rollout = t.cat([rollout.unsqueeze(1), think_tok], dim=0).squeeze() #
                    if think_tok == model.end_thought: # end of thinking. get logprobs on answer (reward)
                        break
                rollout = t.cat([rollout, ans_toks], dim=0) # append the answer tokens
                logits = model(rollout).squeeze()
                logprobs = t.log_softmax(logits[..., :model.cfg.d_normal_vocab], dim=-1) # get the logprobs of the answer tokens
                rollout_len = rollout.shape[0]
                ans_logprobs = logprobs[range(rollout_len - a_len - 1, rollout_len - 1), rollout[rollout_len - a_len:]] # get the logprobs of the answer tokens
                reward = ans_logprobs.sum() # reward is the logprob on correct answer tokens

                rewards.append(reward.cpu().item())
                rollouts.append(rollout)
        
        reward_mean = sum(rewards) / len(rewards)
        normed_rewards = [r - reward_mean for r in rewards]
        
        num_think_per_rollout = [rollouts[i].shape[0] - q_len - a_len for i in range(group_size)]
        mean_num_think = sum(num_think_per_rollout) / len(num_think_per_rollout)

        think_losses = []
        losses = []
        entropy_losses = []
        for g in range(group_size): # we run the rollouts back though with gradients to calculate the loss for each
            rollout = rollouts[g].clone()
            logits = model(rollout).squeeze()
            rollout_len = rollout.shape[0]

            ans_logprobs = t.log_softmax(logits[..., :model.cfg.d_normal_vocab], dim=-1)
            ans_indices = list(range(rollout_len - a_len - 1, rollout_len - 1))
            ans_logprobs = ans_logprobs[ans_indices, rollout[rollout_len - a_len:]] # get the logprobs of the answer tokens
            pred_loss = ans_logprobs.sum()

            think_logprobs = t.log_softmax(logits[..., model.cfg.d_normal_vocab:], dim=-1) # get the logprob distns of the positions which were thinking tokens
            think_indices = list(range(q_len - 1, rollout_len - a_len - 1))
            think_logprobs_sel = think_logprobs[think_indices, rollout[q_len: rollout_len - a_len] - model.cfg.d_normal_vocab] # get the logprobs of the thinking tokens that were outputted
            weighted_think_logprobs = think_logprobs_sel * normed_rewards[g] # weight the logprobs by the reward
            think_loss = weighted_think_logprobs.sum()

            # Entropy regularization for think token outputs
            think_logprobs_all = think_logprobs[think_indices] # shape: (num_think_steps, d_thought_vocab)
            entropy = -(think_logprobs_all.exp() * think_logprobs_all).sum(dim=-1) # shape: (num_think_steps,)
            entropy_loss = entropy.mean() # average over think steps
            entropy_losses.append(entropy_loss)

            loss = pred_loss + beta * think_loss + cfg.gamma * entropy_loss
            losses.append(loss)
            think_losses.append(think_loss.detach().item())
        
        loss = sum(losses) / len(losses)
        loss.backward()


        think_loss_mean = sum(think_losses) / len(think_losses)
        entropy_loss_mean = sum([e.item() for e in entropy_losses]) / len(entropy_losses)

        if b != 0 and b % cfg.batch_size == 0:
            opt.step()
            opt.zero_grad()
            wandb.log({"reward": reward_mean, "think_loss": think_loss_mean, "loss": loss.item(), "num_think": mean_num_think, "entropy_loss": entropy_loss_mean})
            tr.set_description(f"{magenta}reward mean: {reward_mean:.3f}, loss: {loss.item():.3f}, think_loss: {think_loss_mean:.3f}, entropy: {entropy_loss_mean:.3f}, num_think: {mean_num_think:.3f}{endc}")
            printSeq(rollouts[0], model.cfg, model.end_thought)


if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))

    model_cfg = ThinkingModelConfig(d_model=64, seq_len=128, d_mlp=512, d_head=16, n_heads=4, n_layers=4, d_normal_vocab=12, d_thought_vocab=100)
    #model_cfg = ModelConfig(d_model=512, seq_len=128, d_mlp=2048, d_head=64, n_heads=8, n_layers=8, d_vocab=50_257)
    model = GPT2Thinking(model_cfg)
    training_cfg = TrainingConfig(gamma=0.2, batch_size=16, lr=3e-4, weight_decay=1e-6, adam_beta1=0.9, adam_beta2=0.95)

    trainset, testset = makeAdditionDataset(simple_tokenizer, 100, 50_000, "additions_10K_100K", train_split=0.9)
    #dataset = pd.read_pickle("datasets/additions_10_1M.pkl")

    train(model, training_cfg, trainset)

    benchmark_addition_think(model, testset)