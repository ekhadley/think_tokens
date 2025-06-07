import tqdm
import datasets
import wandb
import torch as t
from torch import nn
import random
import pandas as pd
import numpy as np
from eindex import eindex
from normal import GPT2
from utils import *

class SimpleTokenizer:
    def __init__(self, max_int):
        # Create vocabulary: just integers 0 to max_int
        self.vocab = [str(i) for i in range(max_int)]
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        self.max_int = max_int
    
    def encode(self, s):
        # For single numbers, return the token directly
        if s.isdigit():
            return [int(s)]
        
        # For pairs of numbers separated by space, return both tokens
        parts = s.split()
        tokens = []
        for part in parts:
            if part.isdigit():
                tokens.append(int(part))
        return tokens

    def decode(self, ids):
        return ''.join([self.id_to_token[i] for i in ids])


def makeAdditionDataset(tokenizer, int_max, n_questions, train_split: float = 1.0):
    question_str = []
    answer_str = []
    question_toks = []
    answer_tok = []  # Single token, not a list

    # Calculate all possible unique questions
    all_questions = [f"{n1} {n2}" for n1 in range(int_max) for n2 in range(int_max)]
    random.shuffle(all_questions)
    n_unique = len(all_questions)
    if n_questions > n_unique:
        print("Warning: dataset will contain duplicates. Test set will remain unique.")

    # Determine split sizes
    if train_split < 1.0:
        n_test = int(n_questions * (1 - train_split))
        n_train = n_questions - n_test
    else:
        n_test = 0
        n_train = n_questions

    # Assign test set from unique pool
    test_questions = all_questions[:n_test]
    train_questions = all_questions[n_test:n_unique]
    # If more training examples are needed, sample with replacement from all unique questions
    if n_train > len(train_questions):
        extra_needed = n_train - len(train_questions)
        train_questions += random.choices(all_questions, k=extra_needed)
    else:
        train_questions = train_questions[:n_train]

    def add_examples(questions):
        for question in questions:
            n1, n2 = map(int, question.split())
            answer = str((n1 + n2)%int_max)
            #answer = str(n1 + n2)
            toks = np.array(tokenizer.encode(question))
            ans_tok = tokenizer.encode(answer)[0]
            question_str.append(question)
            answer_str.append(answer)
            question_toks.append(toks)
            answer_tok.append(ans_tok)

    # Add training and test examples
    add_examples(train_questions)
    train_dataset = pd.DataFrame({
        "question": question_str,
        "answer": answer_str,
        "question_toks": question_toks,
        "answer_tok": answer_tok,
    })
    train_dataset.attrs['n_examples'] = n_questions
    train_dataset.attrs['input_max'] = int_max
    train_dataset.attrs['question_len'] = len(question_toks[0]) if train_questions else 0
    if n_test > 0:
        # Clear lists and add test examples
        question_str.clear(); answer_str.clear(); question_toks.clear(); answer_tok.clear()
        add_examples(test_questions)
        test_dataset = pd.DataFrame({
            "question": question_str,
            "answer": answer_str,
            "question_toks": question_toks,
            "answer_tok": answer_tok,
        })
        test_dataset.attrs['n_examples'] = n_questions
        test_dataset.attrs['input_max'] = int_max
        test_dataset.attrs['question_len'] = len(question_toks[0]) if test_questions else 0
        return train_dataset, test_dataset
    else:
        train_dataset.to_pickle(f"datasets/additions_{int_max}_{n_questions}.pkl")
        return train_dataset

def benchmark_addition(model: GPT2, dataset: pd.DataFrame, max_answer_len: int = 10):
    """
    Benchmarks the model's addition ability on a dataset.
    Returns:
        - mean_logprob: mean logprob over correct answer tokens
        - accuracy: fraction of exact matches using argmax sampling
    """
    model.eval()
    total_logprob = 0.0
    total_tokens = 0
    correct = 0
    n = len(dataset)
    q_len = dataset.attrs['question_len']
    for i, row in tqdm.tqdm(enumerate(dataset.itertuples()), total=n, desc="Benchmark", ncols=100):
        q_toks = t.tensor(row.question_toks)
        ans_tok = row.answer_tok  # Single token
        
        # Create full sequence by concatenating question and answer
        full_seq = t.cat([q_toks, t.tensor([ans_tok], device=q_toks.device)])
        
        # Get logits for the full sequence
        with t.no_grad():
            logits = model(full_seq).squeeze(0)  # [seq_len, vocab]
            logprobs = t.log_softmax(logits, dim=-1)
        
        # Get logprob for the single answer token
        ans_logprob = logprobs[q_len - 1, ans_tok]  # logprob of answer token conditioned on question
        total_logprob += ans_logprob.item()
        total_tokens += 1
        
        # Argmax sampling - predict single answer token
        with t.no_grad():
            logits = model(q_toks).squeeze(0)  # [q_len, vocab]
        next_token = logits[-1].argmax().item()
        if next_token == ans_tok:
            correct += 1
            
    mean_logprob = total_logprob / total_tokens if total_tokens > 0 else float('nan')
    accuracy = correct / n if n > 0 else float('nan')
    print(f"Mean logprob: {mean_logprob:.4f}, Accuracy: {accuracy:.4f}")
    return mean_logprob, accuracy

def train(model: GPT2, cfg: TrainingConfig, dataset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dataset)//cfg.batch_size)

    input_max = dataset.attrs["input_max"]
    wandb.init(project="add_thoughtful", name=f"normal_{input_max}", config=cfg)
    wandb.watch(model, log="all")

    for b in (tr:=tqdm.trange(0, len(dataset), cfg.batch_size, ncols=100)):
        q_toks = t.tensor(np.stack(dataset.iloc[b:b+cfg.batch_size]['question_toks']))
        ans_toks = t.tensor(dataset.iloc[b:b+cfg.batch_size]['answer_tok'].to_numpy())
        
        batch_indices = t.arange(len(ans_toks), requires_grad=False)

        logits = model.forward(q_toks).squeeze()
        logprobs = t.log_softmax(logits, dim=-1)

        pred_logprobs = logprobs[batch_indices, -1, ans_toks]
        loss = -pred_logprobs.mean()
        loss.backward()
        opt.step()
        scheduler.step()
        opt.zero_grad()

        if b != 0 and b % 10_000 == 0:
            wandb.log({"loss": loss.detach().item()})
            tr.set_description(f"{magenta}loss: {loss.detach().item():.3f}")
            t.save(model.state_dict(), f"saves/add_normal{b}.pth")

INPUT_MAX = 100
NUM_EXAMPLES = 1_000_000

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    
    model_cfg = ModelConfig(d_model=32, seq_len=32, d_mlp=128, d_head=16, n_heads=4, n_layers=2, d_vocab=INPUT_MAX)
    training_cfg = TrainingConfig(batch_size=16, lr=1e-3, weight_decay=1e-3, adam_beta1=0.9, adam_beta2=0.95)
    model = GPT2(model_cfg)

    simple_tokenizer = SimpleTokenizer(max_int=INPUT_MAX)
    trainset, testset = makeAdditionDataset(simple_tokenizer, INPUT_MAX, NUM_EXAMPLES, train_split=0.99)

    train(model, training_cfg, trainset)
    benchmark_addition(model, testset)