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

# Custom tokenizer for digits 0-9, '+', '='

class SimpleTokenizer:
    def __init__(self):
        self.vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=']
        self.token_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.id_to_token = {i: ch for i, ch in enumerate(self.vocab)}
    def encode(self, s):
        return [self.token_to_id[ch] for ch in s]
    def decode(self, ids):
        return ''.join([self.id_to_token[i] for i in ids])

# Use this tokenizer everywhere
simple_tokenizer = SimpleTokenizer()

def makeAdditionDataset(tokenizer, int_max, n_questions, name: str, train_split: float = 1.0):
    question_str = []
    answer_str = []
    question_toks = []
    answer_toks = []
    question_str_toks = []
    question_len = []
    answer_len = []
    seq_lens = []
    full_seqs = []

    # Calculate all possible unique questions
    all_questions = [f"{n1}+{n2}=" for n1 in range(int_max+1) for n2 in range(int_max+1)]
    random.shuffle(all_questions)
    n_unique = len(all_questions)
    crowded = n_questions > n_unique
    if crowded:
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
            n1, n2 = map(int, question[:-1].split('+'))
            answer = str(n1 + n2)
            toks = np.array(tokenizer.encode(question))
            ans_toks = np.array(tokenizer.encode(answer))
            full_seq = np.concatenate([toks, ans_toks])
            question_str.append(question)
            answer_str.append(answer)
            question_toks.append(toks)
            answer_toks.append(ans_toks)
            question_str_toks.append(tokenizer.decode(toks) + tokenizer.decode(ans_toks))
            question_len.append(len(toks))
            answer_len.append(len(ans_toks))
            full_seqs.append(full_seq)
            seq_lens.append(len(full_seq))

    # Add training and test examples
    add_examples(train_questions)
    train_dataset = pd.DataFrame({
        "question": question_str,
        "answer": answer_str,
        "question_toks": question_toks,
        "answer_toks": answer_toks,
        "question_str_toks": question_str_toks,
        "question_len": question_len,
        "answer_len": answer_len,
        "full_seq": full_seqs,
        "seq_len": seq_lens
    })
    if n_test > 0:
        # Clear lists and add test examples
        question_str.clear(); answer_str.clear(); question_toks.clear(); answer_toks.clear()
        question_str_toks.clear(); question_len.clear(); answer_len.clear(); full_seqs.clear(); seq_lens.clear()
        add_examples(test_questions)
        test_dataset = pd.DataFrame({
            "question": question_str,
            "answer": answer_str,
            "question_toks": question_toks,
            "answer_toks": answer_toks,
            "question_str_toks": question_str_toks,
            "question_len": question_len,
            "answer_len": answer_len,
            "full_seq": full_seqs,
            "seq_len": seq_lens
        })
        # Do not save train and test to their own files
        return train_dataset, test_dataset
    else:
        train_dataset.to_pickle(f"datasets/{name}.pkl")
        return train_dataset


def benchmark_addition(model: GPT2, dataset: pd.DataFrame, max_answer_len: int = 10):
    """
    Benchmarks the model's addition ability on a dataset.
    Returns:
        - mean_logprob: mean aggregate logprob over correct answer tokens
        - accuracy: fraction of exact matches using argmax sampling
    """
    model.eval()
    total_logprob = 0.0
    total_tokens = 0
    correct = 0
    n = len(dataset)
    for i, row in tqdm.tqdm(enumerate(dataset.itertuples()), total=n, desc="Benchmark", ncols=100):
        seq = t.tensor(row.full_seq).to(model.parameters().__next__().device)
        q_len = row.question_len
        a_len = row.answer_len
        # Get logits for the full sequence
        with t.no_grad():
            logits = model(seq).squeeze(0)  # [seq_len, vocab]
            logprobs = t.log_softmax(logits, dim=-1)
        # Aggregate logprob for correct answer tokens
        answer_toks = seq[q_len: q_len + a_len]
        answer_logprobs = logprobs[q_len - 1 : q_len + a_len - 1, :]
        # logprob for each answer token, conditioned on previous
        token_logprobs = answer_logprobs.gather(1, answer_toks.unsqueeze(1)).squeeze(1)
        total_logprob += token_logprobs.sum().item()
        total_tokens += a_len
        # Argmax sampling
        generated = []
        input_seq = seq[:q_len].unsqueeze(0)  # [1, q_len]
        for _ in range(a_len):
            with t.no_grad():
                logits = model(input_seq).squeeze(0)  # [cur_len, vocab]
            next_token = logits[-1].argmax().item()
            generated.append(next_token)
            input_seq = t.cat([input_seq, t.tensor([[next_token]], device=input_seq.device)], dim=1)
        # Compare generated answer to true answer
        if generated == answer_toks.cpu().tolist():
            correct += 1
    mean_logprob = total_logprob / total_tokens if total_tokens > 0 else float('nan')
    accuracy = correct / n if n > 0 else float('nan')
    print(f"Mean logprob: {mean_logprob:.4f}, Accuracy: {accuracy:.4f}")
    return mean_logprob, accuracy

def train(model: GPT2, cfg: TrainingConfig, dataset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)

    wandb.init(project="add_thoughtful", name="normal", config=cfg)
    wandb.watch(model, log="all")

    for b in (tr:=tqdm.trange(len(dataset), ncols=100)):
        row = dataset.iloc[b]
        seq = t.tensor(row["full_seq"])
        logits = model.forward(seq).squeeze()
        logprobs = t.log_softmax(logits, dim=-1)
        q_len = row["question_len"]
        pred_logprobs = logprobs[t.arange(q_len-1, seq.shape[0] - 1), seq[q_len:]]

        loss = -pred_logprobs.mean()
        loss.backward()
        if b != 0 and b % cfg.batch_size == 0:
            opt.step()
            opt.zero_grad()
            wandb.log({"loss": loss.detach().item()})
            tr.set_description(f"{magenta}loss: {loss.detach().item():.3f}")


if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))

    model_cfg = ModelConfig(d_model=64, seq_len=128, d_mlp=512, d_head=16, n_heads=4, n_layers=4, d_vocab=12)
    #model_cfg = ModelConfig(d_model=512, seq_len=128, d_mlp=2048, d_head=64, n_heads=8, n_layers=8, d_vocab=12)
    model = GPT2(model_cfg)
    training_cfg = TrainingConfig(gamma=0.95, batch_size=16, lr=3e-4, weight_decay=1e-6, adam_beta1=0.9, adam_beta2=0.95)

    trainset, testset = makeAdditionDataset(simple_tokenizer, 100, 10_000, "additions_1K_100K", train_split=0.9)
    #dataset = pd.read_pickle("datasets/simple_additions_10K_100K.pkl")

    train(model, training_cfg, trainset)
    benchmark_addition(model, testset)