import tqdm
import datasets
import wandb
import torch as t
from torch import nn
from transformers import GPT2TokenizerFast, AutoTokenizer
import random
import pandas as pd
from eindex import eindex
from normal import GPT2
from utils import *



def makeAdditionDataset(tokenizer, int_max, n_questions, name: str):
    question_str = []
    answer_str = []
    question_toks = []
    answer_toks = []
    question_str_toks = []
    question_len = []
    answer_len = []
    seq_lens = []
    full_seqs = []
    for i in tqdm.trange(n_questions):
        n1 = random.randint(0, int_max)
        n2 = random.randint(0, int_max)
        question = f"{n1} + {n2} = "
        answer = str(n1 + n2)
        toks = np.array(tokenizer(question).input_ids)
        ans_toks = np.array(tokenizer(answer).input_ids)
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

    dataset = pd.DataFrame({
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
    dataset.to_pickle(f"datasets/{name}.pkl")
    return dataset


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
    for i, row in tqdm.tqdm(enumerate(dataset.itertuples()), total=n, desc="Benchmark"):
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

    eot_id = model.cfg.d_vocab - 1

    n_batches = len(dataset) // cfg.batch_size
    for b in (tr:=tqdm.trange(n_batches, ncols=100)):
        row = dataset.iloc[b]
        seq = t.tensor(row["full_seq"])
        logits = model(seq).squeeze()
        logprobs = t.log_softmax(logits, dim=-1)
        q_len = row["question_len"]
        pred_logprobs = logprobs[t.arange(q_len-1, seq.shape[0] - 1), seq[q_len:]]

        loss = -pred_logprobs.mean() / cfg.batch_size
        loss.backward()
        if b != 0 and b % cfg.batch_size == 0:
            opt.step()
            opt.zero_grad()
            tr.set_description(f"{magenta}loss: {cfg.batch_size * loss.detach().item():.3f}")

t.backends.cuda.enable_flash_sdp(enabled=True)
t.set_default_device(t.device("cuda"))
if __name__ == "__main__":
    model_cfg = ModelConfig(d_model=512, seq_len=128, d_mlp=2048, d_head=64, n_heads=8, n_layers=8, d_vocab=50_257)
    model = GPT2(model_cfg)
    training_cfg = TrainingConfig(gamma=0.95, batch_size=16, lr=1e-3, weight_decay=1e-3, adam_beta1=0.9, adam_beta2=0.95)

    #dataset = makeAdditionDataset(model.tokenizer, 10_000, 1_000, "additions_10K_1K")
    #dataset = makeAdditionDataset(model.tokenizer, 10_000, 1_000_000, "additions_10K_1M")
    #dataset = makeAdditionDataset(model.tokenizer, 100, 1_000_000, "additions_100_1M")
    #dataset = makeAdditionDataset(model.tokenizer, 10_000, 100_000, "additions_10K_100K")
    dataset = pd.read_pickle("datasets/additions_100_1M.pkl")

    train(model, training_cfg, dataset)

    bench_dataset = dataset.sample(1000)
    benchmark_addition(model, bench_dataset)