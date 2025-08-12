import time
import eindex
import wandb
import tqdm
import datasets

from utils import *
from models import Recycler, RecycleModelConfig, TrainingConfig

# Takes vector of next tokens for the batch and produces single context output vector.
@t.inference_mode()
def test_accuracy_recycler1(model: Recycler, dataset: datasets.Dataset) -> tuple[float, float]:
    tokens = dataset['input_ids']
    batch_size, seq_len = tokens.shape
    
    ctx = t.zeros((batch_size, seq_len, d_model))
    logits = t.zeros((batch_size, seq_len, model.cfg.d_vocab))
    for s in range(seq_len):
        toks = tokens[:, s].reshape(-1, 1) # (batch, 1)
        new_ctx, new_logits = model.forward(toks, ctx[:, :s] if s != 0 else None)
        logits[:, s, :] = new_logits
        ctx[:, s, :] = new_ctx
    logprobs = t.log_softmax(logits, dim=-1)
    logprob_correct = eindex.eindex(logprobs[:, :-1], tokens[:, 1:], "batch seq [batch seq]").mean().item()

    top_moves = logits.argmax(dim=-1)
    correct = (top_moves[:, :-1] == tokens[:, 1:]).float().mean().item()
    return correct, logprob_correct

# changes and returns the whole hidden state on each forward pass.
@t.inference_mode()
def test_accuracy_recycler2(model: Recycler, dataset: datasets.Dataset) -> tuple[float, float]:
    tokens = dataset['input_ids']
    batch_size, seq_len = tokens.shape
    
    ctx = None
    logits = t.zeros((batch_size, seq_len, model.cfg.d_vocab))
    for s in range(seq_len):
        toks = tokens[:, s].reshape(-1, 1)
        ctx, new_logits = model.forward2(toks, ctx)
        logits[:, s, :] = new_logits
    logprobs = t.log_softmax(logits, dim=-1)
    logprob_correct = eindex.eindex(logprobs[:, :-1], tokens[:, 1:], "batch seq [batch seq]").mean().item()

    top_moves = logits.argmax(dim=-1)
    correct = (top_moves[:, :-1] == tokens[:, 1:]).float().mean().item()
    return correct, logprob_correct

# takes all prev and current tokens of each seq in the batch as input. Returns a single new context vector like forward1.
@t.inference_mode()
def test_accuracy_recycler3(model: Recycler, dataset: datasets.Dataset) -> tuple[float, float]:
    tokens = dataset['input_ids']
    batch_size, seq_len = tokens.shape
    
    ctx = t.zeros((batch_size, seq_len, d_model))
    logits = t.zeros((batch_size, seq_len, model.cfg.d_vocab))
    for s in range(seq_len):
        toks = tokens[:, :s+1]
        new_ctx, new_logits = model.forward3(toks, ctx[:, :s] if s != 0 else None)
        logits[:, s, :] = new_logits
        ctx[:, s, :] = new_ctx
    logprobs = t.log_softmax(logits, dim=-1)
    logprob_correct = eindex.eindex(logprobs[:, :-1], tokens[:, 1:], "batch seq [batch seq]").mean().item()

    top_moves = logits.argmax(dim=-1)
    correct = (top_moves[:, :-1] == tokens[:, 1:]).float().mean().item()
    return correct, logprob_correct

@t.inference_mode()
def test_accuracy_recycler4(model: Recycler, dataset: datasets.Dataset) -> tuple[float, float]:
    tokens = dataset['input_ids']
    batch_size, seq_len = tokens.shape
    
    ctx = t.zeros((batch_size, 2*seq_len, d_model))
    logits = t.zeros((batch_size, seq_len, model.cfg.d_vocab))
    for s in range(seq_len):
        toks = tokens[:, s].reshape(-1, 1) # (batch, 1)
        new_ctx, new_logits = model.forward4(toks, ctx[:, :s*2] if s != 0 else None)
        logits[:, s, :] = new_logits
        
        tok_embeds = model.embed(toks).reshape(batch_size, d_model)
        ctx[:, s*2, :] = tok_embeds # put the normal token embedding into the context
        ctx[:, s*2+1, :] = new_ctx # update the context with the new context vector
    
    logprobs = t.log_softmax(logits, dim=-1)
    logprob_correct = eindex.eindex(logprobs[:, :-1], tokens[:, 1:], "batch seq [batch seq]").mean().item()

    top_moves = logits.argmax(dim=-1)
    correct = (top_moves[:, :-1] == tokens[:, 1:]).float().mean().item()
    return correct, logprob_correct


def train(model: Recycler, cfg: TrainingConfig, trainset: datasets.Dataset, testset: datasets.Dataset, epochs: int = 5):
    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)

    model.train()

    run_cfg = {"model": model.cfg.to_dict(), "training": cfg.to_dict()}
    wandb.init(project="gpt_chess", name="recycler", config=run_cfg)

    batch_size = cfg.batch_size
    seq_len = dataset['input_ids'].shape[1]
    d_model = model.cfg.d_model
    d_vocab = model.cfg.d_vocab

    seq_indices = t.arange(seq_len - 1, requires_grad=False)
    grad_norm = 0

    dl = t.utils.data.DataLoader(trainset, batch_size=cfg.batch_size)
    accuracy = 0.0
    for epoch in range(epochs):
        for i, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
            tokens = batch['input_ids']
            batch_size, seq_len = tokens.shape

            #ctx = t.zeros((batch_size, seq_len, d_model))
            ##ctx = None
            #logits = t.zeros((batch_size, seq_len, d_vocab))
            #for s in range(seq_len):
                #toks = tokens[:, s].reshape(-1, 1) # (batch, 1)
                #new_ctx, new_logits = model.forward(toks, ctx[:, :s] if s != 0 else None)
                ##new_ctx, new_logits = model.forward3(tokens[:, :s+1], ctx[:, :s] if s != 0 else None)
                ##print(new_ctx)
                #ctx[:, s, :] = new_ctx # update the context with the new context vector
                ##ctx, new_logits = model.forward2(toks, ctx)
                #logits[:, s, :] = new_logits

            ctx = t.zeros((batch_size, 2*seq_len, d_model))
            logits = t.zeros((batch_size, seq_len, d_vocab))
            for s in range(seq_len):
                toks = tokens[:, s].reshape(-1, 1) # (batch, 1)
                new_ctx, new_logits = model.forward4(toks, ctx[:, :s*2] if s != 0 else None)
                logits[:, s, :] = new_logits
                
                tok_embeds = model.embed(toks).reshape(batch_size, d_model)
                ctx[:, s*2, :] = tok_embeds # put the normal token embedding into the context
                ctx[:, s*2+1, :] = new_ctx # update the context with the new context vector
            
            logprobs = t.log_softmax(logits, dim=-1)
            #loss = -eindex.eindex(logprobs[:, :-1], tokens[:, 1:], "batch seq [batch seq]").mean()
            loss = -logprobs[t.arange(batch_size).unsqueeze(-1), seq_indices.unsqueeze(0), tokens[:, 1:]].mean()
            loss.backward()
            grad_norm = t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, error_if_nonfinite=True)

            if i % 32 == 0:
                accuracy, _ = test_accuracy_recycler4(model, testset)
                #t.save(model.state_dict(), f"saves/chess_normal{i}.pth")
            
            optimizer.step()    
            optimizer.zero_grad()

            wandb.log({"loss": loss.item(), "acc": accuracy, "grad_norm": grad_norm})
            tr.set_description(f"{magenta}loss: {loss.item():.3f}, acc: {accuracy:.3f}, grad_norm: {grad_norm:.3f}")


if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    d_model = 32
    model_cfg = RecycleModelConfig(
        d_model=d_model,
        seq_len=256,
        d_mlp=d_model*4,
        n_heads=4,
        n_layers=6,
        recycle_layer=3,
        d_vocab=64,
    )
    model = Recycler(model_cfg)

    training_cfg = TrainingConfig(
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-9,
    )

    dataset = datasets.load_dataset(f"eekay/chess-games-40moves-3min")["train"]
    dataset.set_format(type='torch')
    trainset, testset = dataset.train_test_split(test_size=0.01).values()
    
    train(model, training_cfg, trainset, testset, epochs=10)