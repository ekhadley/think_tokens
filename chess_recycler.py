import time
import eindex
import wandb
import tqdm
import datasets

from utils import *
from models import Recycler, RecycleModelConfig, TrainingConfig

@t.inference_mode()
def test_accuracy_recycler(model: Recycler, dataset: datasets.Dataset) -> tuple[float, float]:
    tokens = dataset['input_ids']
    batch_size, seq_len = tokens.shape
    
    #ctx = t.zeros((batch_size, seq_len, d_model)) # preaallocate context instead of cating
    ctx = None
    logits = t.zeros((batch_size, seq_len, model.cfg.d_vocab)) # preaallocate context instead of cating
    for s in range(seq_len):
        toks = tokens[:, s].reshape(-1, 1) # (batch, 1)
        #new_ctx, new_logits = model.forward(toks, ctx[:, :s] if s != 0 else None) # process the next token with the current context
        ctx, new_logits = model.forward2(toks, ctx) # process the next token with the current context
        logits[:, s, :] = new_logits
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

    parameters = [p for p in model.parameters() if p.requires_grad]
    grad_norm = 0


    dl = t.utils.data.DataLoader(trainset, batch_size=cfg.batch_size)
    accuracy = 0.0
    for epoch in range(epochs):
        for i, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
            tokens = batch['input_ids']
            batch_size, seq_len = tokens.shape
            
            ctx = t.zeros((batch_size, seq_len, d_model)) # preaallocate context instead of cating
            #ctx = None
            logits = t.zeros((batch_size, seq_len, d_vocab)) # preaallocate context instead of cating
            for s in range(seq_len):
                toks = tokens[:, s].reshape(-1, 1) # (batch, 1)
                new_ctx, new_logits = model.forward(toks, ctx[:, :s] if s != 0 else None)
                ctx[:, s, :] = new_ctx # update the context with the new context vector
                #ctx, new_logits = model.forward2(toks, ctx)
                
                logits[:, s, :] = new_logits
            logprobs = t.log_softmax(logits, dim=-1)
            loss = -eindex.eindex(logprobs[:, :-1], tokens[:, 1:], "batch seq [batch seq]").mean()
            
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()    
            grad_norm = t.norm(t.stack([t.norm(p.grad.detach(), 2.0) for p in parameters]), 2.0).item()
            optimizer.zero_grad()

            wandb.log({"loss": loss.item(), "acc": accuracy, "grad_norm": grad_norm})
            tr.set_description(f"{magenta}loss: {loss.item():.3f}, acc: {accuracy:.3f}, grad_norm: {grad_norm:.3f}")
            time.sleep(0.1)

            if i % 32 == 0:
                accuracy, _ = test_accuracy_recycler(model, testset)
                #t.save(model.state_dict(), f"saves/chess_normal{i}.pth")


if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    d_model = 64
    model_cfg = RecycleModelConfig(
        d_model=d_model,
        seq_len=128,
        d_mlp=d_model*4,
        d_head=16,
        n_heads=4,
        n_layers=4,
        d_vocab=64,
        recycle_layer=3
    )
    model = Recycler(model_cfg)

    training_cfg = TrainingConfig(
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-6,
    )

    dataset = datasets.load_dataset(f"eekay/chess-games-40moves-3min")["train"]
    dataset.set_format(type='torch')
    trainset, testset = dataset.train_test_split(test_size=0.01).values()
    
    train(model, training_cfg, trainset, testset, epochs=10)