import eindex
import wandb
import tqdm
import datasets

from utils import *
from models import GPT2, TrainingConfig, ModelConfig

def train(model: GPT2, cfg: TrainingConfig, dataset: datasets.Dataset):
    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)

    model.train()

    run_cfg = {"model": model.cfg.to_dict(), "training": cfg.to_dict()}
    wandb.init(project="recycler", name="normal", config=run_cfg)

    grad_norm = 0.0

    dl = t.utils.data.DataLoader(dataset, batch_size=cfg.batch_size)
    for i, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
        tokens = batch['input_ids'].to(model.embed.weight.device)
        batch_size, seq_len = tokens.shape

        logits = model(tokens)
        logprobs = t.log_softmax(logits[:, :-1], dim=-1)
        loss = -logprobs[t.arange(batch_size).unsqueeze(-1), t.arange(seq_len - 1).unsqueeze(0), tokens[:, 1:]].mean()
        loss.backward()
        #grad_norm = t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, error_if_nonfinite=True).item()

        optimizer.step()
        optimizer.zero_grad()

        wandb.log({"loss": loss.item(), "grad_norm": grad_norm})
        tr.set_description(f"{magenta}loss: {loss.item():.3f}, grad_norm: {grad_norm:.3f}")

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    d_model = 512
    model_cfg = ModelConfig(
        d_model=d_model,
        seq_len=256,
        d_mlp=d_model * 4,
        n_heads=8,
        n_layers=12,
        d_vocab=50_257
    )
    model = GPT2(model_cfg)
    training_cfg = TrainingConfig(
        batch_size=64,
        lr=1e-4,
        weight_decay=1e-4,
    )

    #dataset = tokenizeAndSaveDataset(model.tokenizer, model_cfg, "HuggingFaceFW/fineweb-edu", "sample-10BT", f"fineweb-edu-tokenized-512", 0.07, pad=False)
    dataset = loadTokenizedDataset("fineweb-edu-tokenized-256")
    
    train(model, training_cfg, dataset)