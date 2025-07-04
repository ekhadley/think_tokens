import eindex
import wandb
import tqdm
import datasets

from utils import *
from models import GPT2, TrainingConfig, ModelConfig

def train(model, cfg: TrainingConfig, dataset: datasets.Dataset):
    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)

    model.train()

    wandb.init(project="thoughtful", name="gpt2s_normal", config=cfg)
    wandb.watch(model, log="all")
    wandb.config.update(model.cfg.to_dict())
    wandb.config.update(cfg.to_dict())

    sample_completion = model.yap("George Washington was")
    print(yellow, sample_completion, endc)
    table_data = [[sample_completion]]
    table = wandb.Table(data=table_data, columns=['completion'])
    wandb.log({"sample_completion": table})

    dl = t.utils.data.DataLoader(dataset, batch_size=cfg.batch_size)
    for i, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
        tokens = batch['input_ids']
        logits = model(tokens)
        logprobs = t.log_softmax(logits[:, :-1], dim=-1)
        loss = -eindex.eindex(logprobs, tokens[:, 1:], "batch seq [batch seq]").mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item()})
        tr.set_description(f"{magenta}loss: {loss.item():.3f}")

        if i%1_000 == 0:
            sample_completion = model.yap("George Washington was")
            print(yellow, sample_completion, endc)
            table_data.append([sample_completion])
            table = wandb.Table(data=table_data, columns=['completion'])
            wandb.log({"sample_completion": table})

            t.save(model.state_dict(), f"saves/normal{i}.pth")

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))

    model_cfg = ModelConfig(d_model=512, seq_len=256, d_mlp=2048, d_head=64, n_heads=8, n_layers=8, d_vocab=50257)
    model = GPT2(model_cfg)
    training_cfg = TrainingConfig(
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.95
    )

    #dataset = tokenizeAndSaveDataset(model.tokenizer, model_cfg, "HuggingFaceFW/fineweb-edu", "sample-10BT", f"fineweb-edu-tokenized-512", 0.07, pad=False)
    dataset = loadTokenizedDataset("fineweb-edu-tokenized-256")
    
    train(model, training_cfg, dataset)