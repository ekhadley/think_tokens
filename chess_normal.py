import eindex
import wandb
import tqdm
import datasets

from utils import *
from models import GPT2, TrainingConfig, ModelConfig

@t.inference_mode()
def benchmark_acc_chess(model: GPT2, dataset: datasets.Dataset) -> tuple[float, float]:
    toks = dataset['input_ids']
    seq_len = toks.shape[1]
    logits = model(toks)
    logprobs = t.log_softmax(logits[:, :-1], dim=-1)
    logprob_on_correct = logprobs[:, t.arange(seq_len - 1), toks[:, 1:]].mean().item()
    acc = (logprobs.argmax(dim=-1) == toks[:, 1:]).float().mean().item()
    return logprob_on_correct, acc

def train(model: GPT2, cfg: TrainingConfig, trainset: datasets.Dataset, testset: datasets.Dataset, epochs: int = 10):
    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    model.train()

    wandb.init(project="gpt_chess", name="normal_skinny", config=cfg)
    run_cfg = {"model": model.cfg.to_dict(), "training": cfg.to_dict()}
    wandb.config.update(run_cfg)

    pred_acc = 0.0

    seq_len = trainset['input_ids'].shape[-1]
    seq_indices = t.arange(seq_len - 1, requires_grad=False)

    for e in range(epochs):
        dl = t.utils.data.DataLoader(trainset, batch_size=cfg.batch_size)
        for i, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
            tokens = batch['input_ids']
            batch_size = tokens.shape[0]
            logits = model(tokens)
            logprobs = t.log_softmax(logits[:, :-1], dim=-1)
            loss = -logprobs[t.arange(batch_size).unsqueeze(-1), seq_indices.unsqueeze(0), tokens[:, 1:]].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item(), "acc": pred_acc})
            tr.set_description(f"{magenta}loss: {loss.item():.3f}, acc: {pred_acc:.3f}")

            if i%32 == 0:
                _, pred_acc = benchmark_acc_chess(model, testset)
                #pred_acc = logprobs.argmax(dim=-1).eq(tokens[:, xd].squeeze()).float().mean().item()

                #t.save(model.state_dict(), f"saves/normal{i}.pth")

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    d_model = 16
    model_cfg = ModelConfig(
        d_model=d_model,
        seq_len=256,
        d_mlp=d_model*4,
        n_heads=4,
        n_layers=24,
        d_vocab=64
    )
    model = GPT2(model_cfg)
    training_cfg = TrainingConfig(
        lr=3e-3,
        batch_size=64,
        weight_decay=1e-6,
    )

    dataset = datasets.load_dataset(f"eekay/chess-games-40moves-3min")["train"]
    dataset.set_format(type='torch')
    trainset, testset = dataset.train_test_split(test_size=0.005).values()
    
    train(model, training_cfg, trainset, testset, epochs=5)