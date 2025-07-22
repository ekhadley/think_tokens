import eindex
import wandb
import tqdm
import datasets

from utils import *
from models import GPT2, TrainingConfig, ModelConfig

@t.inference_mode()
def test_accuracy(model: GPT2, dataset: datasets.Dataset) -> tuple[float, float]:
    tokens = dataset['input_ids']
    logits = model(tokens)
    logprob_correct = t.log_softmax(logits[:, :-1], dim=-1)
    logprob_correct = eindex.eindex(logprob_correct, tokens[:, 1:], "batch seq [batch seq]").mean().item()

    top_moves = logits.argmax(dim=-1)
    correct = (top_moves[:, :-1] == tokens[:, 1:]).float().mean().item()
    return correct, logprob_correct
    

def train(model, cfg: TrainingConfig, trainset: datasets.Dataset, testset: datasets.Dataset, epochs: int = 5):
    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)

    model.train()

    wandb.init(project="gpt_chess", name="normal", config=cfg)
    run_cfg = {"model": model.cfg.to_dict(), "training": cfg.to_dict()}
    wandb.config.update(run_cfg)

    parameters = [p for p in model.parameters() if p.requires_grad]
    grad_norm = 42069

    dl = t.utils.data.DataLoader(trainset, batch_size=cfg.batch_size)
    accuracy = 0.0
    for epoch in range(epochs):
        for i, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
            tokens = batch['input_ids']
            logits = model(tokens)
            logprobs = t.log_softmax(logits[:, :-1], dim=-1)
            loss = -eindex.eindex(logprobs, tokens[:, 1:], "batch seq [batch seq]").mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            grad_norm = t.norm(t.stack([t.norm(p.grad.detach(), 2.0) for p in parameters]), 2.0).item()

            wandb.log({"loss": loss.item(), "acc": accuracy, "grad_norm": grad_norm})
            tr.set_description(f"{magenta}loss: {loss.item():.3f}, acc: {accuracy:.3f}, grad_norm: {grad_norm:.3f}")

            if i % 32 == 0:
                accuracy, _ = test_accuracy(model, testset)
                #t.save(model.state_dict(), f"saves/chess_normal{i}.pth")


def sweep(trainset: pd.DataFrame, count: int, epochs: int = 5):
    def run_training():
        t.set_default_device(t.device("cuda"))
        t.manual_seed(42)
        random.seed(42)
        wandb.init()

        w_config = wandb.config
        model_cfg = ModelConfig(
            d_model=w_config.d_model,
            seq_len=128,
            d_mlp=w_config.d_model*4,
            d_head=w_config.d_model//8,
            n_heads=8,
            n_layers=w_config.n_layers,
            d_vocab=64
        )
        model = GPT2(model_cfg)

        training_cfg = TrainingConfig(batch_size=w_config.batch_size,lr=w_config.lr,weight_decay=w_config.weight_decay)
        wandb.config.update(model.cfg.to_dict())
        wandb.watch(model, log="all")

        train(model, training_cfg, trainset, epochs=epochs)

    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'loss',
            'goal': 'minimize'
        },
        'parameters': {
            'batch_size': {
                'values': [32, 64, 128, 256]
            },
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
            'd_model': {
                'values': [32, 64, 128, 256, 512]
            },
            'n_layers': {
                'values': [2, 4, 6, 8, 12]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="chess-normal-sweep")
    wandb.agent(sweep_id, function=run_training, count=count)

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    d_model = 512
    model_cfg = ModelConfig(
        d_model=d_model,
        seq_len=128,
        d_mlp=d_model*4,
        d_head=d_model//8,
        n_heads=8,
        n_layers=8,
        d_vocab=64,
    )
    model = GPT2(model_cfg)

    training_cfg = TrainingConfig(
        batch_size=128,
        lr=6e-4,
        weight_decay=1e-6,
    )

    dataset = datasets.load_from_disk("./datasets/chess_40moves_3min_hf")
    dataset.set_format(type='torch')
    trainset, testset = dataset.train_test_split(test_size=0.01).values()
    
    train(model, training_cfg, trainset, testset, epochs=5)
    #sweep(dataset, count=128, epochs=1)