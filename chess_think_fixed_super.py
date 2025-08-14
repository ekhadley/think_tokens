import eindex
import wandb
import tqdm
import datasets

from utils import *
from models import GPT2Thinking, TrainingConfig, ThinkingModelConfig

@t.inference_mode()
def test_accuracy_fixed(model: GPT2Thinking, dataset: datasets.Dataset) -> tuple[float, float]:
    tokens = dataset['input_ids']
    batch_size = tokens.shape[0]
    tokens = tokens.reshape(-1, 1)
    correct_thoughts = t.cat([tokens%8, tokens//8], dim=-1) + model.cfg.d_normal_vocab
    rollouts = t.cat([correct_thoughts, tokens], dim=-1)
    rollouts = rollouts.reshape(batch_size, -1)

    logits = model(rollouts)
    logprobs = t.log_softmax(logits[:, :-1], dim=-1)
    logprob_correct = logprobs[t.arange(batch_size).unsqueeze(-1), t.arange(rollouts.shape[1] - 1).unsqueeze(0), rollouts[:, 1:]].mean().item()

    next_toks = logits.argmax(dim=-1)
    correct_token_mask = (next_toks[:, 2:-1] == rollouts[:, 3:]).float()
    correct_move_mask = correct_token_mask.reshape(batch_size, -1, 3).all(dim=-1).float()
    acc = correct_move_mask.mean()
    return acc, logprob_correct

def train(model: GPT2Thinking, cfg: TrainingConfig, trainset: datasets.Dataset, testset: datasets.Dataset, epochs: int = 5):
    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)

    model.train()

    wandb.init(project="gpt_chess", name="normal", config=cfg)
    run_cfg = {"model": model.cfg.to_dict(), "training": cfg.to_dict()}
    wandb.config.update(run_cfg)

    parameters = [p for p in model.parameters() if p.requires_grad]
    grad_norm = 42069

    seq_len = dataset['input_ids'].shape[-1]
    seq_indices = t.arange(seq_len*3 - 1, requires_grad=False)

    dl = t.utils.data.DataLoader(trainset, batch_size=cfg.batch_size)
    accuracy = 0.0
    for epoch in range(epochs):
        for i, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
            tokens = batch['input_ids']
            batch_size = tokens.shape[0]
            tokens = tokens.reshape(-1, 1)
            correct_thoughts = t.cat([tokens%8, tokens//8], dim=-1) + model.cfg.d_normal_vocab
            rollouts = t.cat([correct_thoughts, tokens], dim=-1)
            rollouts = rollouts.reshape(batch_size, -1)

            logits = model(rollouts)
            logprobs = t.log_softmax(logits[:, :-1], dim=-1)
            loss = -logprobs[t.arange(batch_size).unsqueeze(-1), seq_indices, rollouts[:, 1:]].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            grad_norm = t.norm(t.stack([t.norm(p.grad.detach(), 2.0) for p in parameters]), 2.0).item()

            wandb.log({"loss": loss.item(), "acc": accuracy, "grad_norm": grad_norm})
            tr.set_description(f"{magenta}loss: {loss.item():.3f}, acc: {accuracy:.3f}, grad_norm: {grad_norm:.3f}")

            if i % 128 == 0:
                accuracy, _ = test_accuracy_fixed(model, testset)
                #t.save(model.state_dict(), f"saves/chess_normal{i}.pth")


if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    d_model = 128
    model_cfg = ThinkingModelConfig(
        d_model=d_model,
        seq_len=256,
        d_mlp=d_model*4,
        n_heads=8,
        n_layers=6,
        d_normal_vocab=64,
        d_thought_vocab=8,
    )
    model = GPT2Thinking(model_cfg)

    training_cfg = TrainingConfig(
        batch_size=128,
        lr=6e-4,
        think_len=2,
        weight_decay=1e-6,
    )

    dataset = datasets.load_dataset(f"eekay/chess-games-40moves-3min")["train"]
    dataset.set_format(type='torch')
    trainset, testset = dataset.train_test_split(test_size=0.005).values()
    
    train(model, training_cfg, trainset, testset, epochs=10)
    #sweep(dataset, count=128, epochs=1)