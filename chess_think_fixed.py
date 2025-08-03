import eindex
import wandb
import tqdm
import datasets

from utils import *
from models import GPT2Thinking, TrainingConfig, ThinkingModelConfig

@t.inference_mode()
def benchmark_acc(model: GPT2Thinking, cfg: TrainingConfig, dataset: datasets.Dataset, seq_len:int) -> tuple[float, float]:
    locs, accs = [], []

    toks = dataset['input_ids']
    batch_indices = t.arange(toks.shape[0])
    rollouts = toks[:, :seq_len]
    ans_toks = toks[:, seq_len].squeeze()
    for i_t in range(cfg.think_len):
        think_logits = model(rollouts)
        #print(red, think_logits.shape, endc)
        think_toks = think_logits[:, -1, model.cfg.d_normal_vocab:].argmax(dim=-1, keepdim=True) + model.cfg.d_normal_vocab
        #print(blue, think_toks.squeeze(), endc)
        #print(blue, think_toks.shape, endc)
        rollouts = t.cat([rollouts, think_toks], dim=1)
        #print(green, rollouts, endc)
        #print(green, rollouts.shape, endc)

    ans_logits = model(rollouts)
    ans_logprobs = t.log_softmax(ans_logits[:, -1], dim=-1)
    loc = ans_logprobs[batch_indices, ans_toks].mean().item()
    locs.append(loc)

    acc = (ans_logprobs.argmax(dim=-1).squeeze() == ans_toks).float().mean().item()
    accs.append(acc)

    mean_loc = sum(locs) / len(locs)
    mean_acc = sum(accs) / len(accs)

    return mean_loc, mean_acc

def train(model: GPT2Thinking, cfg: TrainingConfig, trainset: datasets.Dataset, testset: datasets.Dataset, epochs: int = 5):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    model.train()

    seq_len = 22

    run_cfg = {"model": model.cfg.to_dict(), "training": cfg.to_dict()}
    wandb.init(project="chess2", name=f"sight_single{seq_len}", config=run_cfg)

    pred_acc = 0.0
    grad_norm = 0

    max_seq_len = trainset['input_ids'].shape[1] - 1 - cfg.think_len

    d_vocab, d_thought = model.cfg.d_normal_vocab, model.cfg.d_thought_vocab
    d_total_vocab = d_vocab + d_thought

    dl = t.utils.data.DataLoader(trainset, batch_size=cfg.batch_size)
    for _ in range(epochs):
        for b, batch in enumerate(tr:=tqdm.tqdm(dl, ncols=140)):
            #for seq_len in range(1, max_seq_len):
            seqs: t.Tensor = batch['input_ids'][:, :seq_len]
            batch_size, _  = seqs.shape
            full_batch_size = batch_size * cfg.group_size

            seqs = seqs.unsqueeze(0).repeat(1, 1, cfg.group_size).reshape(full_batch_size, -1)
            ans_toks = batch['input_ids'][:, seq_len].reshape(-1, 1).repeat(1, cfg.group_size).flatten()

            rollouts_one_hot = t.nn.functional.one_hot(seqs, num_classes=d_total_vocab).float()
            for _ in range(cfg.think_len):
                think_logits = model.forward_one_hot(rollouts_one_hot)
                think_logits[:, -1, :d_vocab] = -1e6
                think_toks_one_hot = t.nn.functional.gumbel_softmax(think_logits[:, -1], hard=True, dim=-1)
                rollouts_one_hot = t.cat([rollouts_one_hot, think_toks_one_hot.unsqueeze(1)], dim=1)

            ans_logits = model.forward_one_hot(rollouts_one_hot)
            ans_logprobs = t.log_softmax(ans_logits[:, -1], dim=-1)
            loss = -ans_logprobs[t.arange(full_batch_size), ans_toks].mean()

            loss.backward()

            with t.inference_mode():
                if b % 128 == 0:
                    _, pred_acc = benchmark_acc(model, cfg, testset, seq_len)
                    grad_norm = t.norm(t.stack([t.norm(p.grad.detach(), 2.0) for p in model.parameters()]), 2.0).item()

                    num_samples = 16
                    summary_len = 4
                    rollout_toks = rollouts_one_hot.argmax(dim=-1)
                    sample_indices = t.randint(full_batch_size, (num_samples,))
                    rollouts_sample = rollout_toks[sample_indices, -summary_len-cfg.think_len:]
                    ans_preds = ans_logprobs[sample_indices].argmax(dim=-1)
                    print()
                    for i, rollout in enumerate(rollouts_sample):
                        summ, thoughts = rollout[:summary_len].tolist(), rollout[-cfg.think_len:].tolist() if cfg.think_len > 0 else []
                        ans_pred = ans_preds[i].item()
                        ans_pred_logprob = ans_logprobs[sample_indices[i], ans_pred].item()
                        real_ans = ans_toks[sample_indices[i]].item()
                        print(f"{magenta}{summ} {purple}{thoughts} {green}{ans_pred} [{ans_pred_logprob:.3f}] {lime}({real_ans}) {endc}")

                wandb.log({
                    "loss": loss.item(),
                    "acc": pred_acc,
                    "ans_grad_norm": grad_norm,
                })
                tr.set_description(f"{magenta}loss: {loss:.3f}, acc: {pred_acc:.3f}")

            opt.step()
            opt.zero_grad()
    
    wandb.finish()


if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    dataset = datasets.load_dataset(f"eekay/chess-games-40moves-3min")["train"]
    dataset.set_format(type='torch')
    trainset, testset = dataset.train_test_split(test_size=0.005).values()

    d_model = 64
    d_vocab = 64
    d_thought_vocab = 64
    think_len = 32
    think_model_cfg = ThinkingModelConfig(
        d_model=d_model,
        seq_len=256,
        d_mlp=d_model*4,
        d_head=16,
        n_heads=4,
        n_layers=4,
        d_normal_vocab=d_vocab,
        d_thought_vocab=d_thought_vocab
    )

    model = GPT2Thinking(think_model_cfg)

    training_cfg = TrainingConfig(
        lr=1e-3,
        think_len=think_len,
        batch_size=32,
        group_size=8,
    )

    train(model, training_cfg, trainset, testset, epochs=5)