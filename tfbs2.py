import eindex
import wandb
import tqdm
import datasets

from utils import *
from models import GPT2SplitModel, TrainingConfig, SplitModelConfig

@t.inference_mode()
def benchmark_acc(answer_model: GPT2SplitModel, think_model: GPT2SplitModel, cfg: TrainingConfig, dataset: datasets.Dataset) -> tuple[float, float]:
    locs, accs = [], []

    toks = dataset['input_ids']
    batch_indices = t.arange(toks.shape[0])
    #for seq_len in range(1, toks.shape[1]):
    seq_len = 22
    rollouts = toks[:, :seq_len]
    ans_toks = toks[:, seq_len].squeeze()
    for i_t in range(cfg.think_len):
        think_logits = think_model(rollouts)
        #print(red, think_logits.shape, endc)
        think_toks = think_logits[:, -1].argmax(dim=-1, keepdim=True) + answer_model.cfg.d_vocab_out
        #print(blue, think_toks.squeeze(), endc)
        #print(blue, think_toks.shape, endc)
        rollouts = t.cat([rollouts, think_toks], dim=1)
        #print(green, rollouts, endc)
        #print(green, rollouts.shape, endc)

    rollout_thoughts = rollouts[:, -cfg.think_len:] - answer_model.cfg.d_vocab_out
    ans_logits = answer_model(rollout_thoughts)
    ans_logprobs = t.log_softmax(ans_logits[:, -1], dim=-1)
    loc = ans_logprobs[batch_indices, ans_toks].mean().item()
    locs.append(loc)

    acc = (ans_logprobs.argmax(dim=-1).squeeze() == ans_toks).float().mean().item()
    accs.append(acc)

    mean_loc = sum(locs) / len(locs)
    mean_acc = sum(accs) / len(accs)

    return mean_loc, mean_acc

def train(answer_model: GPT2SplitModel, think_model: GPT2SplitModel, cfg: TrainingConfig, trainset: datasets.Dataset, testset: datasets.Dataset, epochs: int = 5):
    answer_params, think_params = list(answer_model.parameters()), list(think_model.parameters())
    opt = t.optim.AdamW(answer_params + think_params, lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    answer_model.train()
    think_model.train()

    run_cfg = {"answer_model": answer_model.cfg.to_dict(), "think_model": think_model.cfg.to_dict(), "training": cfg.to_dict()}
    wandb.init(project="chess2", name="think", config=run_cfg)

    pred_acc = 0.0
    ans_grad_norm, think_grad_norm = 0, 0

    max_seq_len = trainset['input_ids'].shape[1] - 1 - cfg.think_len

    d_normal_vocab, d_thought_vocab = answer_model.cfg.d_vocab_out, think_model.cfg.d_vocab_out

    #group_size = cfg.group_size
    #full_batch_size = group_size * cfg.batch_size
    #batch_indices = t.arange(cfg.batch_size, requires_grad=False)
    #full_batch_indices = t.arange(full_batch_size, requires_grad=False)

    seq_len = 22

    dl = t.utils.data.DataLoader(trainset, batch_size=cfg.batch_size)
    for _ in range(epochs):
        for b, batch in enumerate(tr:=tqdm.tqdm(dl, ncols=140)):
            #for seq_len in range(1, max_seq_len):
            seqs: t.Tensor = batch['input_ids'][:, :seq_len]
            batch_size, _  = seqs.shape
            full_batch_size = batch_size * cfg.group_size

            seqs = seqs.unsqueeze(0).repeat(1, 1, cfg.group_size).reshape(full_batch_size, -1)
            ans_toks = batch['input_ids'][t.arange(batch_size), seq_len].reshape(-1, 1).repeat(1, cfg.group_size).flatten()

            rollouts_one_hot = t.nn.functional.one_hot(seqs, num_classes=d_normal_vocab + d_thought_vocab).float()
            for i_t in range(cfg.think_len):
                think_logits = think_model.forward_one_hot(rollouts_one_hot)
                think_toks_one_hot = t.nn.functional.gumbel_softmax(think_logits[:, -1], hard=True, dim=-1)
                think_toks_padded = t.nn.functional.pad(think_toks_one_hot, (d_normal_vocab, 0), value=0.0)
                rollouts_one_hot = t.cat([rollouts_one_hot, think_toks_padded.unsqueeze(1)], dim=1)

            rollout_thoughts_one_hot = rollouts_one_hot[:, seq_len:, -d_thought_vocab:]

            ans_logits = answer_model.forward_one_hot(rollout_thoughts_one_hot)
            ans_logprobs = t.log_softmax(ans_logits[:, -1], dim=-1)
            losses = -ans_logprobs[t.arange(full_batch_size), ans_toks]
            loss = losses.mean()

            loss.backward()

            with t.inference_mode():
                if b % 32 == 0:
                    _, pred_acc = benchmark_acc(answer_model, think_model, cfg, testset)
                    ans_grad_norm = t.norm(t.stack([t.norm(p.grad.detach(), 2.0) for p in answer_params]), 2.0).item()
                    think_grad_norm = t.norm(t.stack([t.norm(p.grad.detach(), 2.0) for p in think_params]), 2.0).item()

                wandb.log({
                    "loss": loss.item(),
                    "acc": pred_acc,
                    "think_grad_norm": think_grad_norm,
                    "ans_grad_norm": ans_grad_norm,
                })
                tr.set_description(f"{magenta}loss: {loss:.3f}, acc: {pred_acc:.3f}")

            opt.step()
            opt.zero_grad()



if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    d_model = 64
    d_vocab = 64
    d_thought_vocab = 64
    think_len = 16
    answer_model_cfg = SplitModelConfig(
        d_model=d_model,
        seq_len=think_len,
        d_mlp=d_model*4,
        d_head=16,
        n_heads=2,
        n_layers=1,
        d_vocab_in=d_thought_vocab,
        d_vocab_out=d_vocab,
        d_thought_vocab=d_thought_vocab
    )
    think_model_cfg = SplitModelConfig(
        d_model=d_model,
        seq_len=256,
        d_mlp=d_model*4,
        d_head=16,
        n_heads=4,
        n_layers=4,
        d_vocab_in=d_vocab + d_thought_vocab,
        d_vocab_out=d_thought_vocab,
        d_thought_vocab=d_thought_vocab
    )

    answer_model = GPT2SplitModel(answer_model_cfg)
    think_model = GPT2SplitModel(think_model_cfg)
    #tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")

    training_cfg = TrainingConfig(
        think_lr=3e-4,
        answer_lr=3e-4,
        think_len=think_len,
        batch_size=32,
        group_size=64,
    )

    dataset = datasets.load_dataset(f"eekay/chess-games-40moves-3min")["train"]
    dataset.set_format(type='torch')
    trainset, testset = dataset.train_test_split(test_size=0.005).values()

    train(answer_model, think_model, training_cfg, trainset, testset, epochs=10)