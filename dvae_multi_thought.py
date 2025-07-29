import eindex
import wandb
import tqdm
import datasets

from utils import *
from models import GPT2SplitModel, TrainingConfig, SplitModelConfig
from tae import test_acc


def train(answer_model: GPT2SplitModel, think_model: GPT2SplitModel, cfg: TrainingConfig, steps: int = 1e9):
    answer_params, think_params = list(answer_model.parameters()), list(think_model.parameters())
    opt = t.optim.AdamW(answer_params + think_params, lr=cfg.answer_lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    answer_model.train()
    think_model.train()

    d_thought = think_model.cfg.d_vocab_out
    d_think_embed = think_model.cfg.d_vocab_in
    inp_max = answer_model.cfg.d_vocab_out

    run_cfg = {"answer_model": answer_model.cfg.to_dict(), "think_model": think_model.cfg.to_dict(), "training": cfg.to_dict()}
    wandb.init(project="tokenized_autoencoder", config=run_cfg, name=f"gumbel{inp_max}_{d_thought}x{cfg.think_len}")

    think_grad_norm, ans_grad_norm = 0, 0

    full_batch_size = cfg.group_size * cfg.batch_size
    #max_seq_len = think_model.cfg.seq_len - 1 - cfg.think_len
    full_batch_indices = t.arange(full_batch_size, requires_grad=False)
    
    pred_acc = 0.0

    
    for b in (tr:=tqdm.trange(int(steps), ncols=140)):
        inp_toks = t.randint(0, inp_max, (cfg.batch_size, 1), requires_grad=False).reshape(-1, 1).repeat(1, cfg.group_size).reshape(full_batch_size, 1)
        #print()
        #print(red, inp_toks, endc)

        inp_toks_one_hot = t.nn.functional.one_hot(inp_toks, num_classes=d_think_embed).float()
        rollout_one_hot = inp_toks_one_hot
        for i_t in range(cfg.think_len):
            #print(pink, think_model.embed.weight[:, :8], endc)
            #print(pink, think_model.embed.weight.shape, endc)
            #print(gray, rollout_one_hot, endc)
            #print(gray, rollout_one_hot.shape, endc)
            rollout_embed = (rollout_one_hot @ think_model.embed.weight)
            #print(cyan, rollout_embed[:, :, :8], endc)
            #print(cyan, rollout_embed.shape, endc)
            think_logits = think_model.forward_embeddings(rollout_embed)
            #print(blue, think_logits, endc)
            #print(blue, think_logits.shape, endc)
            think_toks_one_hot = t.nn.functional.gumbel_softmax(think_logits[:, -1], hard=True, dim=-1)
            #print(green, think_toks_one_hot, endc)
            #print(green, think_toks_one_hot.shape, endc)

            think_toks_padded = t.nn.functional.pad(think_toks_one_hot, (inp_max, 0), value=0.0)
            rollout_one_hot = t.cat([rollout_one_hot, think_toks_padded.unsqueeze(1)], dim=1)

        #print(gray, rollout_one_hot.shape, endc)
        rollout_thoughts_one_hot = rollout_one_hot[:, 1:, -d_thought:]
        #print(lime, rollout_thoughts_one_hot.shape, endc)
        rollout_thoughts_embed = (rollout_thoughts_one_hot @ answer_model.embed.weight)
        #print(cyan, rollout_thoughts_embed[:, :, :8], endc)
        #print(cyan, rollout_thoughts_embed.shape, endc)
        answer_logits = answer_model.forward_embeddings(rollout_thoughts_embed)
        answer_logprobs = t.log_softmax(answer_logits[:, -1], dim=-1)
        losses = -answer_logprobs[full_batch_indices, inp_toks.squeeze()]
        loss = losses.mean()

        loss.backward()
        ans_grad_norm = t.norm(t.stack([t.norm(p.grad.detach(), 2.0) for p in answer_params]), 2.0).item()
        think_grad_norm = t.norm(t.stack([t.norm(p.grad.detach(), 2.0) for p in think_params]), 2.0).item()
        opt.step()
        opt.zero_grad()
        
        with t.inference_mode():
            wandb.log({
                "pred_loss": loss.item(),
                "acc": pred_acc,
                "think_grad_norm": think_grad_norm,
                "ans_grad_norm": ans_grad_norm,
            })
            tr.set_description(f"{magenta}loss: {loss:.3f}, acc: {pred_acc:.3f}")

            if b % 512 == 0:
                print()
                rollouts = rollout_one_hot.argmax(dim=-1)
                action_logprobs = think_logits.log_softmax(dim=-1)[full_batch_indices.unsqueeze(-1), t.arange(cfg.think_len).unsqueeze(0), rollouts[:, 1:].squeeze() - inp_max].mean(dim=-1)
                preds = answer_logprobs.argmax(dim=-1).squeeze()
                for row in range(rollouts.shape[0]):
                    print(f"{blue}{rollouts[row].tolist()} {magenta}{action_logprobs[row].item():.3f} : {cyan}{-losses[row].item():.3f} {gray}{preds[row].item()} ({answer_logprobs[row, preds[row]]:.3f}) {endc}")

                #pred_acc = (pred_logprobs.argmax(dim=-1) == inp_toks.flatten()).float().mean().item()
                logprob_correct, pred_acc = test_acc(answer_model, think_model, cfg, inp_max)

                #t.save(answer_model.state_dict(), f"saves/add_think_fixed_blind_super_clean_split_answer{b}.pth")
                #t.save(think_model.state_dict(), f"saves/add_think_fixed_blind_super_clean_split_think{b}.pth")



INP_MAX = 64
if __name__ == "__main__":
    t.set_default_device(t.device("cpu"))
    t.manual_seed(42)
    random.seed(42)
    t.autograd.set_detect_anomaly(True)

    d_model = 64
    d_thought = 64
    think_len = 1
    answer_model_cfg = SplitModelConfig(
        d_model=d_model,
        seq_len=think_len,
        d_mlp=d_model*4,
        d_head=d_model//4,
        n_heads=4,
        n_layers=0,
        d_vocab_in=d_thought,
        d_vocab_out=INP_MAX,
        d_thought_vocab=d_thought
    )
    think_model_cfg = SplitModelConfig(
        d_model=d_model,
        seq_len=think_len*2,
        d_mlp=d_model*4,
        d_head=d_model//4,
        n_heads=4,
        n_layers=0,
        d_vocab_in=INP_MAX + d_thought,
        d_vocab_out=d_thought,
        d_thought_vocab=d_thought
    )

    answer_model = GPT2SplitModel(answer_model_cfg)
    think_model = GPT2SplitModel(think_model_cfg)

    training_cfg = TrainingConfig(
        think_lr=1e-4,
        answer_lr=1e-4,
        think_len=think_len,
        group_size=16,
        batch_size=64,
        weight_decay=1e-9
    )

    train(answer_model, think_model, training_cfg, steps=1e5)