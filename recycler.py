import time
import eindex
import wandb
import tqdm
import datasets
from typing import Callable

from utils import *
from models import Recycler, RecycleModelConfig, TrainingConfig


def train(model: Recycler, cfg: TrainingConfig, trainset: datasets.Dataset):
    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)

    model.train()

    run_cfg = {"model": model.cfg.to_dict(), "training": cfg.to_dict()}
    wandb.init(project="recycler", name="recycler", config=run_cfg)

    batch_size = cfg.batch_size
    seq_len = trainset['input_ids'].shape[1]
    d_model = model.cfg.d_model
    d_vocab = model.cfg.d_vocab

    seq_indices = t.arange(seq_len - 1, requires_grad=False)
    grad_norm = 0

    dl = t.utils.data.DataLoader(trainset, batch_size=cfg.batch_size)
    for i, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
        tokens = batch['input_ids'].to(model.embed.weight.device)
        batch_size, seq_len = tokens.shape

        context_parts: list[t.Tensor] = []
        logit_parts: list[t.Tensor] = []
        #for s in range(seq_len): # this one is for non-interleaved embedding approaches
            #next_toks = tokens[:, s].reshape(batch_size)
            #cur_toks = tokens[:, :s+1]
            #context = t.cat(context_parts, dim=1) if s != 0 else None
            ##new_ctx, new_logits = model.forward_replace_embeddings(next_toks, context)
            #new_ctx, new_logits = model.forward_attn_gate(cur_toks, context)
            #context_parts.append(new_ctx.unsqueeze(1))
            #logit_parts.append(new_logits.unsqueeze(1))

        for s in range(seq_len): # for interleaved embedding approaches
            next_toks = tokens[:, s].reshape(batch_size)
            cur_toks = tokens[:, :s+1]
            context = t.cat(context_parts, dim=1) if s > 0 else None
            new_ctx, new_logits = model.forward_interleaved_embeddings(next_toks, context)
            #new_ctx, new_logits = model.forward_attn_gate_interleaved(cur_toks, context)
            #new_ctx, new_logits = model.forward_recycler_block_interleaved(next_toks, context)
            logit_parts.append(new_logits.unsqueeze(1))
            
            tok_embeds = model.embed(next_toks).reshape(batch_size, d_model)
            context_parts.append(tok_embeds.unsqueeze(1))
            context_parts.append(new_ctx.unsqueeze(1))
        
        logits = t.cat(logit_parts, dim=1)
        logprobs = t.log_softmax(logits, dim=-1)
        loss = -logprobs[t.arange(batch_size).unsqueeze(-1), t.arange(seq_len - 1).unsqueeze(0), tokens[:, 1:]].mean()
        loss.backward()
        grad_norm = t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, error_if_nonfinite=True)

        optimizer.step()    
        optimizer.zero_grad()

        wandb.log({"loss": loss.item(), "grad_norm": grad_norm})
        tr.set_description(f"{magenta}loss: {loss.item():.3f}, grad_norm: {grad_norm:.3f}")


if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    d_model = 16
    model_cfg = RecycleModelConfig(
        d_model=d_model,
        seq_len=256,
        d_mlp=d_model * 4,
        n_heads=4,
        n_layers=2,
        recycle_layer=1,
        d_vocab=50_257
    )
    model = Recycler(model_cfg)

    training_cfg = TrainingConfig(
        batch_size=64,
        lr=3e-3,
        weight_decay=1e-4,
    )

    #dataset = tokenizeAndSaveDataset(model.tokenizer, model_cfg, "HuggingFaceFW/fineweb-edu", "sample-10BT", f"fineweb-edu-tokenized-512", 0.07, pad=False)
    dataset = loadTokenizedDataset("fineweb-edu-tokenized-256")
    
    train(model, training_cfg, dataset)