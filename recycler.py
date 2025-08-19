import time
import eindex
import wandb
import tqdm
import datasets
from typing import Callable

from utils import *
from models import Recycler, RecycleModelConfig, TrainingConfig
from contextlib import nullcontext


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
    for b, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
        tokens = batch['input_ids'].to(model.embed.weight.device)
        batch_size, seq_len = tokens.shape

        with (t.autocast(device_type="cuda", dtype=t.bfloat16) if cfg.bf16 else nullcontext()):
            context_parts: list[t.Tensor] = []
            logit_parts: list[t.Tensor] = []
            #for s in range(seq_len): # this one is for non-interleaved embedding approaches
                #next_toks = tokens[:, s].reshape(batch_size)
                #cur_toks = tokens[:, :s+1]
                #context = t.cat(context_parts, dim=1) if s != 0 else None
                ##new_ctx, new_logits = model.forward_replace_embed(next_toks, context)
                #new_ctx, new_logits = model.forward_attn_gate(cur_toks, context)
                #context_parts.append(new_ctx.unsqueeze(1))
                #logit_parts.append(new_logits.unsqueeze(1))

            for s in range(seq_len): # for interleaved embedding approaches
                next_toks = tokens[:, s].reshape(batch_size)
                cur_toks = tokens[:, :s+1]
                context = t.cat(context_parts, dim=1) if s > 0 else None
                new_ctx, new_logits = model.forward_interleaved_embeddings(next_toks, context, emb_dropout=0.5)
                #new_ctx, new_logits = model.forward_attn_gate_interleaved(cur_toks, context)
                #new_ctx, new_logits = model.forward_recycler_block_interleaved(next_toks, context)
                logit_parts.append(new_logits.unsqueeze(1))
                
                tok_embeds = (model.embed(next_toks) + model.pos_embed.weight[s]).reshape(batch_size, d_model)
                context_parts.append(tok_embeds.unsqueeze(1))
                context_parts.append(new_ctx.unsqueeze(1))
            
            if b%32 == 0:
                with t.inference_mode():
                    context_parts = []
                    logit_parts = []
                    for s in range(seq_len): # for interleaved embedding approaches
                        next_toks = tokens[:, s].reshape(batch_size)
                        cur_toks = tokens[:, :s+1]
                        context = t.cat(context_parts, dim=1) if s > 0 else None
                        new_ctx, new_logits = model.forward_interleaved_embeddings(next_toks, context, emb_dropout=0.0)
                        #new_ctx, new_logits = model.forward_attn_gate_interleaved(cur_toks, context)
                        #new_ctx, new_logits = model.forward_recycler_block_interleaved(next_toks, context)
                        logit_parts.append(new_logits.unsqueeze(1))
                        
                        tok_embeds = (model.embed(next_toks) + model.pos_embed.weight[s]).reshape(batch_size, d_model)
                        context_parts.append(tok_embeds.unsqueeze(1))
                        context_parts.append(new_ctx.unsqueeze(1))

                    # calculate test loss w/o dropout
                    logits = t.cat(logit_parts, dim=1)
                    logprobs = t.log_softmax(logits, dim=-1)
                    test_loss = -logprobs[t.arange(batch_size).unsqueeze(-1), t.arange(seq_len - 1).unsqueeze(0), tokens[:, 1:]].mean()
            
            logits = t.cat(logit_parts, dim=1)
            logprobs = t.log_softmax(logits, dim=-1)
            train_loss = -logprobs[t.arange(batch_size).unsqueeze(-1), t.arange(seq_len - 1).unsqueeze(0), tokens[:, 1:]].mean()
        train_loss.backward()
        grad_norm = t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, error_if_nonfinite=True)

        optimizer.step()    
        optimizer.zero_grad()

        wandb.log({"train_loss": train_loss.item(), "grad_norm": grad_norm, "loss": test_loss.item()})
        tr.set_description(f"{magenta}train loss: {train_loss.item():.3f}, grad_norm: {grad_norm:.3f}, test loss: {test_loss.item():.3f}")


if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    seq_len = 64
    d_model = 256
    model_cfg = RecycleModelConfig(
        d_model=d_model,
        seq_len=seq_len,
        d_mlp=d_model * 4,
        n_heads=8,
        n_layers=8,
        recycle_layer=6,
        d_vocab=50_257
    )
    model = Recycler(model_cfg)
    training_cfg = TrainingConfig(
        batch_size=4,
        lr=3e-4,
        weight_decay=1e-3,
        bf16=True,
    )

    #dataset = tokenizeAndSaveDataset(model.tokenizer, seq_len, "HuggingFaceFW/fineweb-edu", "sample-10BT", f"fineweb-edu-tokenized-{seq_len}", 0.07, pad=False)
    dataset = loadTokenizedDataset(f"fineweb-edu-tokenized-{seq_len}")
    
    train(model, training_cfg, dataset)