import tqdm
import datasets
import wandb
import torch as t
from torch import nn
from transformers import GPT2TokenizerFast, AutoTokenizer
import random
from eindex import eindex

from utils import *


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.linear1 = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        if x.ndim == 2: x = x.unsqueeze(0)
        seq_len = x.shape[1]
        attn_mask = t.triu(t.ones((seq_len, seq_len)), diagonal=1).bool()
        attn_output, _ = self.attn(x, x, x, is_causal=True, attn_mask=attn_mask)
        x = self.norm1(x + attn_output)
        ff_output = self.linear2(self.act(self.linear1(x)))
        x = self.norm2(x + ff_output)
        return x

class GPT2Thinking(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(GPT2Thinking, self).__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.embed = nn.Embedding(cfg.d_vocab_total, cfg.d_model)
        self.pos_embed = nn.Linear(cfg.d_model, cfg.seq_len, bias=False)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab_total, bias=False)
        self.eot = 50256

        self.tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
    def encode(self, text):
        return t.tensor(self.tokenizer(text).input_ids)
    def decode(self, tokens):
        return self.tokenizer.batch_decode(tokens)
    def forward(self, x: t.Tensor) -> t.Tensor:
        if x.ndim == 1: x = x.unsqueeze(0)
        x = self.embed(x) + self.pos_embed.weight[:x.shape[1], :]
        for i, block in enumerate(self.blocks):
            x = block(x)
        x = self.ln_f(x)
        x = self.unembed(x)
        return x
    def yap(self, prompt: str, max_length: int = 50) -> t.Tensor:
        with t.inference_mode():
            tokens = t.tensor(self.tokenizer(prompt).input_ids).squeeze()
            for _ in range(max_length):
                logits = self(tokens).squeeze()
                next_token = sampleLogits(logits[-1], 0.9)
                tokens = t.cat([tokens, next_token], dim=-1)
        return tokens.squeeze()
    def seqStr(self, seq: t.Tensor) -> str:
        out = ""
        seq = seq.squeeze()
        for token in seq:
            if token <= self.cfg.d_normal_vocab:
                out += f"{blue}{self.tokenizer.decode(token)}{endc}"
            else:
                out += f"{cyan}<think{token.item() - self.cfg.d_normal_vocab}>{endc}"
        return out
    def printSeq(self, seq: t.Tensor) -> None:
        print("\n", self.seqStr(seq))

t.backends.cuda.enable_flash_sdp(enabled=True)
t.set_default_device(t.device("cuda"))
t.autocast(device_type="cuda", enabled=True, dtype=t.bfloat16)

def train(model: GPT2Thinking, cfg: TrainingConfig, dataset: datasets.Dataset, save_dir: str):
    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    sample_completion_prompt = "George Washington was"

    model.train()

    wandb.init(project="thoughtful", name="gpt2s_think_rep_pen", config=cfg)
    wandb.watch(model, log="all")
    completions_table = wandb.Table(columns=['completion'])
    #wandb.log({"sample_completions": completions_table})
    
    seq_len = model.cfg.seq_len
    seq_indices = t.arange(seq_len, dtype=t.int32, requires_grad=False)
    seq_indices_m1 = t.arange(seq_len - 1, dtype=t.int32, requires_grad=False)

    dl = t.utils.data.DataLoader(dataset, batch_size=1)
    #dl = t.utils.data.DataLoader(dataset, batch_size=16)
    for b, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
        with t.inference_mode():
            model.eval()
            full_seq = batch['input_ids']
            ctx = t.full((seq_len - 1, seq_len), model.eot, dtype=t.int32)
            end_indices = t.arange(seq_len - 1, dtype=t.int32)
            for i in range(seq_len - 1): 
                seq = full_seq.clone().squeeze()
                seq[i+1:] = model.eot
                for s in range(seq_len - i - 1):
                    logits: t.Tensor = model(seq[:end_indices[i] + 1]).squeeze()
                    next_token = logits[-1].argmax(-1)
                    if random.random() > 0.1: next_token = random.randint(model.cfg.d_normal_vocab + 1, model.cfg.d_vocab_total - 1)
                    end_indices[i] = i + s + 1
                    if next_token <= model.cfg.d_normal_vocab:
                        seq[end_indices[i]] = full_seq[..., i + 1].detach()
                        break
                    else:
                        seq[end_indices[i]] = next_token
                
                ctx[i] = seq
            
            model.printSeq(full_seq)
            ctx_g = ctx.clone()
            logits = model(ctx_g)

            z = 10
            last_tt = ctx[z, end_indices[z] - 1].detach().item()
            print(purple, f"{last_tt=}", endc)
            pred_nt = logits[z, end_indices[z]].argmax().detach().item()
            real_nt = seq[z + 1].detach().item()
            pred_logits = logits[z, end_indices[z] - 1]
            print(yellow, f"{logits.shape=}, {ctx.shape=}, {end_indices.shape=}, {seq.shape=}", endc)
            print(pink, f"{end_indices[z]}", endc)
            print(red, f"{pred_logits.max()=}, {pred_logits.argmax()=}", endc)
            print(magenta, f"{ctx[z, end_indices[z]]=}", endc)
            print(f"{purple}start: {z}, end: {end_indices[z]}, predicted real tok: {pred_nt}('{model.tokenizer.decode(pred_nt)}') with logit {logits[z, end_indices[z] - 1, pred_nt]}, real next tok: {real_nt}('{model.tokenizer.decode(real_nt)}'), logit on real next tok: {logits[z, end_indices[z]-1, real_nt]}{endc}")
            
            #logprobs = t.log_softmax(logits[:, :-1], dim=-1)
            #ctx_logprobs = logprobs[t.arange(seq_len -1 )[:, None], t.arange(seq_len - 1)[None, :], ctx_g[:, 1:]]
            ctx_logits = logits[seq_indices_m1[:, None], seq_indices_m1[None, :], ctx_g[:, 1:]]
            real_next_logits = logits[seq_indices_m1, end_indices - 1, ctx[-1, 1:]]
            line(real_next_logits)


            imshow(ctx_logits, title=f"ctx_logits ({ctx_logits.shape})")
            imshow(ctx, title=f"tokens ({ctx.shape})")
            ctx_map = t.zeros_like(ctx)
            ctx_map[ctx > model.eot] = 1
            ctx_map[ctx < model.eot] = -1
            ctx_map[ctx == model.eot] = 0
            #imshow(rewards.unsqueeze(0))
            imshow(ctx_map, title=f"ctx_map ({ctx_map.shape})")

            discounts = t.zeros_like(ctx_logits)
            for i in range(seq_len - 1):
                discounts[i, i:end_indices[i]] = t.pow(cfg.gamma, t.arange(end_indices[i] - i)).flip(dims=(0,))

            imshow(discounts, title=f"discounts ({discounts.shape})")
        
            weighted_ctx_logits = ctx_logits * discounts
            imshow(weighted_ctx_logits, title=f"weighted_ctx_logits ({weighted_ctx_logits.shape})")
            exit()

if __name__ == "__main__":
    model_cfg = ThinkingModelConfig(d_model=512, seq_len=128, d_mlp=2048, d_head=64, n_heads=8, n_layers=8, d_normal_vocab=50257, d_thought_vocab=2048)
    training_cfg = TrainingConfig(gamma=0.95, batch_size=8, lr=3e-4, epochs=1, warmup_steps=1000, weight_decay=1e-2, adam_beta1=0.9, adam_beta2=0.95)
    model = GPT2Thinking(model_cfg)

    #dataset = tokenizeAndSaveDataset(model.tokenizer, training_cfg, "HuggingFaceFW/fineweb-edu", "sample-10BT", f"fineweb-edu-tokenized-128", 0.07, pad=False)
    dataset = loadTokenizedDataset(f"fineweb-edu-tokenized-128")

    train(model, training_cfg, dataset, "./saves")