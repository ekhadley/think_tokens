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
    def __init__(self, cfg: ThinkingModelConfig):
        super(GPT2Thinking, self).__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.embed = nn.Embedding(cfg.d_vocab_total, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab_total, bias=False)
        self.eot = 50256
        self.end_thought = cfg.d_vocab_total - 1
        self.tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")

    def encode(self, text):
        return t.tensor(self.tokenizer(text).input_ids)
    def decode(self, tokens):
        return self.tokenizer.batch_decode(tokens)
    def forward(self, x: t.Tensor) -> t.Tensor:
        if x.ndim == 1: x = x.unsqueeze(0)
        x = self.embed(x) + self.pos_embed(t.arange(x.shape[1], device=x.device)).unsqueeze(0)
        for i, block in enumerate(self.blocks):
            x = block(x)
        x = self.ln_f(x)
        x = self.unembed(x)
        return x
    def yap(self, prompt: str, max_length: int = 50) -> t.Tensor:
        with t.inference_mode():
            tokens = t.tensor(self.tokenizer(prompt).input_ids)
            for _ in range(max_length):
                next_token = 0
                while next_token != self.end_thought:
                    logits = self.forward(tokens)
                    next_token = logits[0, -1, self.cfg.d_normal_vocab:].argmax(-1).item() + self.cfg.d_normal_vocab
                    tokens = t.cat((tokens, t.tensor([next_token], device=tokens.device)), dim=0)
                    if tokens.shape[-1] >= self.cfg.seq_len: return tokens
                next_token = logits[0, -1, :self.cfg.d_normal_vocab].argmax(-1).item()
                tokens = t.cat((tokens, t.tensor([next_token], device=tokens.device)), dim=0)
        return tokens.squeeze()
    def seqStr(self, seq: t.Tensor) -> str:
        out = ""
        seq = seq.squeeze()
        for token in seq:
            if token <= self.cfg.d_normal_vocab:
                out += f"{blue}{self.tokenizer.decode(token)}{endc}"
            elif token == self.end_thought:
                out += f"{magenta}<end_thought>{endc}"
            else:
                out += f"{cyan}<think{token.item() - self.cfg.d_normal_vocab}>{endc}"
        return out
    def printSeq(self, seq: t.Tensor) -> None:
        print("\n", self.seqStr(seq))

t.backends.cuda.enable_flash_sdp(enabled=True)
t.set_default_device(t.device("cuda"))

def train(model: GPT2Thinking, cfg: TrainingConfig, dataset: datasets.Dataset, save_dir: str):
    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    sample_completion_prompt = "George Washington was"

    model.train()

    wandb.init(project="thoughtful", name="gpt2s_supervised_rollout", config=cfg)
    wandb.watch(model, log="all")
    completions_table = wandb.Table(columns=['completion'])
    #wandb.log({"sample_completions": completions_table})
    
    seq_len = model.cfg.seq_len
    seq_indices = t.arange(seq_len - 1, dtype=t.int32)
    lower_mask = t.tril(t.ones((seq_len - 1, seq_len - 1)), diagonal=-1).bool()

    dl = t.utils.data.DataLoader(dataset, batch_size=1)
    for b, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
        with t.inference_mode():
            model.eval()
            full_seq = batch['input_ids']
            ctx = t.full((seq_len - 1, seq_len), model.eot, dtype=t.int32)
            endices = t.arange(seq_len - 1, dtype=t.int32)
            for i in range(seq_len - 1): # iterates over the sunseqnences of the input, doing a rollout for each one
                seq = full_seq.clone().squeeze()
                seq[i+1:] = model.eot
                for s in range(seq_len - i - 1): # iterates over the string of thinking tokens the model produces
                    logits: t.Tensor = model(seq[:endices[i] + 1])

                    next_token = logits[0, -1, model.cfg.d_normal_vocab:].argmax(-1).item() + model.cfg.d_normal_vocab - 1 # sampling only from thinking tokens
                    if i + s >= 126 or random.random() > 0.7: next_token = model.end_thought # artificially inflate prob of producing end thought token
                    #print(red, model.embed, blue, next_token, model.end_thought, endc)

                    endices[i] = i + s + 1
                    seq[i + s + 1] = next_token
                
                    if next_token == model.end_thought: # if we produced the end_thought token, rollout is over.
                        break
                
                ctx[i] = seq
            
            if b%100 == 0:
                completion = model.yap(batch['text'][0][:seq_len//2])
                completion_str = model.seqStr(completion)
                print("\n", completion_str)
                completions_table.add_data(completion_str)

        ctx = ctx.clone()
        logits = model(ctx) # These are the model's logits (with gradients) on the ctx sequence.
        endices = endices.clone()

        #ctx_map = t.zeros_like(ctx)
        #ctx_map[ctx > model.eot] = 1
        #ctx_map[ctx < model.eot] = -1
        #ctx_map[ctx == model.eot] = 0
        #ctx_map[ctx == model.end_thought] = -2
        #imshow(ctx_map, title=f"ctx_map ({ctx_map.shape})")
        #z = 0
        #last_tt = ctx[z, endices[z] - 1].detach().item()
        #print(purple, f"{last_tt=}", endc)
        #pred_nt = logits[z, endices[z]].argmax().detach().item()
        #real_nt = seq[z + 1].detach().item()
        #print(yellow, f"{logits.shape=}, {ctx.shape=}, {endices.shape=}, {seq.shape=}", endc)
        #print(pink, f"{endices[z]}", endc)
        #print(magenta, f"{ctx[z, endices[z]]=}", endc)
        #print(f"{purple}start: {z}, end: {endices[z]}, predicted real tok: {pred_nt}('{model.tokenizer.decode(pred_nt)}') with logit {logits[z, endices[z], pred_nt]}, real next tok: {real_nt}('{model.tokenizer.decode(real_nt)}'), logit on real next tok: {logits[z, endices[z], real_nt]}{endc}")
        #print(blue, logits.shape, green, ctx.shape, endc)


        #next_tok_logits = logits[seq_indices, endices, ctx[-1, 1:]]
        next_tok_logprobs = logits[seq_indices, endices].log_softmax(dim=-1)[seq_indices, ctx[-1, 1:]]
        #print(purple, next_tok_logprobs.shape, endc)
        #line(next_tok_logprobs.detach())

        #ctx_logits = logits[seq_indices[:, None], seq_indices[None, :], ctx[:, 1:]]
        #think_mask = (ctx >= model.cfg.d_normal_vocab) * (ctx < model.end_thought)
        #ctx_logprobs = logits.log_softmax(dim=-1)[seq_indices[:, None], seq_indices[None, :], ctx[:, 1:]]
        #think_logprobs = ctx_logprobs * think_mask
        #imshow(think_logprobs, title=f"ctx_logprobs ({ctx_logprobs.shape})")



        loss = next_tok_logprobs.mean()
        loss = loss / training_cfg.batch_size
        loss.backward()

        if b != 0 and b%training_cfg.batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

            tr.set_description(f"{magenta}loss: {loss.detach().item()*training_cfg.batch_size:.3f} ")

if __name__ == "__main__":
    model_cfg = ThinkingModelConfig(d_model=512, seq_len=128, d_mlp=2048, d_head=64, n_heads=8, n_layers=8, d_normal_vocab=50257, d_thought_vocab=2048)
    training_cfg = TrainingConfig(gamma=0.95, batch_size=16, lr=3e-4, epochs=1, warmup_steps=1000, weight_decay=1e-3, adam_beta1=0.9, adam_beta2=0.95)
    model = GPT2Thinking(model_cfg)

    #dataset = tokenizeAndSaveDataset(model.tokenizer, training_cfg, "HuggingFaceFW/fineweb-edu", "sample-10BT", f"fineweb-edu-tokenized-128", 0.07, pad=False)
    dataset = loadTokenizedDataset(f"fineweb-edu-tokenized-128")

    train(model, training_cfg, dataset, "./saves")