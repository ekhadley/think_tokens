import tqdm
import datasets
import wandb
import torch as t
from torch import nn
from torch.nn import functional as F
import transformers as tf
from transformers import GPT2TokenizerFast, AutoTokenizer
import einops
import random
from eindex import eindex

from utils import *

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(TransformerBlock, self).__init__()
        self.linear1 = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.linear2 = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + ff_output)
        return x

class GPT2Thinking(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(GPT2Thinking, self).__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.embed = nn.Embedding(cfg.d_vocab_total, cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab_total - 1, bias=False)
        self.eot = 50256

        self.tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
    def encode(self, text):
        return t.tensor(self.tokenizer(text).input_ids)
    def decode(self, tokens):
        return self.tokenizer.batch_decode(tokens)
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.embed(x)
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
    
    ctx = t.full((cfg.seq_len - 1, cfg.seq_len), model.eot, dtype=t.int32)
    seq_indices = t.arange(cfg.seq_len, dtype=t.int32)
    
    z = 3

    dl = t.utils.data.DataLoader(dataset, batch_size=1)
    #dl = t.utils.data.DataLoader(dataset, batch_size=16)
    for b, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
        with t.inference_mode():
            model.eval()
            full_seq = batch['input_ids']
            end_indices = t.arange(cfg.seq_len - 1, dtype=t.int32)
            for i in range(training_cfg.seq_len - 1): 
                seq = full_seq.clone().squeeze()
                seq[i+1:] = model.eot

                for s in range(training_cfg.seq_len - i - 1):
                    logits: t.Tensor = model(seq[:end_indices[i] + 1])
                    #next_token = sampleLogits(logits[-1], temperature=0.9)
                    next_token = logits[-1].argmax(-1)
                    #print(magenta, logits.shape, logits[s + i - 1].shape, endc)
                    if random.random() > 0.1: next_token = random.randint(model.cfg.d_normal_vocab + 1, model.cfg.d_vocab_total - 1)
                    seq[i + s + 1] = next_token
                    end_indices[i] = i + s + 1
                    if i == z:
                        print(cyan, end_indices[z], endc)
                        print(lime, seq[:end_indices[i] + 1], endc)
                        print(magenta, logits.shape, endc)
                        print(magenta, logits[-1].argmax(-1), endc)
                        print(magenta, logits[-1].max(), endc)
                        print(lime, seq[:end_indices[i] + 1], endc)
                        print(next_token)
                    if next_token <= model.cfg.d_normal_vocab:
                        break
                ctx[i] = seq # save the context sequence

        ctx_g = ctx.clone()

        model.printSeq(ctx[z])
        logits = model(ctx_g)


        pred_nt = ctx[z, end_indices[z]].detach().item()
        real_nt = seq[z + 1].detach().item()
        pred_logits = logits[z, end_indices[z] - 1]
        print(red, pred_logits.max(), pred_logits.argmax(), endc)
        print(magenta, ctx[z, end_indices[z]], endc)
        print(f"{purple}start: {z}, end: {end_indices[z]}, predicted real tok: {pred_nt}('{model.tokenizer.decode(pred_nt)}') with logit {logits[z, end_indices[z] - 1, pred_nt]}, real next tok: {(rnt:=seq[z+1])}('{model.tokenizer.decode(rnt.detach().item())}'), logit on real next tok: {logits[z, end_indices[z]+1, seq[z+1]]}{endc}")


        pred_logits = logits[seq_indices[:-1], end_indices - 1, seq[seq_indices[:-1]]] # the logit on the correct token for the last token in the rollout (the real token prediction)
        print(orange, pred_logits, endc)
        rewards = (pred_logits - pred_logits.mean()) / pred_logits.std() # the reward is the normalized logit on the correct token
        think_mask = ctx_g > model.cfg.d_normal_vocab
        
        ctx_map = t.zeros_like(ctx)
        ctx_map[think_mask] = 1
        ctx_map[ctx < model.eot] = -1
        ctx_map[ctx == model.eot] = 0
        imshow(rewards.unsqueeze(0))
        imshow(ctx_map)
        exit()
        

if __name__ == "__main__":
    model_cfg = ThinkingModelConfig(d_model=512, d_mlp=2048, d_head=64, n_heads=8, n_layers=8, d_normal_vocab=50257, d_thought_vocab=2048)
    training_cfg = TrainingConfig(seq_len=128, gamma=0.95, batch_size=8, lr=3e-4, epochs=1, warmup_steps=1000, weight_decay=1e-2, adam_beta1=0.9, adam_beta2=0.95)
    model = GPT2Thinking(model_cfg)

    #dataset = tokenizeAndSaveDataset(model.tokenizer, training_cfg, "HuggingFaceFW/fineweb-edu", "sample-10BT", f"fineweb-edu-tokenized-think-128-600M", 0.07, pad=False)
    #dataset = loadTokenizedDataset(f"fineweb-edu-tokenized-think-256-600M")
    dataset = loadTokenizedDataset(f"fineweb-edu-tokenized-think-128-600M")

    train(model, training_cfg, dataset, "./saves")
    
    with t.no_grad():
        ref = loadReferenceModel("openai-community/gpt2-xl")
        print(vars(model))
        ref_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
        seq = dataset[0]['input_ids'][0:-20].unsqueeze(0)
        print(blue, seq, endc)
        str_toks = model.tokenizer.decode(seq.squeeze())
        print(red, str_toks, endc)
        toks = t.tensor(ref_tokenizer(str_toks)['input_ids']).unsqueeze(0)
        print(purple, toks, endc)
        with t.no_grad():
            for _ in range(50):
                logits = ref(toks).logits.squeeze()
                next_token = sampleLogits(logits[-1])
                toks = t.cat([toks, next_token.unsqueeze(0)], dim=-1)
        out = "".join(ref_tokenizer.batch_decode(toks))
        print(green, out, endc)