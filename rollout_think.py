import tqdm
import datasets
import wandb
import torch as t
from torch import nn
from torch.nn import functional as F
import transformers as tf
from transformers import GPT2TokenizerFast, AutoTokenizer
import einops
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
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab_total - 1, bias=False) # -1 because we don't want to predict <pad>

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
        print(self.seqStr(seq))

def apply_repetition_penalty_batch(logits: t.Tensor, generated: list[list[int]], penalty: float):
    B = logits.shape[0]
    device = logits.device
    indices = t.arange(logits.shape[1], device=device)
    for b in range(B):
        token_ids = t.tensor(list(set(generated[b])), device=device)
        logits_b = logits[b, indices, token_ids]

        # Apply penalty to just the selected tokens
        logits[b, indices, token_ids] = t.where(
            logits_b > 0,
            logits_b / penalty,
            logits_b * penalty
        )
    return logits


t.backends.cuda.enable_flash_sdp(enabled=True)
t.set_default_device(t.device("cuda"))
t.autocast(device_type="cuda", enabled=True, dtype=t.float16)

def train(model, cfg: TrainingConfig, dataset: datasets.Dataset, save_dir: str):
    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    sample_completion_prompt = "George Washington was"

    model.train()

    wandb.init(project="thoughtful", name="gpt2s_think_rep_pen", config=cfg)
    wandb.watch(model, log="all")
    completions_table = wandb.Table(columns=['completion'])
    #wandb.log({"sample_completions": completions_table})

    #ref_model_name = "meta-llama/Meta-Llama-3.1-8b"
    ref_model_name = "openai-community/gpt2-xl"
    ref = loadReferenceModel(ref_model_name)
    ref.eval()

    discounts = t.zeros((cfg.seq_len, cfg.seq_len))
    for i in range(cfg.seq_len):
        for j in range(i, cfg.seq_len):
            discounts[i, j] = cfg.gamma ** (j - i)
    
    #l = t.utils.data.DataLoader(dataset, batch_size=cfg.batch_size)
    dl = t.utils.data.DataLoader(dataset, batch_size=16)
    for b, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
        with t.inference_mode():
            model.eval()
            #seq = batch['input_ids'].repeat(cfg.batch_size, 1)
            seq = batch['input_ids']
            
            repetition_penalty = 1.2
            for i in range(training_cfg.seq_len): # inference out next tokens
                logits = model(seq)
                next_token = sampleLogits(logits[:, -1, :])
                seq = t.cat([seq, next_token], dim=-1)

            real_toks_mask = seq < model.cfg.d_normal_vocab # mask out thought tokens
            masked_toks = seq * real_toks_mask # mask out thought tokens
            ref_logits = ref(masked_toks, attention_mask=real_toks_mask).logits # get logits from reference model. this tells us house likely* the learning model's outputs were
            #ref_logits = apply_repetition_penalty_batch(ref_logits, masked_toks, repetition_penalty) # apply repetition penalty to reference model logits
            rewards: t.Tensor = eindex(ref_logits[:, training_cfg.seq_len-1:-1, :], masked_toks[:, training_cfg.seq_len:], "batch seq [batch seq]") * real_toks_mask[:, training_cfg.seq_len-1:-1] # rewards are the logits of the reference model for the tokens we generated
            reward_mean, reward_std = rewards.mean(), rewards.var()
            normalized_rewards = (rewards - reward_mean) / (reward_std + 1e-8) # normalize rewards
            discounted_rewards = (normalized_rewards.unsqueeze(1) * discounts.unsqueeze(0)).sum(-1) # discounted rtg

            if b%100 == 0:
                completions_table.add_data(model.tokenizer.decode(seq[0]))
                wandb.log({"sample_completions": completions_table})
                model.printSeq(seq[0])
                t.save(model.state_dict(), f"{save_dir}/think_save_{b}.pth")

        seq = seq.clone()
        discounted_rewards = discounted_rewards.clone()

        model.train()
        logits = model(seq)
        seq_logits: t.Tensor = eindex(logits[:, training_cfg.seq_len-1:-1, :], seq[:, training_cfg.seq_len:], "batch seq [batch seq]")
        weighted_logits = seq_logits * discounted_rewards # higher reward is good. higher logit means higher probability. want prob to go up for positive reward actions
        entropy_weight = 0.01
        entropy = -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(-1).mean()
        loss = weighted_logits.mean() + entropy_weight * entropy # maximize the weighted logits and minimize the entropy
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({"reward_mean": reward_mean})
        wandb.log({"reward_std": reward_std})
        wandb.log({"weighted_token_logits": loss.detach().item()})
        tr.set_description(f"{blue}reward_mean: {reward_mean.detach().item():.3f}, loss: {loss.detach().item():.3f}")



if __name__ == "__main__":
    model_cfg = ThinkingModelConfig(d_model=512, d_mlp=2048, d_head=64, n_heads=8, n_layers=8, d_normal_vocab=50257, d_thought_vocab=2048)
    training_cfg = TrainingConfig(seq_len=128, gamma=0.95, batch_size=8, lr=3e-4, epochs=1, warmup_steps=1000, weight_decay=1e-2, adam_beta1=0.9, adam_beta2=0.95)
    model = GPT2Thinking(model_cfg)

    #dataset = tokenizeAndSaveDataset(model.tokenizer, training_cfg, "HuggingFaceFW/fineweb-edu", "sample-10BT", f"fineweb-edu-tokenized-think-128-600M", 0.07, pad=False)
    #dataset = loadTokenizedDataset(f"fineweb-edu-tokenized-think-256-600M")
    dataset = loadTokenizedDataset(f"fineweb-edu-tokenized-think-128-600M")

    train(model, training_cfg, dataset, "./saves")