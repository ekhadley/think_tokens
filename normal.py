import torch as t
from torch import nn
from torch.nn import functional as F
from transformers import GPT2TokenizerFast
import wandb
import tqdm
import datasets

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

class GPT2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(GPT2, self).__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)

        self.tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
    
    def encode(self, text):
        return self.tokenizer.tokenize(text)
    def decode(self, tokens):
        return self.tokenizer.batch_decode(tokens)
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.embed(x)
        if x.ndim == 2: x = x.unsqueeze(0)
        x = x + self.pos_embed.weight[:x.shape[1]]
        for i, block in enumerate(self.blocks):
            x = block(x)
        x = self.ln_f(x)
        x = self.unembed(x)
        return x
    def yap(self, prompt: str, max_length: int = 50):
        with t.no_grad():
            tokens = t.tensor(self.tokenizer(prompt).input_ids).squeeze()
            for _ in range(max_length):
                logits = self(tokens).squeeze()
                next_token = sampleLogits(logits[-1], 0.9)
                tokens = t.cat([tokens, next_token], dim=-1)
        return "".join(self.tokenizer.batch_decode(tokens))

t.backends.cuda.enable_flash_sdp(enabled=True)
t.set_default_device(t.device("cuda"))
t.autocast(device_type="cuda", enabled=True, dtype=t.float16)

def train(model, cfg: TrainingConfig, dataset: datasets.Dataset, save_dir: str):
    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)

    model.train()

    wandb.init(project="thoughtful", name="gpt2s_normal", config=cfg)
    wandb.watch(model, log="all")

    sample_completion = model.yap("George Washington was")
    print(yellow, sample_completion, endc)
    table = wandb.Table(data=[[sample_completion]], columns=['completion'])
    wandb.log({"sample_completion": table})

    seq_indices = t.arange(model.cfg.seq_len, requires_grad=False)

    dl = t.utils.data.DataLoader(dataset, batch_size=cfg.batch_size)
    for i, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
        tokens = batch['input_ids']
        logits = model(tokens)
        #logprobs = t.log_softmax(logits[:, :-1], dim=-1)
        #loss = logprobs.gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        #loss = loss.mean()
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
        #loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item()})
        tr.set_description(f"{magenta}loss: {loss.item():.3f}")

        if i%1000 == 0:
            sample_completion = model.yap("George Washington was")
            print(yellow, sample_completion, endc)
            table = wandb.Table(data=[[sample_completion]], columns=['completion'])
            wandb.log({"sample_completion": table})

            t.save(model.state_dict(), f"{save_dir}/save_{i}.pth")

if __name__ == "__main__":
    model_cfg = ModelConfig(d_model=512, seq_len=256, d_mlp=2048, d_head=64, n_heads=8, n_layers=8, d_vocab=50257)
    model = GPT2(model_cfg)

    training_cfg = TrainingConfig(batch_size=16, lr=3e-4, weight_decay=1e-2, adam_beta1=0.9, adam_beta2=0.95)
    dataset = datasets.load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train").train_test_split(0.07)['test']
    dataset = dataset.map(lambda x: model.tokenizer(x['text'], truncation=True, max_length=model_cfg.seq_len))
    #dataset.save_to_disk(f"fineweb-edu-tokenized-600M")
    #dataset = datasets.load_from_disk(f"datasets/fineweb-edu-tokenized-600M")
    dataset.save_to_disk(f"datasets/fineweb-edu-tokenized-256")
    exit()

    dataset.set_format(type='torch')
    train(model, training_cfg, dataset, "./saves")

# current state:
# why running oom?
# update thinking versions to use logprob loss, oops