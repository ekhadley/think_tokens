import torch as t
from torch import nn
from torch.nn import functional as F
from transformers import GPT2TokenizerFast
import wandb
import huggingface_hub
import tqdm
import datasets

from utils import *

t.backends.cuda.enable_flash_sdp(enabled=True)
t.set_default_device(t.device("cuda"))


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

class GPT2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(GPT2, self).__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)

        self.tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2", pad_token="<PAD>")
        #self.tokenizer.pad_token = "<PAD>"
        #self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    
    def encode(self, text):
        #return self.tokenizer.encode(text)
        return self.tokenizer.tokenize(text)
    def decode(self, tokens):
        #return self.tokenizer.decode(tokens)
        return self.tokenizer.batch_decode(tokens)
    def forward(self, x: t.Tensor) -> t.Tensor:
        #print(red, x.shape, endc)
        x = self.embed(x)
        #print(orange, x.shape, endc)
        for i, block in enumerate(self.blocks):
            #print(f"{pink} x[{i}]: {x.shape}{endc}")
            x = block(x)
        #print(lime, x.shape, endc)
        x = self.ln_f(x)
        #print(magenta, x.shape, endc)
        x = self.unembed(x)
        #print(blue, x.shape, endc)
        return x
    def yap(self, prompt: str, max_length: int = 50):
        with t.no_grad():
            tokens = t.tensor(self.tokenizer(prompt).input_ids).squeeze()
            for _ in range(max_length):
                logits = self(tokens).squeeze()
                next_token = sampleLogits(logits[-1], 0.9)
                tokens = t.cat([tokens, next_token], dim=-1)
        return "".join(self.tokenizer.batch_decode(tokens))


def train(model, cfg: TrainingConfig, dataset: datasets.Dataset, save_dir: str):
    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)

    model.train()

    wandb.init(project="thoughtful", name="gpt2s_normal", config=cfg)
    wandb.watch(model, log="all")

    sample_completion = model.yap("George Washington was")
    print(yellow, sample_completion, endc)
    table = wandb.Table(data=[[sample_completion]], columns=['completion'])
    wandb.log({"sample_completion": table})

    dl = t.utils.data.DataLoader(dataset, batch_size=cfg.batch_size)
    for i, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
        tokens = batch['input_ids']
        logits = model(tokens)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item()})
        tr.set_description(f"{purple}loss: {loss.item():.3f}")

        if i%1000 == 0:
            sample_completion = model.yap("George Washington was")
            print(yellow, sample_completion, endc)
            table = wandb.Table(data=[[sample_completion]], columns=['completion'])
            wandb.log({"sample_completion": table})

            t.save(model.state_dict(), f"{save_dir}/save_{i}.pth")

if __name__ == "__main__":
    model_cfg = ModelConfig(d_model=512, d_mlp=2048, d_head=64, n_heads=8, n_layers=8, d_vocab=50257 + 1)
    model = GPT2(model_cfg)

    training_cfg = TrainingConfig(batch_size=16, lr=3e-4, epochs=1, warmup_steps=1000, weight_decay=1e-2, adam_beta1=0.9, adam_beta2=0.95)
    #dataset = datasets.load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train").train_test_split(0.07)['test']
    #dataset = dataset.map(lambda x: model.tokenizer(x['text'], padding='max_length', truncation=True, max_length=training_cfg.seq_len))
    #dataset.save_to_disk(f"fineweb-edu-tokenized-600M")
    dataset = datasets.load_from_disk(f"fineweb-edu-tokenized-600M")

    dataset.set_format(type='torch')
    train(model, training_cfg, dataset, "./saves")