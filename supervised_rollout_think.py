import tqdm
import datasets
import wandb
import torch as t
from torch import nn
from transformers import GPT2TokenizerFast, AutoTokenizer
import random
from eindex import eindex

from utils import *
from normal import TransformerBlock

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
        x = self.embed(x) + self.pos_embed(t.arange(x.shape[1])).unsqueeze(0)
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

t.set_default_device(t.device("cuda"))

def train(model: GPT2Thinking, cfg: TrainingConfig, dataset: datasets.Dataset):
    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    sample_completion_prompt = "George Washington was"

    model.train()

    wandb.init(project="thoughtful", name="gpt2s_supervised_rollout", config=cfg)
    wandb.watch(model, log="all")
    completions_table = wandb.Table(columns=['completion'])
    #wandb.log({"sample_completions": completions_table})
    
    seq_len = model.cfg.seq_len
    seq_indices = t.arange(seq_len - 1, dtype=t.int32)

    dl = t.utils.data.DataLoader(dataset, batch_size=1)
    #dl = t.utils.data.DataLoader(dataset, batch_size=16)
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

                    next_token = logits[0, -1, model.cfg.d_normal_vocab:].argmax(-1).item() + model.cfg.d_normal_vocab # sampling only from thinking tokens
                    if i + s >= 126 or random.random() > 0.7: next_token = model.end_thought # artificially inflate prob of producing end thought token
                    #print(red, model.embed, blue, next_token, model.end_thought, endc)

                    endices[i] = i + s + 1
                    seq[i + s + 1] = next_token
                
                    if next_token == model.end_thought: # if we produced the end_thought token, rollout is over.
                        break
                
                ctx[i] = seq
            
            
            # This is the sequence which we will be attempting to do next token prediction on.
            # A single next token prediction consists of giving the model a sequence from the dataset, doing inference, producing thinking tokens, then producing an end_thought token.
            # The prediction on the end_thought sequence position is the real next token prediction.
            if b%100 == 0:
                completion = model.yap(batch['text'][0][:seq_len//2])
                completion_str = model.seqStr(completion)
                print("\n", completion_str)
                completions_table.add_data(completion_str)
                wandb.log({"sample_completions": completions_table})
                t.save(model.state_dict(), f"saves/supervised_rollout_think{b}.pth") 
        # ctx holds all our tokens. We generate it withou gradients during the inference step, then clone it.
        # For a sequence of length s, we perform s rollouts. So each row of ctx is an input subsequence followed by a rollout, ending with the end_thought token.
        # ctx has s rows.
        ctx = ctx.clone()
        logits = model(ctx) # These are the model's logits (with gradients) on the ctx sequence.
        endices = endices.clone()

        #z = 0
        #last_tt = ctx[z, endices[z] - 1].detach().item()
        #print(purple, f"{last_tt=}", endc)
        #pred_nt = logits[z, endices[z]].argmax().detach().item()
        #real_nt = seq[z + 1].detach().item()
        #print(yellow, f"{logits.shape=}, {ctx.shape=}, {endices.shape=}, {seq.shape=}", endc)
        #print(pink, f"{endices[z]}", endc)
        #print(magenta, f"{ctx[z, endices[z]]=}", endc)
        #print(f"{purple}start: {z}, end: {endices[z]}, predicted real tok: {pred_nt}('{model.tokenizer.decode(pred_nt)}') with logit {logits[z, endices[z] - 1, pred_nt]}, real next tok: {real_nt}('{model.tokenizer.decode(real_nt)}'), logit on real next tok: {logits[z, endices[z]-1, real_nt]}{endc}")
        
        # These are the models next-token logits and logprobs for each token in each sequence.
        ctx_logits = logits[seq_indices[:, None], seq_indices[None, :], ctx[:, 1:]]
        #ctx_logprobs = logits[seq_indices[:, None], seq_indices[None, :], ctx[:, 1:]].log_softmax(dim=-1)
        ctx_logprobs = logits[seq_indices[:, None], seq_indices[None, :]].log_softmax(dim=-1)[..., ctx[:, 1:]]

        # The logit value at the end_thought token position corresponding to the true next token. We want to maximize these: the logit for the actual next token. These are our rewards.
        pred_logits = logits[seq_indices, endices, ctx[-1, 1:]]
        logit_mean, logit_std = pred_logits.mean().detach(), pred_logits.std().detach() # detach mean so that gradients push mean upwawrds? maybe?
        # normalize rewards across all the rollouts. The think token logprobs of the top half,
        # (in terms of logit on correct token on the end_thought position) are reinforced, the bottom half are pushed down.
        rewards = (pred_logits - logit_mean) / (logit_std + 1e-8)

        discounts = t.zeros_like(ctx_logits)
        for i in range(seq_len - 1):
             discounts[i, i:endices[i]] = t.pow(cfg.gamma, t.arange(endices[i] - i)).flip(dims=(0,)) 
        #imshow(discounts, title=f"discounts ({discounts.shape})")

        weighted_logprobs = ctx_logprobs * discounts
        weighted_action_scores = weighted_logprobs * rewards.unsqueeze(1)
        loss = weighted_action_scores.sum()
        loss.backward()
        if b != 0 and b % cfg.batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        wandb.log({"reward_mean": logit_mean})
        wandb.log({"reward_std": logit_std})
        wandb.log({"think_tok_prop": (think_tok_prop:=(seq_len/((endices-seq_indices).sum().detach().item())))})
        wandb.log({"weighted_token_logits": loss.detach().item()})
        tr.set_description(f"{magenta}reward_mean: {logit_mean.detach().item():.3f}, loss: {loss.detach().item():.3f}, %think: {think_tok_prop:.3f}")

if __name__ == "__main__":
    model_cfg = ThinkingModelConfig(d_model=512, seq_len=128, d_mlp=2048, d_head=64, n_heads=8, n_layers=8, d_normal_vocab=50257, d_thought_vocab=2048)
    training_cfg = TrainingConfig(gamma=0.95, batch_size=8, lr=3e-4, weight_decay=1e-3, adam_beta1=0.9, adam_beta2=0.95)
    model = GPT2Thinking(model_cfg)

    #dataset = tokenizeAndSaveDataset(model.tokenizer, training_cfg, "HuggingFaceFW/fineweb-edu", "sample-10BT", f"fineweb-edu-tokenized-128", 0.07, pad=False)
    dataset = loadTokenizedDataset(f"fineweb-edu-tokenized-128")

    train(model, training_cfg, dataset, "./saves")