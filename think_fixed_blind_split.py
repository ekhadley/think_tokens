import eindex
import wandb
import tqdm
import datasets

from utils import *
from models import GPT2SplitModel, TrainingConfig, SplitModelConfig




class TokenIndexIter: # let's us randomly iterate over tokens (a random position from a random sequence) from the dataset
    def __init__(self, n_sequences: int, seq_len: int):
        self.n_sequences = n_sequences
        self.seq_len = seq_len

def train(answer_model: GPT2SplitModel, think_model: GPT2SplitModel, tokenizer: GPT2TokenizerFast, cfg: TrainingConfig, dataset: datasets.Dataset):
    answer_opt = t.optim.AdamW(answer_model.parameters(), lr=cfg.answer_lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    think_opt = t.optim.AdamW(think_model.parameters(), lr=cfg.think_lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    answer_model.train()
    think_model.train()

    wandb.init(project="thoughtful", name="gpt2s_think_fixed_blind_split", config=cfg)
    wandb.config.update(cfg.to_dict())

    #sample_completion = model.yap("George Washington was")
    #print(yellow, sample_completion, endc)
    #table = wandb.Table(data=[[sample_completion]], columns=['completion'])
    #wandb.log({"sample_completion": table})
    group_size = cfg.group_size
    full_batch_size = group_size * cfg.batch_size
    max_seq_len = think_model.cfg.seq_len - 1 - cfg.think_len

    pred_acc = 0.0

    d_normal_vocab = answer_model.cfg.d_vocab_out
    d_thought_vocab = think_model.cfg.d_vocab_out
    end_of_text = d_normal_vocab - 1
    d_total_vocab = d_normal_vocab + d_thought_vocab

    batch_indices = t.arange(cfg.batch_size, requires_grad=False)
    group_indices = t.arange(group_size, requires_grad=False)
    full_batch_indices = t.arange(full_batch_size, requires_grad=False)

    epsilon = 0.0
    
    dl = t.utils.data.DataLoader(dataset, batch_size=cfg.batch_size)
    for b, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=140))):
        with t.inference_mode():
            seq_len = random.randint(1, max_seq_len)
            seqs = batch['input_ids'][:, :seq_len]
            ans_toks = batch['input_ids'][batch_indices, seq_len].reshape(-1, 1).repeat(1, group_size).flatten()
            seqs = seqs.unsqueeze(0).repeat(1, 1, group_size).reshape(full_batch_size, -1)
            #print()
            #print(orange, seqs.shape, endc)

            for i_t in range(cfg.think_len):
                think_logits = think_model(seqs)
                think_probs = t.softmax(think_logits[:, -1], dim=-1)
                think_toks = t.multinomial(think_probs, num_samples=1) + d_normal_vocab
                seqs = t.cat([seqs, think_toks], dim=1)

            rollouts = seqs[:, -cfg.think_len:] - d_normal_vocab
            #print(pink, seqs.shape, lime, rollouts.shape, endc)
            logits = answer_model(rollouts).squeeze()
            logprobs = t.log_softmax(logits[:, -1], dim=-1)
            pred_rewards = logprobs[full_batch_indices, ans_toks]  # ans_tok is the single token ID
            pred_reward_mean = pred_rewards.mean().item() # mean of the predicted rewards
            normed_pred_rewards = (pred_rewards - pred_reward_mean) / (pred_rewards.std() + 1e-8) # normalize the rewards
            #print(magenta, normed_pred_rewards.shape, endc)
            
            epsilon = max(epsilon * cfg.eps_decay, cfg.eps_min)

        seqs = seqs.clone() 
        rollouts = rollouts.clone()
        normed_pred_rewards = normed_pred_rewards.clone()
        ans_toks = ans_toks.clone()

        pred_logits = answer_model(rollouts).squeeze()
        #print(yellow, pred_logits.shape, endc)
        pred_logprobs = t.log_softmax(pred_logits[:, -1], dim=-1) # real token logprob distn on the last thought token
        #print(cyan, pred_logprobs.shape, endc)
        pred_reward = pred_logprobs[:, ans_toks].sum() # logprob value on the correct answer token
        pred_reward.backward()
        
        think_logits = think_model(seqs).squeeze()
        #print(blue, think_logits.shape, endc)
        think_logprobs = t.log_softmax(think_logits[full_batch_indices, -cfg.think_len - 1:-1], dim=-1) # logprob distns for the positions where thinking tokens were emitted
        #print(red, think_logprobs.shape, endc)
        #print(gray, rollouts, endc)
        action_logprobs = think_logprobs[full_batch_indices.unsqueeze(-1), t.tensor(range(cfg.think_len)).unsqueeze(0), rollouts] # logprob of the thinking tokens that were outputted
        weighted_action_logprobs = action_logprobs * normed_pred_rewards.unsqueeze(-1)
        #print(purple, action_logprobs.shape, endc)
        think_reward = weighted_action_logprobs.sum()
        think_reward.backward()

        answer_opt.step()
        think_opt.step()
        answer_opt.zero_grad()
        think_opt.zero_grad()


        with t.inference_mode():
            pred_prob_var = t.exp(pred_rewards).var().item() # answer prob variance for logging
            pred_reward_var = pred_rewards.var().item() # variance of the predicted rewards for logging
            think_loss = action_logprobs[(pred_rewards > 0)].mean()
            
            wandb.log({
                "pred_reward": pred_reward_mean,
                "think_reward": think_reward,
                "num_think": cfg.think_len,
                "pred_reward_var": pred_reward_var,
                "pred_prob_var": pred_prob_var,
                #"prob_force_end_thought": 0.0,
                "epsilon": epsilon,
                #"think_logprobs": think_logprobs[0],
                #"entropy_reward": entropy,
                "think_loss": think_loss,
            })
            #printSeq(rollouts[0], simple_tokenizer, model.cfg)
            tr.set_description(f"{magenta}pred reward: {pred_reward_mean:.3f}, think reward: {think_reward:.3f}, epsilon: {epsilon:.3f}, pred acc: {pred_acc:.3f}")


        if b % 32_000 == 0:
            print()
            rollout_mean_logprob = action_logprobs.mean(dim=-1)
            for row in range(rollouts.shape[0]):
                print(f"{blue}{rollouts[row].tolist()} {magenta}{rollout_mean_logprob[row].item():.3f} : {cyan}{pred_rewards[row].item():.3f} {green}({normed_pred_rewards[row].item():.3f}){endc}")
            best_rollout_idx = pred_rewards.argmax().item()
            print(magenta, think_logprobs[best_rollout_idx].T, endc)

            pred_acc = (pred_logprobs.argmax(dim=-1) == ans_toks).float().mean().item()
            wandb.log({"benchmark_accuracy": pred_acc})
            #t.save(answer_model.state_dict(), f"saves/add_think_fixed_blind_super_clean_split_answer{b}.pth")
            #t.save(think_model.state_dict(), f"saves/add_think_fixed_blind_super_clean_split_think{b}.pth")




if __name__ == "__main__":
    t.set_default_device(t.device("cpu"))

    d_vocab = 50_257
    d_thought_vocab = 2048
    think_len = 2
    #answer_model_cfg = SplitModelConfig(d_model=32, seq_len=128, d_mlp=128, d_head=16, n_heads=4, n_layers=1, d_vocab_in=d_thought_vocab, d_vocab_out=d_vocab, d_thought_vocab=d_thought_vocab)
    #think_model_cfg =  SplitModelConfig(d_model=64, seq_len=128, d_mlp=128, d_head=32, n_heads=4, n_layers=2, d_vocab_in=d_vocab + d_thought_vocab, d_vocab_out=d_thought_vocab, d_thought_vocab=d_thought_vocab)
    answer_model_cfg = SplitModelConfig(d_model=512, seq_len=think_len, d_mlp=2048, d_head=64, n_heads=4, n_layers=2, d_vocab_in=d_thought_vocab, d_vocab_out=d_vocab, d_thought_vocab=d_thought_vocab)
    think_model_cfg =  SplitModelConfig(d_model=512, seq_len=128,       d_mlp=2048, d_head=64, n_heads=8, n_layers=8, d_vocab_in=d_vocab + d_thought_vocab, d_vocab_out=d_thought_vocab, d_thought_vocab=d_thought_vocab)

    answer_model = GPT2SplitModel(answer_model_cfg)
    think_model = GPT2SplitModel(think_model_cfg)
    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")

    training_cfg = TrainingConfig(
        think_lr=3e-4,
        answer_lr=3e-4,
        weight_decay=1e-6,
        entropy_reward_weight=0.0,
        think_len=think_len,
        group_size=32,
        batch_size=64,
        eps_decay=0.999998,
        eps_min=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95
    )

    #dataset = tokenizeAndSaveDataset(model.tokenizer, model_cfg, "HuggingFaceFW/fineweb-edu", "sample-10BT", f"fineweb-edu-tokenized-512", 0.07, pad=False)
    dataset = loadTokenizedDataset("fineweb-edu-tokenized-128")

    train(answer_model, think_model, tokenizer, training_cfg, dataset)