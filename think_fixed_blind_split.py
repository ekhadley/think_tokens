import eindex
import wandb
import tqdm
import datasets

from utils import *
from models import GPT2SplitModel, TrainingConfig, SplitModelConfig

MOVE_IDX = 20

def train(answer_model: GPT2SplitModel, think_model: GPT2SplitModel, cfg: TrainingConfig, dataset: datasets.Dataset, epochs: int = 5):
    answer_opt = t.optim.AdamW(answer_model.parameters(), lr=cfg.answer_lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    think_opt = t.optim.AdamW(think_model.parameters(), lr=cfg.think_lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay, maximize=True)
    answer_model.train()
    think_model.train()

    run_cfg = {"answer_model": answer_model.cfg.to_dict(), "think_model": think_model.cfg.to_dict(), "training": cfg.to_dict()}
    wandb.init(project="gpt_chess", name="think_fixed_blind_split", config=run_cfg)

    parameters = [p for p in think_model.parameters() if p.requires_grad]
    grad_norm = 42069

    group_size = cfg.group_size
    full_batch_size = group_size * cfg.batch_size
    #max_seq_len = think_model.cfg.seq_len - 1 - cfg.think_len
    max_seq_len = dataset['input_ids'].shape[1] - 1 - cfg.think_len
    
    pred_acc = 0.0

    d_normal_vocab = answer_model.cfg.d_vocab_out
    d_thought_vocab = think_model.cfg.d_vocab_out

    batch_indices = t.arange(cfg.batch_size, requires_grad=False)
    group_indices = t.arange(group_size, requires_grad=False)
    full_batch_indices = t.arange(full_batch_size, requires_grad=False)

    epsilon = 1.0
    
    dl = t.utils.data.DataLoader(dataset, batch_size=cfg.batch_size)
    for epoch in range(epochs):
        for b, batch in enumerate(tr:=tqdm.tqdm(dl, ncols=140)):
            with t.inference_mode(): # generate rollouts without gradients
                seq_len = random.randint(1, max_seq_len)
                seqs: t.Tensor = batch['input_ids'][:, :seq_len]
                seqs = seqs.unsqueeze(0).repeat(1, 1, group_size).reshape(full_batch_size, -1) # repeat each seq in the batch group_size times
                ans_toks = batch['input_ids'][batch_indices, seq_len].reshape(-1, 1).repeat(1, group_size).flatten()

                for i_t in range(cfg.think_len): # generate thinking tokens
                    think_logits = think_model(seqs)
                    think_probs = t.softmax(think_logits[:, -1], dim=-1)
                    think_toks = t.multinomial(think_probs, num_samples=1) + d_normal_vocab # samples a continuation token for each sequence (each group element in each batch element for the whole batch). This gives us natural rollout variance
                    #if random.random() < epsilon:
                        #think_toks = t.randint(d_normal_vocab, d_normal_vocab+d_thought_vocab, (full_batch_size, 1))
                    #else:
                        #think_toks = think_logits[:, -1].argmax(dim=-1, keepdim=True) + d_normal_vocab
                    seqs = t.cat([seqs, think_toks], dim=1)

                rollouts = seqs[:, -cfg.think_len:] - d_normal_vocab
                logits = answer_model(rollouts).squeeze()
                logprobs = t.log_softmax(logits[:, -1], dim=-1)
                pred_rewards = logprobs[full_batch_indices, ans_toks]  # answer model's logprob of the correct answer token is our reward signal
                pred_reward_mean = pred_rewards.mean().item() # mean of the predicted rewards
                pred_reward_mean_by_group = pred_rewards.reshape(cfg.batch_size, group_size).mean(dim=-1, keepdim=True)
                normed_pred_rewards = pred_rewards.reshape(cfg.batch_size, group_size) - pred_reward_mean_by_group
                normed_pred_rewards = t.clamp_min(normed_pred_rewards.flatten(), 0) # normalize the rewards
                epsilon = max(epsilon * cfg.eps_decay, cfg.eps_min)

            seqs = seqs.clone() 
            rollouts = rollouts.clone()
            normed_pred_rewards = normed_pred_rewards.clone()
            ans_toks = ans_toks.clone()

            pred_logits = answer_model(rollouts).squeeze()
            pred_logprobs = t.log_softmax(pred_logits[:, -1], dim=-1) # real token logprob distn on the last thought token
            pred_reward = pred_logprobs[full_batch_indices, ans_toks].mean()
            pred_reward.backward()
            
            think_logits = think_model(seqs).squeeze()
            think_logprobs = t.log_softmax(think_logits[full_batch_indices, -cfg.think_len - 1:-1], dim=-1) # logprob distns for the positions where thinking tokens were emitted
            action_logprobs = think_logprobs[full_batch_indices.unsqueeze(-1), t.tensor(range(cfg.think_len)).unsqueeze(0), rollouts] # logprob of the thinking tokens that were outputted
            weighted_action_logprobs = action_logprobs * normed_pred_rewards.unsqueeze(-1)
            think_reward = weighted_action_logprobs.mean()
            think_reward.backward()
            grad_norm = t.norm(t.stack([t.norm(p.grad.detach(), 2.0) for p in parameters]), 2.0).item()

            answer_opt.step()
            think_opt.step()
            answer_opt.zero_grad()
            think_opt.zero_grad()

            with t.inference_mode():
                pred_prob_var = t.exp(pred_rewards).var().item() # answer prob variance for logging. Tells us how much the rollout influences the answer model's final output
                pred_reward_var = pred_rewards.var().item() # variance of the predicted rewards for logging. tells us similar thing.

                wandb.log({
                    "loss": -pred_reward_mean,
                    "think_reward": think_reward,
                    "num_think": cfg.think_len,
                    "acc": pred_acc,
                    "pred_reward_var": pred_reward_var,
                    "pred_prob_var": pred_prob_var,
                    "grad_norm": grad_norm,
                    #"prob_force_end_thought": 0.0,
                    "epsilon": epsilon,
                    #"think_logprobs": think_logprobs[0],
                    #"entropy_reward": entropy,
                })
                tr.set_description(f"{magenta}pred reward: {pred_reward_mean:.3f}, think reward: {think_reward:.3f}, acc: {pred_acc:.3f}, epsilon: {epsilon:.3f}")

                if b % 32 == 0:
                    print()
                    rollout_mean_logprob = action_logprobs.mean(dim=-1)
                    for row in range(rollouts.shape[0]):
                        print(f"{blue}{rollouts[row].tolist()} {magenta}{rollout_mean_logprob[row].item():.3f} : {cyan}{pred_rewards[row].item():.3f} {green}({normed_pred_rewards[row].item():.3f}){endc}")
                    best_rollout_idx = pred_rewards.argmax().item()
                    print(magenta, think_logprobs[best_rollout_idx].T, endc)

                    pred_acc = (pred_logprobs.argmax(dim=-1) == ans_toks).float().mean().item()
                    #t.save(answer_model.state_dict(), f"saves/add_think_fixed_blind_super_clean_split_answer{b}.pth")
                    #t.save(think_model.state_dict(), f"saves/add_think_fixed_blind_super_clean_split_think{b}.pth")




if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    d_vocab = 64
    d_thought_vocab = 64
    think_len = 16
    answer_model_cfg = SplitModelConfig(d_model=64, seq_len=think_len, d_mlp=256, d_head=32, n_heads=4, n_layers=2, d_vocab_in=d_thought_vocab, d_vocab_out=d_vocab, d_thought_vocab=d_thought_vocab)
    think_model_cfg =  SplitModelConfig(d_model=64, seq_len=256,       d_mlp=256, d_head=32, n_heads=8, n_layers=6, d_vocab_in=d_vocab + d_thought_vocab, d_vocab_out=d_thought_vocab, d_thought_vocab=d_thought_vocab)

    answer_model = GPT2SplitModel(answer_model_cfg)
    think_model = GPT2SplitModel(think_model_cfg)
    #tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")

    training_cfg = TrainingConfig(
        think_lr=6e-4,
        answer_lr=6e-4,
        think_len=think_len,
        group_size=32,
        batch_size=64,
        eps_decay=0.999,
        eps_min=0.01,
    )

    dataset = datasets.load_dataset(f"eekay/chess-games-40moves-3min")["train"]
    dataset.set_format(type='torch')
    trainset, testset = dataset.train_test_split(test_size=0.005).values()

    train(answer_model, think_model, training_cfg, dataset, epochs=10)