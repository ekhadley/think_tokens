import eindex
import wandb
import tqdm
import datasets

from utils import *
from models import GPT2SplitModel, TrainingConfig, SplitModelConfig

@t.inference_mode()
def test_acc(answer_model: GPT2SplitModel, think_model: GPT2SplitModel, cfg: TrainingConfig, inp_max: int) -> tuple[t.Tensor, t.Tensor]:
    inp_toks = t.arange(0, inp_max).reshape(-1, 1)
    rollouts = inp_toks.clone()
    
    for i_t in range(cfg.think_len): # generate thinking tokens
        think_logits = think_model(rollouts)
        think_toks = think_logits[:, -1].argmax(dim=-1, keepdim=True) + inp_max
        rollouts = t.cat([rollouts, think_toks], dim=1)

    rollout_thoughts = rollouts[:, 1:] - inp_max
    pred_logits = answer_model(rollout_thoughts)
    pred_logprobs = t.log_softmax(pred_logits[:, -1], dim=-1)
    logpbrobs_on_correct = pred_logprobs[inp_toks.squeeze(), inp_toks.squeeze()] # heh.
    preds = pred_logprobs.argmax(dim=-1).squeeze()
    acc = (preds == inp_toks.squeeze()).float().mean()
    return logpbrobs_on_correct, acc

def sweep(answer_model_cfg: SplitModelConfig, think_model_cfg: SplitModelConfig, train_func: callable, steps: int, count: int, sweep_project_name: str):
    def run_training():
        t.set_default_device(t.device("cuda"))
        t.manual_seed(42)
        random.seed(42)
        wandb.init()
        w_config = wandb.config

        training_cfg = TrainingConfig(batch_size=w_config.batch_size,group_size=w_config.group_size,lr=w_config.lr,weight_decay=1e-9,think_len=1)

        answer_model = GPT2SplitModel(answer_model_cfg)
        think_model = GPT2SplitModel(think_model_cfg)
        train_func(answer_model, think_model, training_cfg, steps=steps, display=False)

    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'acc',
            'goal': 'maximize'
        },
        'parameters': {
            'batch_size': {
                'values': [32, 64, 128, 256]
            },
            'group_size': {
                'values': [1, 16, 32, 64, 128]
            },
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=sweep_project_name)
    wandb.agent(sweep_id, function=run_training, count=count)


def train(answer_model: GPT2SplitModel, think_model: GPT2SplitModel, cfg: TrainingConfig, steps: int = 1e9):
    answer_opt = t.optim.AdamW(answer_model.parameters(), lr=cfg.answer_lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    think_opt = t.optim.AdamW(think_model.parameters(), lr=cfg.think_lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    #answer_opt = t.optim.SGD(answer_model.parameters(), lr=cfg.answer_lr, momentum=0.9, weight_decay=cfg.weight_decay)
    #think_opt = t.optim.SGD(think_model.parameters(), lr=cfg.think_lr, momentum=0.9, weight_decay=cfg.weight_decay)
    answer_model.train()
    think_model.train()

    logits_bias = 1

    run_cfg = {"answer_model": answer_model.cfg.to_dict(), "think_model": think_model.cfg.to_dict(), "training": cfg.to_dict()}
    wandb.init(project="tokenized_autoencoder", config=run_cfg, name=f"sample_bias+{logits_bias}")

    think_parameters = [p for p in think_model.parameters() if p.requires_grad]
    answer_parameters = [p for p in answer_model.parameters() if p.requires_grad]
    think_grad_norm, ans_grad_norm = 0, 0

    full_batch_size = cfg.group_size * cfg.batch_size
    #max_seq_len = think_model.cfg.seq_len - 1 - cfg.think_len
    
    pred_acc = 0.0

    inp_max = answer_model.cfg.d_vocab_out
    d_thought = think_model.cfg.d_vocab_out

    batch_indices = t.arange(cfg.batch_size, requires_grad=False)
    group_indices = t.arange(cfg.group_size, requires_grad=False)
    full_batch_indices = t.arange(full_batch_size, requires_grad=False)
    think_indices = t.arange(cfg.think_len, requires_grad=False).unsqueeze(0)

    epsilon = 1.0
    
    for b in (tr:=tqdm.trange(int(steps), ncols=140)):
        with t.inference_mode(): # generate rollouts without gradients
            inp_toks = t.randint(0, inp_max, (cfg.batch_size, 1)).reshape(-1, 1).repeat(1, cfg.group_size).reshape(full_batch_size, 1)
            rollouts = inp_toks.clone()

            for i_t in range(cfg.think_len): # generate thinking tokens
                think_logits = think_model(rollouts)
                if b < 8_000000: think_logits[full_batch_indices, :, inp_toks.flatten()] += logits_bias
                think_probs = t.softmax(think_logits[:, -1], dim=-1)
                think_toks = t.multinomial(think_probs, num_samples=1) + inp_max
                #if random.random() < epsilon:
                    #think_toks = t.randint(inp_max, inp_max+d_thought, (full_batch_size, 1))
                #else:
                    #think_toks = think_logits[:, -1].argmax(dim=-1, keepdim=True) + inp_max
                rollouts = t.cat([rollouts, think_toks], dim=1)

            rollout_thoughts = rollouts[:, 1:] - inp_max
            pred_reward_logits = answer_model(rollout_thoughts)
            #logprobs = t.log_softmax(logits[:, -1], dim=-1)
            #pred_rewards = logprobs[full_batch_indices, inp_toks.squeeze()]
            pred_rewards = pred_reward_logits[full_batch_indices, -1, inp_toks.squeeze()]

            pred_reward_mean = pred_rewards.mean().item() # mean of the predicted rewards
            pred_reward_mean_by_group = pred_rewards.reshape(cfg.batch_size, cfg.group_size).mean(dim=-1, keepdim=True)
            normed_pred_rewards = t.flatten((pred_rewards.reshape(cfg.batch_size, cfg.group_size) - pred_reward_mean_by_group))
            #normed_pred_rewards = t.clamp_min(normed_pred_rewards, 0) # normalize the rewards
            
            #normed_pred_rewards = pred_rewards.clone()
            #pred_reward_mean_by_group = pred_rewards.reshape(cfg.batch_size, cfg.group_size).mean(dim=-1, keepdim=True)
            #normed_pred_rewards = (pred_rewards.reshape(cfg.batch_size, cfg.group_size) - pred_reward_mean_by_group).flatten()
            #pred_reward_mean = normed_pred_rewards.mean().item()
            #normed_pred_rewards -= pred_reward_mean

            epsilon = max(epsilon * cfg.eps_decay, cfg.eps_min)

        rollouts = rollouts.clone()
        rollout_thoughts = rollout_thoughts.clone()
        normed_pred_rewards = normed_pred_rewards.clone()
        inp_toks = inp_toks.clone()

        pred_logits = answer_model(rollout_thoughts)
        pred_logprobs = t.log_softmax(pred_logits[:, -1], dim=-1)
        pred_loss = -pred_logprobs[full_batch_indices, inp_toks.squeeze()].mean()
        
        think_logits = think_model(rollouts)
        think_logprobs = t.log_softmax(think_logits[full_batch_indices, :-1], dim=-1)
        action_logprobs = think_logprobs[full_batch_indices.unsqueeze(-1), think_indices, rollout_thoughts]
        think_loss = (-action_logprobs * normed_pred_rewards.unsqueeze(-1)).mean()

        if b % 1 == 0:
            pred_loss.backward()
            ans_grad_norm = t.norm(t.stack([t.norm(p.grad.detach(), 2.0) for p in answer_parameters]), 2.0).item()
            answer_opt.step()
            answer_opt.zero_grad()
        
        think_loss.backward()
        think_grad_norm = t.norm(t.stack([t.norm(p.grad.detach(), 2.0) for p in think_parameters]), 2.0).item()
        if b % 1 == 0:
            think_opt.step()
            think_opt.zero_grad()

        with t.inference_mode():
            pred_prob_var = pred_reward_logits.softmax(dim=-1).var().item() # answer prob variance for logging. Tells us how much the rollout influences the answer model's final output
            pred_reward_var = pred_rewards.var().item() # variance of the predicted rewards for logging. tells us similar thing.

            wandb.log({
                "pred_reward": pred_reward_mean,
                "pred_loss": pred_loss.item(),
                "think_loss": think_loss,
                "num_think": cfg.think_len,
                "acc": pred_acc,
                "pred_reward_var": pred_reward_var,
                "pred_prob_var": pred_prob_var,
                "think_grad_norm": think_grad_norm,
                "ans_grad_norm": ans_grad_norm,
                #"prob_force_end_thought": 0.0,
                "epsilon": epsilon,
                #"think_logprobs": think_logprobs[0],
                #"entropy_reward": entropy,
            })
            tr.set_description(f"{magenta}pred loss: {pred_loss:.3f}, pred reward: {pred_reward_mean+0.01:.3f}, think loss: {think_loss:.3f}, acc: {pred_acc:.3f}")

            if b % 512 == 0:
                print()
                rollout_mean_logprob = action_logprobs.mean(dim=-1)
                preds = pred_logprobs.argmax(dim=-1).squeeze()
                for row in range(rollouts.shape[0]):
                    print(f"{blue}{rollouts[row].tolist()} {magenta}{rollout_mean_logprob[row].item():.3f} : {cyan}{pred_rewards[row].item():.3f} {green}({normed_pred_rewards[row].item():.3f}) {gray}{preds[row].item()} ({pred_logprobs[row, preds[row]]:.3f}) {endc}")

                #pred_acc = (pred_logprobs.argmax(dim=-1) == inp_toks.flatten()).float().mean().item()
                logprob_correct, pred_acc = test_acc(answer_model, think_model, cfg, inp_max)

                #t.save(answer_model.state_dict(), f"saves/add_think_fixed_blind_super_clean_split_answer{b}.pth")
                #t.save(think_model.state_dict(), f"saves/add_think_fixed_blind_super_clean_split_think{b}.pth")



INP_MAX = 64
if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    d_model = 64
    d_thought = 8
    think_len = 2
    answer_model_cfg = SplitModelConfig(
        d_model=d_model,
        seq_len=think_len,
        d_mlp=d_model*4,
        d_head=d_model//4,
        n_heads=4,
        n_layers=0,
        d_vocab_in=d_thought,
        d_vocab_out=INP_MAX,
        d_thought_vocab=d_thought
    )
    think_model_cfg = SplitModelConfig(
        d_model=d_model,
        seq_len=think_len*2,
        d_mlp=d_model*4,
        d_head=d_model//4,
        n_heads=4,
        n_layers=0,
        d_vocab_in=INP_MAX + d_thought,
        d_vocab_out=d_thought,
        d_thought_vocab=d_thought
    )

    answer_model = GPT2SplitModel(answer_model_cfg)
    think_model = GPT2SplitModel(think_model_cfg)

    training_cfg = TrainingConfig(
        think_lr=1e-4,
        answer_lr=1e-4,
        think_len=think_len,
        group_size=32,
        batch_size=64,
        eps_decay=0.99995,
        eps_min=0.01,
        weight_decay=1e-9
    )

    #train(answer_model, think_model, training_cfg, steps=1e9)
    sweep(answer_model_cfg, think_model_cfg, train, steps=1e5, count=256, sweep_project_name="tae-sweep")