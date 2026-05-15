# train.py
import torch
import random
import numpy as np
from config import Config
from env import NetworkDismantlingEnv
from agent import MaxShotAgent
from buffer import ReplayBuffer
from utils import generate_ba_graphs, evaluate_agent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # 生成训练图
    print("Generating training graphs...")
    train_graphs = generate_ba_graphs(cfg.train_graph_sizes, m=cfg.ba_m, num_per_size=cfg.train_graphs_per_size)
    print(f"Total training graphs: {len(train_graphs)}")

    # 生成验证图（评估用）
    val_graphs = generate_ba_graphs([(50, 50)], m=cfg.ba_m, num_per_size=cfg.eval_episodes)

    agent = MaxShotAgent(cfg, device)
    buffer = ReplayBuffer(cfg.buffer_capacity)

    # 记录
    best_eval_gcc = float('inf')

    for episode in range(1, cfg.total_episodes + 1):
        # 随机选取一个图开始新回合
        G = random.choice(train_graphs).copy()
        env = NetworkDismantlingEnv(G)
        state = env.reset()
        done = False
        step = 0
        episode_loss = 0
        while not done and step < cfg.max_steps_per_episode:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = agent.act(state, valid)
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            if len(buffer) >= cfg.batch_size:
                batch = buffer.sample(cfg.batch_size)
                loss = agent.train_step(batch)
                episode_loss += loss
            state = next_state
            step += 1

        agent.update_epsilon()

        # 定期评估
        if episode % cfg.eval_freq == 0:
            mean_gcc, mean_maxdeg = evaluate_agent(agent, val_graphs)
            print(
                f"Episode {episode}: Eval GCC AUC = {mean_gcc:.4f}, MaxDeg AUC = {mean_maxdeg:.4f}, epsilon={agent.epsilon:.3f}")
            if mean_gcc < best_eval_gcc:
                best_eval_gcc = mean_gcc
                torch.save({
                    'encoder': agent.encoder.state_dict(),
                    'decoder': agent.decoder.state_dict(),
                }, "best_model.pt")
                print("  -> New best model saved.")

    print("Training finished.")


if __name__ == "__main__":
    train()