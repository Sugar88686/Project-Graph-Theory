import os
import time
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from env import NetworkDismantlingEnv
from agent import MaxShotAgent
from buffer import ReplayBuffer
from utils import generate_ba_graphs, evaluate_agent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")
    if device.type == 'cuda':
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    run_name = f"MaxShot_Train_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print("=" * 50)
    print(f"[*] TensorBoard 日志目录: {log_dir}")
    print("=" * 50)

    print("[*] 正在生成训练图和验证图...")
    train_graphs = generate_ba_graphs(cfg.train_graph_sizes, m=cfg.ba_m, num_per_size=cfg.train_graphs_per_size)
    val_graphs = generate_ba_graphs(cfg.train_graph_sizes, m=cfg.ba_m, num_per_size=cfg.eval_episodes)
    print(f"[*] 训练图数量: {len(train_graphs)} | 验证图数量: {len(val_graphs)}")

    agent = MaxShotAgent(cfg, device)
    buffer = ReplayBuffer(cfg.buffer_capacity)

    best_eval_gcc = float('inf')
    start_time = time.time()

    print("========== 开始训练 ==========")
    pbar = tqdm(range(1, cfg.total_episodes + 1), desc="Training")

    for episode in pbar:
        G = random.choice(train_graphs).copy()
        env = NetworkDismantlingEnv(G)

        # 【提速优化】：接收预计算的 PyG 数据
        state_nx, state_pyg = env.reset()
        done = False
        step = 0
        episode_loss = 0
        train_steps = 0
        episode_reward = 0

        while not done and step < cfg.max_steps_per_episode:
            valid_nodes = env.get_valid_actions()
            if not valid_nodes:
                break

            # 【提速优化】：传入预计算的 PyG 数据
            action = agent.act(state_nx, state_pyg, valid_nodes)
            if action is None:
                break

            (next_state_nx, next_state_pyg), reward, done = env.step(action)

            buffer.push(state_nx, state_pyg, action, reward, next_state_nx, next_state_pyg, done)
            episode_reward += reward

            # 【提速优化】：加入 train_freq 控制，每 N 步才更新一次网络
            if len(buffer) >= cfg.batch_size and step % cfg.train_freq == 0:
                batch = buffer.sample(cfg.batch_size)
                loss = agent.train_step(batch)
                episode_loss += loss
                train_steps += 1

            state_nx = next_state_nx
            state_pyg = next_state_pyg
            step += 1

        agent.update_epsilon()
        avg_loss = (episode_loss / train_steps) if train_steps > 0 else 0.0

        pbar.set_postfix({
            'Loss': f"{avg_loss:.4f}",
            'Eps': f"{agent.epsilon:.3f}",
            'Reward': f"{episode_reward:.2f}"
        })

        writer.add_scalar('Train/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Train/Epsilon', agent.epsilon, episode)
        writer.add_scalar('Train/Steps', step, episode)
        if train_steps > 0:
            writer.add_scalar('Train/Average_Loss', avg_loss, episode)

        if episode % cfg.eval_freq == 0:
            tqdm.write(f"\n>>> 正在进行多进程评估 ({len(val_graphs)}张图, {cfg.eval_num_workers}个进程)...")

            # 【修改点】：传入 cfg 以支持多进程
            mean_gcc, mean_maxdeg = evaluate_agent(agent, val_graphs, cfg)

            writer.add_scalar('Eval/GCC_AUC', mean_gcc, episode)
            writer.add_scalar('Eval/MaxDeg_AUC', mean_maxdeg, episode)
            writer.flush()

            elapsed = (time.time() - start_time) / 3600
            tqdm.write(
                f"=== Episode: {episode} | Time: {elapsed:.2f}h | "
                f"GCC AUC: {mean_gcc:.4f} | MaxDeg AUC: {mean_maxdeg:.4f} ==="
            )

            if mean_gcc < best_eval_gcc:
                best_eval_gcc = mean_gcc
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    'encoder': agent.encoder.state_dict(),
                    'decoder': agent.decoder.state_dict(),
                    'episode': episode,
                    'gcc_auc': mean_gcc,
                }, "checkpoints/best_model.pth")
                tqdm.write(
                    f"[*] 发现更好模型！已保存至 checkpoints/best_model.pth (当前最佳 GCC AUC: {best_eval_gcc:.4f})")

    writer.close()
    print("========== 训练结束 ==========")
    print(f"总耗时: {(time.time() - start_time) / 3600:.2f} 小时")
    print(f"最佳 GCC AUC: {best_eval_gcc:.4f}")


if __name__ == "__main__":
    # 解决 Windows/Linux 下多进程可能引发的 CUDA 初始化报错问题
    import multiprocessing

    multiprocessing.set_start_method('spawn', force=True)
    train()
