import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from env import GraphDismantleEnv
from models import GCNEncoder, DQN
from utils import nx_to_pyg_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(num_nodes, embed_dim=16):
    encoder = GCNEncoder(in_channels=1, hidden_dim=32, out_dim=embed_dim).to(device)
    dqn = DQN(num_nodes=num_nodes, embed_dim=embed_dim).to(device)
    encoder.load_state_dict(torch.load("gcn_encoder.pth", map_location=device))
    dqn.load_state_dict(torch.load("maxshot_dqn.pth", map_location=device))
    encoder.eval()
    dqn.eval()
    return encoder, dqn

def evaluate_policy(env, encoder, dqn, epsilon=0.0):
    state = env.reset()
    total_reward = 0
    steps = 0
    gcc_history = [env.initial_gcc_size]
    done = False
    while not done:
        valid_actions = list(env.graph.nodes)
        if not valid_actions:
            break
        node_feat, global_feat = state
        # 贪婪动作（epsilon=0）
        if np.random.rand() < epsilon:
            action = np.random.choice(valid_actions)
        else:
            data = nx_to_pyg_data(env.graph, node_feat).to(device)
            with torch.no_grad():
                graph_emb = encoder(data.x, data.edge_index).unsqueeze(0)
                global_feat_t = torch.FloatTensor(global_feat).unsqueeze(0).to(device)
                q_vals = dqn(graph_emb, global_feat_t).squeeze(0).cpu().numpy()
            for r in set(range(env.num_nodes)) - set(env.graph.nodes):
                q_vals[r] = -1e9
            action = int(np.argmax(q_vals))
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        state = next_state
        gcc_history.append(env._get_gcc_size())
    return gcc_history, total_reward, steps

def random_policy(env):
    state = env.reset()
    total_reward = 0
    gcc_history = [env.initial_gcc_size]
    done = False
    while not done:
        valid_actions = list(env.graph.nodes)
        if not valid_actions:
            break
        action = np.random.choice(valid_actions)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        gcc_history.append(env._get_gcc_size())
    return gcc_history, total_reward

if __name__ == "__main__":
    # 使用相同随机种子创建测试图
    NUM_NODES = 30
    test_graph = nx.erdos_renyi_graph(NUM_NODES, 0.12, seed=123)
    env = GraphDismantleEnv(test_graph)

    encoder, dqn = load_models(NUM_NODES)

    # MaxShot策略
    gcc_maxshot, reward_max, steps_max = evaluate_policy(env, encoder, dqn, epsilon=0.0)
    # 随机策略
    env.reset()
    gcc_random, reward_rand = random_policy(env)

    # 绘制瓦解曲线
    steps_max = list(range(len(gcc_maxshot)))
    steps_rand = list(range(len(gcc_random)))
    plt.figure(figsize=(8,5))
    plt.plot(steps_max, np.array(gcc_maxshot)/env.initial_gcc_size, 'b-o', label='MaxShot (trained)')
    plt.plot(steps_rand, np.array(gcc_random)/env.initial_gcc_size, 'r--s', label='Random')
    plt.xlabel("Number of removals")
    plt.ylabel("Remaining GCC ratio")
    plt.title("Graph Dismantling Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig("dismantle_curve.png")
    plt.show()
    print(f"MaxShot: total reward={reward_max:.3f}, steps={steps_max}")
    print(f"Random: total reward={reward_rand:.3f}")