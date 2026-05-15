# evaluate.py
import torch
import networkx as nx
from config import Config
from agent import MaxShotAgent
from utils import generate_ba_graphs, evaluate_agent, BaselineHDA
import time

def load_model(agent, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    agent.encoder.load_state_dict(checkpoint['encoder'])
    agent.decoder.load_state_dict(checkpoint['decoder'])
    return agent

def compare_baselines(test_graphs):
    results = {}
    # HDA
    start = time.time()
    hda_gcc_auc, hda_maxdeg_auc = evaluate_agent_baseline(BaselineHDA, test_graphs)
    results['HDA'] = (hda_gcc_auc, hda_maxdeg_auc, time.time()-start)
    # 类似添加其他基线...
    return results

def evaluate_agent_baseline(algorithm_class, graphs):
    # 基线评估函数，需根据算法实现调整
    pass

if __name__ == "__main__":
    cfg = Config()
    test_graphs = generate_ba_graphs([(50, 100)], num_per_size=100)
    agent = MaxShotAgent(cfg, 'cpu')
    agent = load_model(agent, "best_model.pt")
    gcc_auc, maxdeg_auc = evaluate_agent(agent, test_graphs)
    print(f"MaxShot: GCC AUC={gcc_auc}, MaxDeg AUC={maxdeg_auc}")