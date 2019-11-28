# coding: utf-8
# 2019/11/26 @ tongshiwei
import os
import networkx as nx

__all__ = ["Graph"]


def load_graph(filename, graph_nodes_num=None):
    graph = nx.DiGraph()

    if graph_nodes_num is not None:
        graph.add_nodes_from(range(graph_nodes_num))

    with open(filename) as f:
        edges = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

    graph.add_edges_from(edges)
    return graph


def load_id2idx(filename):
    id2idx = {}
    with open(filename) as f:
        for line in f:
            if line.strip():
                vid, idx = line.strip().split(',')
                id2idx[vid] = int(idx)

    return id2idx


def load_idx2id(filename):
    idx2id = {}
    with open(filename) as f:
        for line in f:
            if line.strip():
                vid, idx = line.strip().split(',')
                idx2id[int(idx)] = vid

    return idx2id


class Graph(object):
    def __init__(self, dataset=None, graph_nodes_num=None, disable=False):
        filename = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../meta_data/%s/data/graph_edges.idx" % dataset))
        id2idx_filename = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../meta_data/%s/data/vertex_id2idx" % dataset))

        self._id2idx = load_id2idx(id2idx_filename) if id2idx_filename is not None else None
        self._idx2id = load_idx2id(id2idx_filename) if id2idx_filename is not None else None

        _graph_nodes_num = max(self._id2idx.values())
        assert _graph_nodes_num == max(self._idx2id.keys())

        self.graph = load_graph(filename, graph_nodes_num if graph_nodes_num is not None else _graph_nodes_num)

        self.no_pre = [node for node in self.nodes if not list(self.graph.predecessors(node))]

        self.disable = disable

        self.connected_graph = {}

        self.path_table = nx.shortest_path(self.graph)

        self.__initial_connected_graph()

    def __call__(self, *args, **kwargs):
        return list(self.nodes), []

    @property
    def nodes(self):
        return self.graph.nodes

    def id2idx(self, vid):
        return self._id2idx[vid]

    def idx2id(self, idx):
        return self._idx2id[idx]

    def predecessors(self, idx):
        return list(self.graph.predecessors(idx))

    def successors(self, idx):
        return list(self.graph.successors(idx))

    def __initial_connected_graph(self):
        for node in self.graph.nodes:
            if node in self.connected_graph:
                continue
            else:
                queue = [node]
                _connected_graph = set()
                while queue:
                    visit = queue.pop()
                    if visit not in _connected_graph:
                        _connected_graph.add(visit)
                        queue.extend(self.predecessors(visit))
                        queue.extend(self.successors(visit))
                for node in _connected_graph:
                    self.connected_graph[node] = _connected_graph


def episode_reward_dispatch(episode_reward: (float, int), step_num: int, step_rewards=None) -> list:
    """To dispatch the episode reward to each step"""

    def global_reward_decay(_global_reward):
        _global_reward *= 0.99
        return _global_reward

    dispatched_rewards = [0] * step_num
    step_rewards = [0] * step_num if step_rewards is None else step_rewards

    assert len(dispatched_rewards) == len(step_rewards)

    reward = episode_reward

    for i in range(step_num - 1, -1, -1):
        reward = reward + step_rewards[i]
        dispatched_rewards[i] = reward
        reward = global_reward_decay(reward)

    return dispatched_rewards
