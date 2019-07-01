# coding: utf-8
# create by tongshiwei on 2019/7/1
import random
import os

import networkx as nx

__all__ = ["RandomAgent", "Agent", "Graph"]


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

        self.initial_connected_graph()

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

    def parents(self, idx):
        return list(self.predecessors(idx))

    def grandparents(self, idx):
        gp = []

        for node in self.parents(idx):
            gp += self.predecessors(node)

        return gp

    def parents_siblings(self, idx):
        ps = []

        for node in self.grandparents(idx):
            ps += self.graph.successors(node)

        return ps

    def initial_connected_graph(self):
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


class Agent(object):
    def __init__(self, graph):
        """

        Parameters
        ----------
        graph: Graph
        """
        self.graph = graph
        self.path = None

    def begin_episode(self, *args, **kwargs):
        raise NotImplementedError

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def end_episode(self, *args, **kwargs):
        raise NotImplementedError

    def state_transform(self, action, correct, add_to_path=True, *args, **kwargs):
        raise NotImplementedError

    def is_terminal_action(self, action):
        return False

    @staticmethod
    def rewards(reward_values):
        return reward_values


class RandomAgent(Agent):
    def begin_episode(self, *args, **kwargs):
        self.path = []

    def step(self, *args, **kwargs):
        return random.choice(self.graph()[0]), None

    def end_episode(self, *args, **kwargs):
        self.path = None

    def state_transform(self, action, correct, *args, **kwargs):
        self.path.append(action)
