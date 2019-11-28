# coding: utf-8
# 2019/11/27 @ tongshiwei

from EduSim.Agent.utils import episode_reward_dispatch


def test_graph(kss_graph):
    graph = kss_graph
    graph.predecessors(1)
    graph.successors(1)
    assert 1 == graph.id2idx(graph.idx2id(1))


def test_reward_dispatch():
    assert len(episode_reward_dispatch(1, 20)) == 20
