# coding: utf-8
# create by tongshiwei on 2019/7/1

from EduSim import RandomAgent, Graph


def test_graph():
    Graph(dataset="KSS")


def test_agent():
    agent = RandomAgent(Graph(dataset="KSS"))
    action, q = agent.step()
    assert isinstance(action, int)
