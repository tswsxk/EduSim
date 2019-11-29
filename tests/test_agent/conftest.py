# coding: utf-8
# 2019/11/27 @ tongshiwei

import pytest
from EduSim import Graph, RandomGraphAgent


@pytest.fixture(scope="module")
def kss_graph():
    return Graph("KSS")


@pytest.fixture(scope="module")
def random_agent(kss_graph):
    return RandomGraphAgent(kss_graph)
