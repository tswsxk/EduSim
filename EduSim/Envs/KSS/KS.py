# coding: utf-8
# 2020/5/8 @ tongshiwei

from EduSim.Envs.shared.KSS_KES import KS

graph_edges = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (2, 8),
    (3, 4),
    (4, 8),
    (5, 4),
    (5, 9),
    (6, 7),
    (7, 8),
    (8, 9),
]


def get_knowledge_structure():
    ks = KS()
    ks.add_edges_from(graph_edges)
    return ks
