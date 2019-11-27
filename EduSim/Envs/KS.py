# coding: utf-8
# 2019/11/26 @ tongshiwei

import networkx as nx


class KS(nx.DiGraph):
    def dump_id2idx(self, filename):
        with open(filename, "w") as wf:
            for node in self.nodes:
                print("%s,%s" % (node, node), file=wf)

    def dump_graph_edges(self, filename):
        with open(filename, "w") as wf:
            for edge in self.edges:
                print("%s,%s" % edge, file=wf)
