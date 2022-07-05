from collections.abc import Sized
from graph.base_graph import BaseGraph, Iterable
from graph.weighted_graph import WeightedGraph
from graph.network_graph import FlowNetwork
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    @staticmethod
    def create(vertices: Iterable = None, edges: Sized = None,
               bi_dir: bool = False, weighted: bool = False,
               weight_dict: dict or None = None) -> BaseGraph or WeightedGraph:
        """Wrapper class for the two graph classes. Allows creating a graph
        (weighted or un-weighted) using one AIO function.
        This function can handle any sensible graph input."""
        # fix input to accept pretty much any sensible format (V)
        assert vertices is None or isinstance(vertices, Iterable)
        assert edges is None or isinstance(edges, Iterable)
        if edges is None:
            if weight_dict is not None:
                edges = weight_dict.keys()
            else:
                edges = set()
        if weighted and weight_dict is None and edges is not None:
            weight_dict = {edge: 1 for edge in edges}
        if vertices is None:
            vertices = set()
        # return the correct graph class
        if weighted:
            return WeightedGraph(vertices, edges, bi_dir, weights_dict=weight_dict)
        else:
            return BaseGraph(vertices, edges, bi_dir)


def generate_graph_plot(edges, weights=None, directed=True, xlabel="graph", g_name="Template graph"):
    """Directed only currently"""
    # fig = plt.figure()
    if directed:
        g = nx.DiGraph(name=g_name)  # Generate a Networkx object
    else:
        g = nx.Graph(name=g_name)
    ed_ls = sorted(edges)
    weights = weights if weights else {edge: 1 for edge in edges}  # too tired
    G = nx.DiGraph()
    G.add_weighted_edges_from(((*edge, weights[edge]) for edge in ed_ls))
    g = G
    # if drowsy
    # nx.draw_kamada_kawai(g)
    # nx.draw(g,pos=nx.spring_layout(g))
    # pos = nx.kamada_kawai_layout(g)
    pos = nx.kamada_kawai_layout(g)
    # if sleepy
    # pos = nx.nx_agraph.graphviz_layout(G)
    nx.draw_networkx(g, pos, arrows=directed)
    labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
    # if very sleepy
    # nx.draw(g,nx.spring_layout(g),with_labels=True)
    plt.xlabel(xlabel)
    plt.show()


def graph_a():
    # example 1 - un-weighted graph
    #   1 ---> 2<   /--->7
    #   |      ^ \ /    ^
    #   V     /   |    /
    #   3-->4 --->5--->6
    #
    V = [1, 2, 3, 4, 5, 6, 7]
    E = [(1, 2), (1, 3), (3, 4), (4, 2), (4, 5), (5, 6), (5, 7), (6, 7), (5, 2)]
    g = Graph.create(V, E)


def test_2():
    # example 2 - weighted graph (not shown in drawing)

    #   1 <--- 2<   /--->7
    #   |      ^ \ /    ^
    #   V     /   |    /
    #   3-->4 --->5--->6

    # we can call .create() without vertices or edges, just with the weights dictionary
    # V = [1, 2, 3, 4, 5, 6, 7]
    # E = [(2, 1), (1, 3), (3, 4), (4, 2), (4, 5), (5, 6), (5, 7), (6, 7), (5, 2)]
    w = {(1, 2): 1, (1, 3): 3, (2, 4): 1, (2, 3): 4, (4, 5): 1, (5, 6): 2, (6, 7): 1, (5, 2): 0}
    # g = Graph.create(V, E, False, True, {(1, 2):2, (2, 3):3})
    g = Graph.create(bi_dir=False, weighted=True, weight_dict=w)
    print(g)
    print(g.strongly_connected_components())
    print(g.dfs_path_from_to(1, 7))
    print(g.shortest_path_from_to(1, 7))
    x = g.shortest_path_from_to(1, 7)
    l = x[0]
    for i in range(1, len(l)):
        print(f"edge: {l[i - 1], l[i]}, weight: {g.weights_dict[l[i - 1], l[i]]}")

    # plot the ting
    generate_graph_plot(edges=g._edges, weights=g.weights_dict)


# one could even create an empty graph without passing anything
g = Graph.create(bi_dir=True, weighted=True)
g.add_edge(1, 2, 1)  # u, v, weight
g.add_edge(2, 4, 2)
g.add_edge(3, 4, 3)
g.add_edge(1, 5, 1)
g.add_edge(2, 3, 1)
g.add_edge(1, 4, 4)
g.add_edge(2, 5, 5)
# print(g.weights_dict)
# generate_graph_plot(g.edges, g.weights_dict)
print(f'minimal spanning tree - {g.kruskal()=}, {g.prim()=}')
t = Graph.create(bi_dir=True, weighted=True)
# t.add_edge(1, 2, 1)  # u, v, weight
# t.add_edge(2, 4, 2)
# t.add_edge(3, 4, 3)
# t.add_edge(1, 5, 1)
# t.add_edge(2, 3, 1)
# t.add_edge(1, 4, 4)
# t.add_edge(2, 5, 4.99999999999999989)
# print(g == t)
# print(t.weights_dict, g.weights_dict)