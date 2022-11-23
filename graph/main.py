import copy
from collections.abc import Sized
from graph.base_graph import BaseGraph, Iterable
from graph.weighted_graph import WeightedGraph
from graph.network_graph import FlowNetwork
import networkx as nx
import matplotlib.pyplot as plt

from random import randint


# import scipy

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

    # G = nx.DiGraph()
    # print(weights, [edge for edge in edges])
    # for x in ((*edge, weights[edge]) for edge in ed_ls):
    #     print(x)
    # exit()
    g.add_weighted_edges_from(((*edge, weights[edge]) for edge in ed_ls))
    # g = g
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


def generate_graph_plot_wrapper(g: BaseGraph or WeightedGraph):
    if isinstance(g, WeightedGraph):
        generate_graph_plot(edges=list(g.get_edges()), weights=g.get_weights(), directed=g.is_directed)
    elif isinstance(g, BaseGraph):
        generate_graph_plot(edges=list(g.get_edges()), directed=g.is_directed)
    else:
        raise AttributeError(f"Error plotting {g.__class__}")


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


def main_test():
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


def new_test():
    vertices = [i for i in range(1, 20)]
    edges = set((randint(1, 20), randint(1, 20)) for _ in range(30))
    edges = {(u, v) for u, v in edges if u != v}
    weights = {edge: randint(1, 10) for edge in edges}
    # print(edges)
    g = Graph.create(vertices=vertices, weight_dict=weights, edges=edges, weighted=True, bi_dir=True)
    # print(g)

    for node in g.get_vertices():
        p = g.shortest_path_from_to(1, node)
        if p[0] is not None:
            print(f"Path from {1} to {node}: {p[0]}, distance: {p[1]}")
    generate_graph_plot_wrapper(g)
    # print(f"Graph g type: {g.__class__}")
    generate_graph_plot(edges=g.kruskal(), weights=g.get_weights(),
                        directed=False, g_name="Minimal spanning tree")
    # generate_graph_plot_wrapper(g)
    # generate_graph_plot([e for e in g.get_edges()])
    # print([e for e in g.get_edges()])


def plot_graph_with_shortest_path(graph: WeightedGraph or BaseGraph, path: tuple):
    """Show a graph with the path on it in a different color
    * works better without weights for now"""
    graph_copy = copy.copy(graph)
    if isinstance(graph, BaseGraph):
        graph_copy = WeightedGraph(graph.get_vertices(), graph.get_edges(),
                                   not graph.is_directed, {(u, v): 1 for u, v in graph.get_edges()})
    # Generate a Networkx object
    if graph_copy.is_directed:
        g = nx.DiGraph(name="test")
    else:
        g = nx.Graph(name="test")
    ed_ls = sorted(graph_copy.get_edges())
    weights = graph_copy.get_weights()

    nodes_on_path = set(path)
    nodes_not_on_path = set(graph_copy.get_vertices()) - nodes_on_path

    edges_for_path = set([(path[i], path[i + 1]) for i in range(len(path) - 1)])
    if not graph_copy.is_directed:
        edges_for_path = edges_for_path.union({(v, u) for u, v in edges_for_path})

    edges_not_on_path = set(graph_copy.get_edges()) - edges_for_path

    g.add_weighted_edges_from(((*edge, weights[edge]) for edge in ed_ls))

    # draw the thing
    # pos = nx.kamada_kawai_layout(g)
    pos = nx.spring_layout(g, seed=3113794652)  # positions for all nodes

    # nodes
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    nx.draw_networkx_nodes(g, pos, nodelist=list(nodes_on_path), node_color="tab:red", **options)
    nx.draw_networkx_nodes(g, pos, nodelist=list(nodes_not_on_path), node_color="tab:blue", **options)

    # edges
    nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=list(edges_for_path),
        width=8,
        alpha=0.5,
        edge_color="tab:red",
    )
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=list(edges_not_on_path),
        width=8,
        alpha=0.5,
        edge_color="tab:blue",
    )
    nx.draw_networkx_labels(g, pos, labels=None, font_size=22, font_color="whitesmoke")

    # plt stuff
    plt.tight_layout()
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    # new_test()
    # g = BaseGraph([], [(1, 2), (2, 3), (1, 3), (1, 4), (4, 5)], bi_directional=True)
    g = WeightedGraph([], [(1, 2), (2, 3), (1, 3), (1, 4), (4, 5)], bi_directional=True,
                      weights_dict={(1, 2): 3, (2, 3): 2, (1, 3): 4, (1, 4): 0, (4, 5): 5})
    plot_graph_with_shortest_path(g, g.shortest_path_from_to(1, 5)[0])
