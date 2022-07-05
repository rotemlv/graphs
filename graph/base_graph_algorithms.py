from queue import SimpleQueue
from math import inf as _inf
from graph.base_graph import BaseGraph


def dfs_visit(graph: BaseGraph, s, visited: set):
    """helper method for DFS - visit a node and its neighbors recursively"""
    if s not in visited:
        visited.add(s)
        yield s
        for u in graph.get_neighbors(s):
            if u not in visited:
                yield from dfs_visit(graph, u, visited)


def dfs(graph: BaseGraph):
    """Very basic DFS *traversal* of a graphs (generator)"""
    visited = set()
    for node in graph.get_vertices():
        # node is source, targets (as a list), and weights
        # just go over nodes
        yield from dfs_visit(graph, node, visited)


def dfs_forests(graph: BaseGraph) -> set:
    """Returns a set of sets of nodes. Nodes are separated as the different DFS trees
    for the graphs G (the calling object)."""
    visited = set()
    forests = set()
    for node in graph.get_vertices():
        # node is source, targets (as a list), and weights
        # just go over nodes
        s = frozenset(dfs_visit(graph, node, visited))
        if len(s) > 0:  # set is not empty
            forests.add(s)
    return forests  # frozen set for now


def dfs_forests_ordered(graph: BaseGraph, ordered_nodes) -> set:
    """Used for strongly connected components - insert the ordered nodes from _dfs_scc's finished"""
    visited = set()
    forests = set()
    for node in ordered_nodes:
        # node is source, targets (as a list), and weights
        # just go over nodes
        s = frozenset(dfs_visit(graph, node, visited))
        if len(s) > 0:  # set is not empty
            forests.add(s)
    return forests  # set of frozen set for now


def dfs_visit_path(input_graph: BaseGraph, s, color, prev, discovered, finished):
    """helper method for DFS - visit a node and its neighbors recursively"""
    color[s] = 'r'
    for neighbor in input_graph.get_neighbors(s):
        if color[neighbor] == 'w':
            prev[neighbor] = s
            dfs_visit_path(input_graph, neighbor, color, prev, discovered, finished)
    color[s] = 'b'


def dfs_path(input_graph, _from):
    # initialize dicts
    _color, _prev, _discovered, _finished = dict(), dict(), dict(), dict()
    for v in input_graph.get_vertices():
        _color[v] = 'w'
        _prev[v] = None
    for node in input_graph.get_vertices():
        # if color is white
        if _color[node] == 'w':
            dfs_visit_path(input_graph, node, _color, _prev, _discovered, _finished)
    return _prev


def _dfs_scc(graph: BaseGraph):
    """helper function for the strong-connected-component finding function"""
    time = 0  # initialize time to 0

    def _dfs_visit_scc(input_graph: BaseGraph, s,  # s is the vertex
                       color: dict, prev: dict,
                       discovered: dict, finished: dict) -> None:
        """inner helper function - visit a node and its neighbors recursively"""
        nonlocal time
        time += 1
        discovered[s] = time
        color[s] = 'r'
        for neighbor in input_graph.get_neighbors(s):
            if color[neighbor] == 'w':
                prev[neighbor] = s
                _dfs_visit_scc(input_graph, neighbor, color, prev, discovered, finished)
        color[s] = 'b'
        time += 1
        finished[s] = time

    # create & initialize dicts
    _color, _prev, _discovered, _finished = dict(), dict(), dict(), dict()
    for v in graph.get_vertices():
        _color[v] = 'w'
        _prev[v] = None
    for node in graph.get_vertices():
        # if color is white
        if _color[node] == 'w':
            _dfs_visit_scc(graph, node, _color, _prev, _discovered, _finished)
    return _finished


def scc(graph: BaseGraph):
    # get the dfs finish time for each node in the graph
    finish = _dfs_scc(graph)
    # sort nodes by it (descending!)
    lst = [k for k, _ in sorted(finish.items(), key=lambda item: -item[1])]
    # get the transposed graph
    transposed = graph.transpose()
    # get each dfs-forest in the ordered dfs run on the transposed graph as an SCC
    yield from (set(x) for x in dfs_forests_ordered(transposed, lst))


def _dfs_visit_top_sort(graph: BaseGraph, node,
                        color, res, idx, is_circle=False) -> tuple:
    # my standard implementation for DFS-Visit for top-sort
    color[node] = 'r'
    for neighbor in graph.get_neighbors(node):
        if color[neighbor] == 'w':
            idx, is_circle = _dfs_visit_top_sort(graph, neighbor, color, res, idx)
        elif color[neighbor] == 'r':
            is_circle = True  # dfs fact
            break
    color[node] = 'b'
    res[idx] = node
    return idx - 1, is_circle


def topological_sorting(graph: BaseGraph) -> list or None:
    """Returns a list of nodes in a graphs, in an order fitting a topological sort
    of the graphs [last finished in DFS -> first in list]"""
    res = [0 for _ in range(graph.n)]  # list to hold the ordered elements of the graph
    color = {n: 'w' for n in graph.get_vertices()}
    j = graph.n - 1  # start at the end of the list and go back (as if it's a linked list).
    for node in graph.get_vertices():
        if color[node] == 'w':
            j, is_circle = _dfs_visit_top_sort(graph, node, color, res, j)
            if is_circle:
                return None  # no possible ordering
    return res


def bfs_og(graph: BaseGraph, source) -> iter:
    """Return a generator for a breadth first traversal over the given graph,
    from a given source node - to all reachable nodes."""
    prev = {}
    dist = {}
    color = {}
    for node in graph.get_vertices():
        prev[node] = None
        dist[node] = _inf
        color[node] = 'w'
    dist[source] = 0
    color[source] = 'r'
    queue = SimpleQueue()
    queue.put(source)
    while not queue.empty():
        v = queue.get()
        yield v  # yields rather than returns (for now), can be modified
        for u in graph.get_neighbors(v):
            if color[u] == 'w':
                prev[u] = v
                color[u] = 'r'
                dist[u] = 1 + dist[v]
                queue.put(u)
        color[v] = 'b'


def bfs(graph: BaseGraph, source) -> iter:
    """light-version:\n
    Return a generator for a breadth first traversal over the given graph,
    from a given source node - to all reachable nodes."""
    visited = set()
    visited.add(source)
    queue = SimpleQueue()
    queue.put(source)
    while not queue.empty():
        v = queue.get()
        yield v  # yields rather than returns (for now), can be modified
        for u in graph.get_neighbors(v):
            if u not in visited:
                queue.put(u)


def _bfs(graph: BaseGraph, source) -> tuple:
    """Good old textbook implementation of breadth-first-search"""
    # shortest path bfs version (returns distances and previous for each node)
    prev = {}
    dist = {}
    color = {}
    for node in graph.get_vertices():
        prev[node] = None
        dist[node] = _inf
        color[node] = 'w'
    dist[source] = 0
    color[source] = 'r'
    queue = SimpleQueue()
    queue.put(source)
    while not queue.empty():
        v = queue.get()
        for u in graph.get_neighbors(v):
            if color[u] == 'w':
                prev[u] = v
                color[u] = 'r'
                dist[u] = 1 + dist[v]
                queue.put(u)
        color[v] = 'b'
    # TODO: weighted uses prev,dist format, it (works fine but) looks ugly now
    return dist, prev