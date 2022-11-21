from collections.abc import Iterable, Hashable
from typing import Dict, Any, Set, Tuple, List
from math import inf as _inf
# TODO: the imports inside functions are used to avoid circular imports,
#  either fix the imports or bring the functions here (from base_graph_algorithms)


class BaseGraph:
    _neighbors: Dict[Any, Any]
    _edges: Set[Any]
    _bi_directional: bool

    def __eq__(self, other):
        if isinstance(other, BaseGraph):
            return self._edges == other._edges and self._neighbors == other._neighbors and \
                    self._bi_directional == other._bi_directional
        return False

    def __init__(self, vertices: Iterable, edges: Iterable,
                 bi_directional: bool = False):
        """By default, creates a directed, un-weighted graph, given two iterable objects
        for vertices and edges."""
        self._neighbors = {}
        self._edges = set()
        self._bi_directional = bi_directional
        self.top_sort = None
        self.top_sort_is_relevant = False
        for v in vertices:
            self.add_vertex(v)
        for e in edges:
            self.add_unweighted_edge(*e)

    def add_vertex(self, v) -> None:
        assert isinstance(v, Hashable)
        if self._neighbors.setdefault(v, set()) != set():
            self.top_sort_is_relevant = False

    def add_unweighted_edge(self, u: Hashable, v: Hashable) -> None:
        #  this is used because of the conflict between parent and subclasses
        """Insert an un-weighted edge to a graph. If one or both of the vertices do not exist,
        add them."""
        self.top_sort_is_relevant = False
        self.add_vertex(u)
        self.add_vertex(v)
        self._edges.add((u, v))
        self._neighbors[u].add(v)
        if self._bi_directional:
            self._neighbors[v].add(u)
            self._edges.add((v, u))
            # yes it is a bit of a mess

    def add_edge(self, u: Hashable, v: Hashable) -> None:
        """Insert an edge to a graph. If one or both of the vertices do not exist,
        add them. Supports weighted edges. Edge form (directed or undirected) is
        according to the initialization of the calling graph."""
        self.add_unweighted_edge(u, v)

    def get_neighbors(self, u) -> iter:
        """Returns a generator for a given vertex' neighbor vertices.
        If input vertex isn't in the graph, adds it."""
        yield from iter(self._neighbors.setdefault(u, set()))

    def get_neighbors_without_add(self, u) -> iter:
        """Returns a generator for a given vertex' neighbor vertices.
        Does not add a non-existent vertex."""
        r = self._neighbors.get(u)
        yield from r or set()

    def remove_edge(self, u, v, bi_flag=True) -> None:
        """Remove an edge from Graph. If edge does not exist, do nothin' """
        self._edges.discard((u, v))
        self.add_vertex(u)  # make sure Christopher, I told you already!
        self._neighbors[u].discard(v)
        if self._bi_directional and bi_flag:
            self.remove_edge(v, u, False)

    def remove_vertex(self, u) -> None:
        """Remove vertex from Graph. If it doesn't exist, do nothin'"""
        self.top_sort_is_relevant = False
        s = set(self.get_neighbors(u))
        for v in s:
            self.remove_edge(u, v)
        if s:
            del self._neighbors[u]

    @property
    def n(self) -> int:
        """Number of vertices in a graph"""
        return len(self._neighbors)

    @property
    def m(self) -> int:
        """Number of edges in a graph"""
        return len(self._edges)

    @property
    def is_directed(self) -> bool:
        """Number of edges in a graph"""
        return not self._bi_directional

    def get_vertices(self) -> iter:
        yield from self._neighbors.keys()

    def get_edges(self) -> iter:
        yield from self._edges

    def __iter__(self) -> iter:
        """Iterate over each node -> yields node and its neighbors (as a set
        associated with the node in a tuple).
        If the graph is weighted, each neighbor (key)
        is associated with a weight (value) over the edge towards it."""
        yield from ((v, k) for v, k in self._neighbors.items())

    def __str__(self) -> str:
        """Returns a string representation of a graph object. For each vertex, append a
        tuple that holds the vertex data, the neighbors for that vertex.
        If graph is weighted, neighbors are shown in a dictionary - {neighbor: weight-to}."""
        s = "Graph: " if self._bi_directional else "DiGraph: "
        vertices_count = self.n
        for i, n in enumerate(self):
            s += f"{n}"
            if i == vertices_count - 1:
                return s
            s += ", "
        return s + "{ empty }"

    def dfs_visit(self, source) -> iter:
        """Returns a generator for the depth tree of a given source vertex."""
        from graph.base_graph_algorithms import dfs_visit
        yield from dfs_visit(self, source, set())

    def bfs(self, source) -> Tuple[dict, dict]:
        """Returns two dicts - prev and dist for each vertex in the graph."""
        from graph.base_graph_algorithms import _bfs
        return _bfs(self, source)

    def dfs(self) -> iter:
        """Returns a generator for a dfs-traversal **for the whole graph.**"""
        from graph.base_graph_algorithms import dfs
        yield from dfs(self)

    def bfs_traversal(self, source) -> iter:
        """Returns a generator for the breadth tree of a given source vertex."""
        from graph.base_graph_algorithms import bfs
        yield from bfs(self, source)

    def topological_sort(self) -> iter or None:
        """Returns the topological sort of a graph (if exists)."""
        from graph.base_graph_algorithms import topological_sorting
        if self.top_sort_is_relevant:
            return self.top_sort
        self.top_sort = topological_sorting(self)
        self.top_sort_is_relevant = True
        return self.top_sort

    @property
    def is_dag(self) -> bool:
        """Check if a given graph is directed-acyclic. *This is a linear time operation*"""
        from graph.base_graph_algorithms import topological_sorting
        if not self.top_sort_is_relevant:
            self.top_sort = topological_sorting(self)
        self.top_sort_is_relevant = True
        if self._bi_directional or self.top_sort is None:
            return False
        return True

    def dfs_path_from_to(self, _from, _to) -> list or None:
        """Returns the depth path between two vertices on a graph (if exists)."""
        from graph.base_graph_algorithms import dfs_path
        prev = dfs_path(self, _from)
        path = []
        tmp = _to
        while tmp is not None:
            path.append(tmp)
            tmp = prev[tmp]
        path.reverse()
        return path if path else None

    def shortest_path_from_to(self, _from, _to) -> Tuple[list or None, int or float]:
        """Returns the shortest path between two vertices, along the distance."""
        dist, prev = self.bfs(_from)
        path = [_to]
        if prev[_to] is None:
            return None, -_inf
        tmp = _to
        while prev[tmp] is not None:  # second condition was debug (?)
            tmp = prev[tmp]
            path.append(tmp)
        path.reverse()
        return path, dist[_to]

    def transpose(self):
        """Returns a BaseGraph object containing the transposed graph."""
        assert not self._bi_directional  # what's the point
        transposed_vertices = self.get_vertices()
        transposed_edges = set()
        for edge in self.get_edges():
            u, v = edge
            transposed_edges.add((v, u))
        return BaseGraph(transposed_vertices, transposed_edges, bi_directional=False)

    def strongly_connected_components(self) -> List[set]:
        """Returns a list of sets of vertices, each set is a SCC for the input graph."""
        from graph.base_graph_algorithms import scc
        assert not self._bi_directional
        return list(scc(self))