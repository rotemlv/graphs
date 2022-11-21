from queue import PriorityQueue
from graph.base_graph import BaseGraph, Iterable, Hashable, _inf, Tuple


class WeightedGraph(BaseGraph):
    def __eq__(self, other):
        if super().__eq__(other):
            return self.weights_dict == other.weights_dict
        return False

    def __init__(self, vertices: Iterable, edges: Iterable,
                 bi_directional: bool = False,
                 weights_dict: dict or None = None):
        super().__init__(vertices, edges, bi_directional)
        self.weights_dict = weights_dict
        if self._bi_directional:
            for u, v in set(self.weights_dict.keys()):
                self.weights_dict[v, u] = self.weights_dict[u, v]

    def get_weights(self):
        return self.weights_dict.copy()

    def add_edge(self, u: Hashable, v: Hashable, w=None) -> None:
        """Insert an edge to a graph. If one or both of the vertices do not exist,
        add them. Supports weighted edges. Edge form (directed or undirected) is
        according to the initialization of the calling graph."""
        self.add_unweighted_edge(u, v)
        self.weights_dict[u, v] = w
        if self._bi_directional:
            self.weights_dict[v, u] = w

    def remove_edge(self, u, v, bi_flag=True) -> None:
        """Remove an edge from Graph. If edge does not exist, do nothin' """
        super().remove_edge(u, v, bi_flag)
        self.weights_dict.pop((u, v), None)
        if bi_flag and self._bi_directional:
            self.weights_dict.pop((v, u), None)

    def get_neighbors_weighted(self, u) -> iter:
        """Returns a generator for a given vertex's neighbor vertices
        alongside the weight of the edge to it.
        If input vertex isn't in the graph, adds it."""
        for v in self.get_neighbors(u):
            yield v, self.weights_dict[u, v]

    def __initialize_single_source(self, dist, prev, source):
        for n in self._neighbors.keys():
            dist[n] = _inf
            prev[n] = None
        dist[source] = 0

    @staticmethod
    def __relax(u, v, w, dist, prev):
        if dist[v] > dist[u] + w:
            dist[v] = dist[u] + w
            prev[v] = u

    def di_acyclic_shortest_path(self, source):
        """Linear time algorithm for single-source-shortest-paths in
         a directed acyclic graph."""
        # necessary pre-requisites for using this here algo
        assert self._neighbors.get(source) is not None  # check source is in graph
        assert not self._bi_directional  # if diGraph
        assert self.is_dag

        prev = {n: None for n in self.top_sort}
        dist = {n: _inf for n in self.top_sort if n != source}
        dist[source] = 0
        i = 0
        # find source in top-sort
        while self.top_sort[i] != source:
            i += 1
        # find the shortest path from source to nodes ahead of it in top-sort
        # using relaxation (and meditation) over outgoing edges
        for u in self.top_sort[i:]:
            for v in self.get_neighbors(u):
                w = self.weights_dict[u, v]
                self.__relax(u, v, w, dist, prev)
        return dist, prev

    def dijkstra(self, source):
        """Slightly modified dijkstra.
        Instead of decrease-key-ing the heap, this version starts the
        iterative process with a nearly empty heap, and adds the improved guesses
        into the heap (increasing its size) during its runtime.
        Pros: easier to write, easier on operations for some heap implementations (faster for some heaps).
        Cons: cache issues? heap size can stay at |V| for longer periods. """
        # check that source node exists in G
        assert source in self._neighbors.keys()
        # check that weights are all non-negative
        assert all(weight >= 0 for weight in self.weights_dict.values())
        # initialize
        dist, prev = {}, {}
        self.__initialize_single_source(dist, prev, source)
        visited = set()
        p_queue = PriorityQueue()
        p_queue.put((0, source))
        # while queue is non-empty
        while not p_queue.empty():
            d_u, u = p_queue.get()  # (TODO: make fib heap)
            if u not in visited:
                visited.add(u)
                for v in self.get_neighbors(u):
                    relaxed = d_u + self.weights_dict[u, v]
                    if dist[v] > relaxed:
                        dist[v] = relaxed
                        prev[v] = u
                        p_queue.put((dist[v], v))
        return dist, prev

    def bellman_ford(self, source: Hashable) -> Tuple[dict, dict] or None:
        """Standard Bellman-Ford implementation.
        This algorithm considers a bidirectional edge of negative weight as a negative circle,
        and it will return None in such cases."""
        if self._bi_directional and any(self.weights_dict[edge] < 0 for edge in self._edges):
            return None
        dist, prev = {}, {}
        self.__initialize_single_source(dist, prev, source)
        for i in range(self.n - 1):
            for edge in self._edges:
                u, v = edge
                w = self.weights_dict[edge]
                self.__relax(u, v, w, dist, prev)
        for edge in self._edges:
            u, v = edge
            if dist[v] > dist[u] + self.weights_dict[edge]:
                return None
        return dist, prev

    def bellman_ford_one_direction_neg_edges(self, source: Hashable) -> Tuple[dict, dict] or None:
        """For bi-directional graphs - a negative edge can only be traversed a single direction.
        (breaks negative circles of size 1)"""
        dist, prev = {}, {}
        self.__initialize_single_source(dist, prev, source)
        for i in range(self.n - 1):
            for edge in self._edges:
                w = self.weights_dict[edge]
                u, v = edge
                if prev[u] != v:
                    self.__relax(u, v, w, dist, prev)
        for edge in self._edges:
            u, v = edge
            w = self.weights_dict[edge]
            if prev[u] != v:
                if dist[v] > dist[u] + w:
                    return None
        return dist, prev

    def shortest_path_from_to(self, _from, _to) -> tuple:
        """Returns a tuple: a list (the path between the two vertices) and a distance (float or int)"""
        if self.is_dag:  # if a negative edges but non-negative circle exists we go bellman
            dist, prev = self.di_acyclic_shortest_path(_from)
        else:
            # try dijkstra if applicable, else try bellman ford if applicable, else just don't
            try:
                dist, prev = self.dijkstra(_from)
            except AssertionError:
                bug_check = None
                # TODO: clean this up
                try:
                    bug_check = self.bellman_ford(_from)
                except AssertionError:
                    pass  # couldn't find an elegant-ier way
                finally:
                    # also couldn't find an elegant-ier way
                    if bug_check is None:
                        print(f"No shortest path exist from node "
                              f"{_from} on a graph with negative cycles")
                        return bug_check, float('inf')
                dist, prev = bug_check
        path = [_to]
        tmp = _to
        while prev[tmp] is not None:  # second condition is debug
            tmp = prev[tmp]
            path.append(tmp)
        path.reverse()
        return path, dist[_to]

    def __insert_neighbors_to_queue(self, source: Hashable, heap: PriorityQueue) -> None:
        # place all edges from a given node inside the given heap (PriQueue)
        for target, weight in self.get_neighbors_weighted(source):
            heap.put((weight, (source, target)))

    def kruskal(self) -> set:
        """Returns a list of edges for the MST of a graph G per the Kruskal MST algorithm."""
        from disjoint_set import make_set, find, union
        assert self._bi_directional
        set_edges = set()
        # disjoint set prep
        rank, size, parent = dict(), dict(), dict()
        for node in self._neighbors.keys():  # create the disjoint sets
            make_set(node, parent, rank)
        # kruskal final prep (sort edges by weight)
        sorted_edges = sorted(self._edges, key=lambda e: self.weights_dict[e])
        # add lightest "promising" edge
        for edge in sorted_edges:
            # if sets are different
            set_of_u, set_of_v = find(edge[0], parent), find(edge[1], parent)
            if set_of_u != set_of_v:
                union(set_of_u, set_of_v, parent, rank)
                set_edges.add(edge)
        return set_edges

    def prim(self) -> set:
        """Returns a list of edges for the MST of a graph G using the Slim-Jim MST algorithm."""
        n = self.n
        # choose a tzomet kolshei
        x = list(self._neighbors.keys())[0]  # "1st" vertex
        blob: set = set()
        # tree starts from kolshei node
        blob.add(x)
        edge_set: set = set()  # no edges
        heap_queue: PriorityQueue = PriorityQueue()
        # fill heap with the first node's outgoing edges
        self.__insert_neighbors_to_queue(x, heap_queue)
        # go over the heap until blob set is full of nodes (and heap is not empty, obviously)
        # this runs at most E times, at best V-1 times
        while not heap_queue.empty() and len(blob) < n:
            weight, edge = heap_queue.get()  # log|heap|
            u, v = edge
            if v not in blob:  # should be constant time
                # add v to blob and add all edges from v to queue
                blob.add(v)
                self.__insert_neighbors_to_queue(v, heap_queue)  # log|heap| cost
                edge_set.add(edge)
        return edge_set