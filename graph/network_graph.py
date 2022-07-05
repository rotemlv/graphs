from copy import deepcopy
from graph.base_graph import BaseGraph, Iterable, Hashable


class FlowNetwork(BaseGraph):
    def __init__(self, vertices: Iterable, edges: Iterable,
                 source: Hashable, target: Hashable, n_capacity: dict,
                 n_flow: dict or None = None, bi_directional: bool = False):
        if bi_directional:
            print(f"graph < {vertices=}, {edges=} > might be problematic!\n")
        super().__init__(vertices, edges, bi_directional)
        self.capacities_dict: dict = n_capacity
        if n_flow is None:
            self.flows_dict: dict = {edge: 0 for edge in self._edges}
        else:
            self.flows_dict: dict = n_flow
        # check for fucker behavior
        assert source in self.get_vertices()
        assert target in self.get_vertices()
        self.source = source
        self.target = target

    def get_edge_flow(self, edge) -> int or float:
        u, v = edge
        forwards = self.flows_dict.get((u, v))
        if forwards is None:
            backwards = self.flows_dict.get((v, u), 0)  # negate the flow in the opposite direction
            return - backwards
        return forwards

    def get_edge_capacity(self, edge) -> int or float:
        forwards = self.capacities_dict.get(edge)
        # capacity is zero if edge does not exist, no one cares about the other direction here
        return forwards or 0

    def get_vertex_flow(self, vertex) -> int or float:
        """Now it's actually all fine"""
        sum_out = sum(self.get_edge_flow((vertex, u)) for u in self.get_neighbors(vertex))
        sum_in = sum(self.get_edge_flow((u, vertex)) for u in self.get_vertices())
        if vertex == self.source:
            return sum_out
        if vertex == self.target:
            return sum_in
        return sum_in + sum_out

    def get_vertex_capacity(self, vertex) -> None or int or float:
        """Get the sum of the capacities of all the edges leaving the vertex.
        Technically edges entering the vertex are also cunt-ed, but these are all 0s so nope."""
        x = 0
        for u in self.get_neighbors(vertex):
            x += self.get_edge_capacity((vertex, u))
        return x

    def check_legal_network(self) -> bool:
        """
        This method checks the current state of the calling FlowNetwork.
        If a certain node breaks the conservation axiom of the network definition,
        or if the source node releases a different amount than the target node receives -
        returns False (else, True).
        * There are additional tests to do *
        """
        for v in self.get_vertices():
            if v == self.source and \
                    self.get_vertex_flow(v) != - self.get_vertex_flow(self.target):
                return False
            elif v != self.target and self.get_vertex_flow(v) != 0:
                return False
        return True

    def set_flow(self, _from, _to, new_flow):
        """Unsafe for the 3 axioms of a flow network - no loss of flow check etc.
        Set an edge's flow to the new given. If new flow > capacity, throws assertion exception."""
        assert (_from, _to) in self._edges or (_to, _from) in self._edges
        assert new_flow <= self.get_edge_capacity((_from, _to))
        self.flows_dict[_from, _to] = new_flow
        self.flows_dict[_to, _from] = - new_flow

    def set_flow_with_propagation(self, _from, _to, new_flow):
        """Set flow on an edge. If new flow is im-balanced, propagate the excess
        all the way to the source and target nodes.
        Throws assertion exception if you did something wrong with the capacity limits of this edge.
        (**UN-SAFE if you don't consider capacity "around" edge**)"""
        assert (_from, _to) in self.capacities_dict
        assert new_flow <= self.get_edge_capacity((_from, _to))
        self.flows_dict[_from, _to] = new_flow
        self.flows_dict[_to, _from] = - new_flow
        # this part is linear time if valid input
        if not self.check_legal_network():
            # TODO: This part ain't safe! There's no check for over-flow (it will throw an exception)
            # find path from edge (_from) to source
            path1, _ = self.shortest_path_from_to(self.source, _from)
            # ditto for _to and target
            path2, _ = self.shortest_path_from_to(_to, self.target)
            if path1 is not None:
                for i in range(1, len(path1)):
                    self.set_flow(path1[i - 1], path1[i], new_flow)
            if path2 is not None:
                for i in range(1, len(path2)):
                    self.set_flow(path2[i - 1], path2[i], new_flow)

    def set_capacity_with_prop(self, _from, _to, new_cap: int or float):
        assert (_from, _to) in self._edges
        self.capacities_dict[_from, _to] = new_cap
        self.capacities_dict[_to, _from] = - new_cap
        curr_flow = self.get_edge_flow((_from, _to))
        if curr_flow > new_cap:
            # correct the flow and such
            new_form = new_cap - curr_flow  # will be negative if nonsense
            # add the new_form along some path and so on
            self.set_flow_with_propagation(_from, _to, curr_flow + new_form)

    def _build_residual_network(self) -> BaseGraph:
        """Returns a BaseGraph object for the residual network.
        Contains the (un-weighted) edges on which the residual capacity is non-zero,
        and the same vertices as the calling flow network.
        Use this WITH the flow network for future calculations."""
        # natural edges
        e1 = {edge for edge in self._edges
              if self.get_edge_capacity(edge) - self.get_edge_flow(edge) > 0}
        # opposite direction
        e2 = {(v, u) for u, v in self._edges
              if self.get_edge_capacity((v, u)) - self.get_edge_flow((v, u)) > 0}
        # just grab all edges with non-zero Cf
        return BaseGraph(self.get_vertices(), e1.union(e2), bi_directional=False)

    def ford_fulkerson(self):
        """BFS version"""
        residual_network = self._build_residual_network()
        # can just modify the call to return edges
        path, _ = residual_network.shortest_path_from_to(self.source, self.target)
        # check if path exists and stuff
        min_residual = float('inf')
        while path is not None:
            for i in range(1, len(path)):
                edge = path[i - 1], path[i]
                residual = self.get_edge_capacity(edge) - self.get_edge_flow(edge)
                if residual < min_residual:
                    min_residual = residual
            # now, increase flow by that min on every edge on path
            for i in range(1, len(path)):
                u, v = path[i - 1], path[i]
                self.set_flow(u, v, self.get_edge_flow((u, v)) + min_residual)
            # re-build and stuff
            residual_network = self._build_residual_network()
            path, _ = residual_network.shortest_path_from_to(self.source, self.target)

    def ford_fulkerson_dfs(self):
        """."""
        residual_network = self._build_residual_network()
        # can just modify the call to return edges
        # path = dfs
        path, _ = residual_network.dfs_path_from_to(self.source, self.target)
        # check if path exists and stuff
        while path is not None:
            min_residual = float('inf')
            for i in range(1, len(path)):
                edge = path[i - 1], path[i]
                residual = self.get_edge_capacity(edge) - self.get_edge_flow(edge)
                if residual < min_residual:
                    min_residual = residual
            # now, increase flow by that min on every edge on path
            for i in range(1, len(path)):
                u, v = path[i - 1], path[i]
                self.set_flow(u, v, self.get_edge_flow((u, v)) + min_residual)
            # re-build and stuff
            residual_network = self._build_residual_network()
            path, _ = residual_network.shortest_path_from_to(self.source, self.target)

    def get_current_flow(self):
        """Returns the current output (flow) from the source of the network."""
        return self.get_vertex_flow(self.source)

    def get_current_capacity(self):
        """Returns the current potential (capacity) of the network.
        This means - returns the capacity over a minimal cut."""
        # * the best way I can find to do this (as of now) is to
        # return the max-flow using F-F on a temp network
        tmp = deepcopy(self)
        tmp.ford_fulkerson()
        return tmp.get_vertex_flow(self.source)


if __name__ == '__main__':
    capacity = {(1, 2): 5, (2, 3): 2, (1, 3): 4, (2, 4): 5, (3, 4): 2}
    nf = FlowNetwork([1, 2, 3, 4], [(1, 2), (2, 3), (1, 3), (2, 4), (3, 4)], 1, 4, capacity)

    nf.ford_fulkerson()
    print(nf.get_current_flow())

    nf.set_capacity_with_prop(1, 2, 3)
    print(nf.get_current_flow())

    nf.ford_fulkerson()
    print(nf.get_current_flow())

    for e in nf.get_edges():
        print(f"{e=}, {nf.get_edge_capacity(e)=}, {nf.get_edge_flow(e)=}")
