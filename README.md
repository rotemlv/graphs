# graphs
Implementation of graphs and flow networks:

Basic graph algorithms:
- DFS
- DFS forests
- BFS (all-reachable-nodes traversal from a given vertex)
- BFS (shortest unweighted path from a given vertex)
- Topological-sort
- Strongly-connected-components (DFS based)

Weighted graph algorithms:

  Shortest path:
  - Dijkstra
  - Bellman-Ford
  - SSSP-DAG (Top-sort based linear time algorithm)
  
  MST:
  - Kruskal (*)
  - Prim
  
Flow networks:
- Ford-Fulkerson (BFS and DFS versions)
  
*Kruskal is built on top of the highly-efficient disjoint-set data-structure, implemented in the disjoint_set.py file.

the main.py file holds some examples for how to use the BaseGraph and WeightedGraph classes.


External modules required for plotting: networkx, matplotlib
