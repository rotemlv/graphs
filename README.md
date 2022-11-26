# graphs
Implementation of graphs and flow networks (a somewhat OO approach):

Basic graph algorithms:
- DFS
- DFS forests
- BFS (all-reachable-nodes traversal from a given vertex)
- BFS (shortest unweighted path from a given vertex)
- Topological-sort
- Strongly-connected-components (DFS based)

Weighted graph algorithms:

  Shortest path:
  - Dijkstra (**)
  - Bellman-Ford:
    - Standard version for directed graphs with negative weights
    - Special version for un-directed graphs (treats negative edges as directed when encountered).
  - SSSP-DAG (Top-sort based linear time algorithm)
  
  MST:
  - Kruskal (*)
  - Prim (**)
  
Flow networks:
- Ford-Fulkerson (BFS and DFS versions)
  
*Kruskal is built on top of the highly-efficient disjoint-set data-structure, implemented in the disjoint_set.py file.

\*\*Dijkstra and Prim implementation are not using a fibonacci heap.

~~The main.py file holds some examples for how to use the BaseGraph and WeightedGraph classes.~~

main.py:

Besidees the simple examples, main.py contains two plotting functions for graphs (using the networkx module), 
which allow us to visualize graphs (both weighted and un-weighted).
Graph class (defined in main.py) allows creating a generic graph with the relevant data.

network_graph.py:

Holds some examples for using a flow network with the FlowNetwork class.

External modules required for plotting: networkx, matplotlib

TODO:
improve documentation for main.py
