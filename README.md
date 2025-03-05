# graphs
Implementation of graphs and flow networks (a somewhat OO approach):

## Basic graph algorithms:
- DFS
- DFS forests
- BFS (all-reachable-nodes traversal from a given vertex)
- BFS (shortest unweighted path from a given vertex)
- Topological-sort
- Strongly-connected-components (DFS based)

## Weighted graph algorithms:

  ### Shortest path:
  - Dijkstra (**)
  - Bellman-Ford:
    - Standard version for directed graphs with negative weights
    - Special version for un-directed graphs (treats negative edges as directed when encountered).
  - SSSP-DAG (Top-sort based linear time algorithm)
  
  ### MST:
  - Kruskal (*)
  - Prim (**)

<sub>\* Kruskal is built on top of the highly-efficient disjoint-set data-structure, implemented in the disjoint_set.py file.</sub>

<sub>\** Dijkstra and Prim implementation are not using a fibonacci heap.</sub>

## Flow networks:
- Ford-Fulkerson (BFS and DFS versions)

### main.py:

Besides some simple examples, main.py contains two plotting functions for graphs (using the networkx module), 
allowing visualization for the different graph types (both weighted and un-weighted).
Graph class (defined in main.py) allows creating a generic graph with any relevant data.

network_graph.py:

Holds some examples for using a flow network with the FlowNetwork class.

External modules required for plotting: networkx, matplotlib

### TODO:
improve documentation for main.py
