import networkx as nx

## CREATE GRAPH STRUCTURE ##

# Add node to the graph with attributes
def add_node(graph, machine):
    graph.add_node(machine[0], type=machine[1], tasks=machine[2], alpha=machine[3], beta=machine[4], op_schedule=machine[5])

# Add weighted edges between nodes (complete undirected graph)
def add_edge_costs(machine_graph, t_times_array, col_names):
    weighted_edges_list = []
    for i in range(len(t_times_array)):
        for j in range(len(t_times_array)):
            #edge_tuple = (col_names[i], col_names[j], t_times_array[i][j])
            weighted_edges_list.append((col_names[i], col_names[j], t_times_array[i][j]))
    machine_graph.add_weighted_edges_from(weighted_edges_list)

# Get graph data
def get_graph_info(graph):
    return graph.nodes.data()

# Get specific node's data
def get_node_info(graph, node):
    return graph.nodes[node]

# Get total number of nodes
def get_num_of_nodes(graph):
    return graph.number_of_nodes()

# Get total number of edges
def get_num_of_edges(graph):
    return graph.number_of_edges()