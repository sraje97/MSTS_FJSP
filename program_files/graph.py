import networkx as nx

## CREATE GRAPH STRUCTURE ##

def add_node(graph, machine):
    graph.add_node(machine[0], type=machine[1], tasks=machine[2], alpha=machine[3], beta=machine[4], op_schedule=machine[5])
    #print(graph.nodes)
    
def get_graph_info(graph):
    return graph.nodes.data()

def get_node_info(graph, node):
    return graph.nodes[node]

def get_num_of_nodes(graph):
    return graph.number_of_nodes()

def get_num_of_edges(graph):
    return graph.number_of_edges()