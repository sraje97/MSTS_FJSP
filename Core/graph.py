import networkx as nx

############################################################

## CREATE GRAPH STRUCTURE ##
def add_node(graph, machine):
    graph.add_node(machine, alpha=1, beta=1, op_schedule=[])

# Add weighted edges between nodes (complete undirected graph)
def add_edge_costs(machine_graph, t_times_array, col_names):
    weighted_edges_list = []
    for i in range(len(t_times_array)):
        for j in range(len(t_times_array)):
            #edge_tuple = (col_names[i], col_names[j], t_times_array[i][j])
            weighted_edges_list.append((col_names[i], col_names[j], t_times_array[i][j]))
    machine_graph.add_weighted_edges_from(weighted_edges_list)


## GRAPH INFORMATION ##

# Get graph data
def get_graph_info(graph):
    return graph.nodes.data()

# Get specific node's data
def get_node_info(graph, node):
    return graph.nodes[node]

# Clear particular node's data
def clear_node_data(graph, node, attribute):
    if attribute == 'op_schedule':
        graph.nodes[node][attribute] = []
    else:
        graph.nodes[node][attribute] = None

# Clear all node's particular attribute
def clear_multiple_nodes_data(graph, attribute):
    for i in range(nx.number_of_nodes(graph)):
        node = "M" + str(i+1)
        if attribute == 'op_schedule':
            graph.nodes[node][attribute] = []
        else:
            graph.nodes[node][attribute] = None

# Sort machine's op_schedule
def sort_op_schedule(graph):
    res = nx.get_node_attributes(graph, 'op_schedule')
    
    for mach, schedule in res.items():
        schedule.sort(key=lambda a: a[2])
        nx.set_node_attributes(graph, {mach: {'op_schedule': schedule} } )