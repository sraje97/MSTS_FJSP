import sys
import os

from numpy import random

# Sets base directory one level higher than current file (@ X:\\..\\MSTS_FJSP)
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')

# Add base directory and the data directory to the system path
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'data'))


import networkx as nx
import numpy as np

import graph
import import_data

MG = nx.Graph()
num_mach = 4

marr = ['M1', 'Mill', np.array(['Milling', 'Turning', 'Boring', 'Drilling'], dtype='<U8'), 1.5, 1.01, list([])]
marr2 = ['M2', 'Lathe', np.array(['Turning', 'Grooving', 'Grinding'], dtype='<U8'), 0.77, 1.32, list([])]
marr3 = ['M3', 'Drill', np.array(['Drilling'], dtype='<U8'), 1.18, 1.14, list([])]
marr4 = ['M4', 'Hone', np.array(['Honing'], dtype='<U8'), 0.99, 1.04, list([])]

filepath = base_dir + 'data\Machines_Data.xlsx'
t_times_df = import_data.import_transition_times(filepath)
col_names = list(t_times_df.columns)
t_times_array = t_times_df.to_numpy()
print(t_times_array)
print(col_names)


graph.add_node(MG, marr)
graph.add_node(MG, marr2)
graph.add_node(MG, marr3)
graph.add_node(MG, marr4)

weight_edges_list = []

for i in range(len(t_times_array)):
    node_a = col_names[i]
    for j in range(len(t_times_array)):
        node_b = col_names[j]
        edge_tuple = (node_a, node_b, t_times_array[i][j])
        weight_edges_list.append(edge_tuple)

MG.add_weighted_edges_from(weight_edges_list)
print(MG.edges.data())
print(MG.edges[marr[0], marr2[0]]['weight'])
"""
print("Testing")
print(graph.get_node_info(MG, 'M1'))
print(graph.get_node_info(MG, 'M2'))

print("Getting node attributes")
res = nx.get_node_attributes(MG, 'tasks')
print(res)

elg_mach = []

for key, val in res.items():
    #print(key)
    x = np.where(val == "Drilling")
    if x[0].size != 0:
            print("This machine", key, "has value", val[x] , "with index", x[0])
            elg_mach.append(key)
    print("Xx")
#print(elg_mach[0])


#nx.set_node_attributes(MG, {elg_mach[0]:{'op_schedule':("O11", 0, 705)}})
print("Node info M1")
#print(graph.get_node_info(MG, 'M1'))
MG.nodes[elg_mach[0]]['op_schedule'] = ("O11", 0, 705)
print(MG.nodes['M1'])
"""