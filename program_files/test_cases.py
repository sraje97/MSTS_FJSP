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
import pandas as pd

import graph
import import_data
import machine_assignment_algo

# INPUTS
num_jobs = 2
jobs_array = []

# BEGIN
filepath = base_dir + 'data\Operations_Data.xlsx'
operation_df = import_data.import_job_data(filepath)

for i in range(num_jobs):
    job_number = 'J' + str(i+1)
    temp_job = operation_df[operation_df['Job'] == job_number]
    temp_job = temp_job.to_numpy()
    #print(temp_job)
    jobs_array.append(temp_job)

assert (len(jobs_array) > 0), "No Jobs detected"

## IMPORT MACHINES ##
filepath = base_dir + 'data\Machines_Data.xlsx'
machine_df = import_data.import_machine_data(filepath)

# Convert dataframe to numpy array
machines_array = machine_df.to_numpy()

# Convert machine capabilities into array of strings
# And set operation schedule to be an array
for i in range(len(machines_array)):
    machines_array[i][2] = machines_array[i][2].split(',')
    machines_array[i][2] = np.array(machines_array[i][2])
    machines_array[i][-1] = []

assert len(machines_array) > 0, "No machines detected"

## IMPORT TRANISITION TIMES ##
t_times_df = import_data.import_transition_times(filepath)
col_names = list(t_times_df.columns)
t_times_array = t_times_df.to_numpy()
assert len(t_times_array) > 0, "No transition time matrix detected"

## INITIALISE GRAPH ##
TG = nx.Graph()
for i in range(len(machines_array)):
    graph.add_node(TG, machines_array[i])

graph.add_edge_costs(TG, t_times_array, col_names)

res = nx.get_node_attributes(TG, 'tasks')
assert len(res) > 0, "No nodes found with tasks"
assert G.edges[machines_array[0][0], machines_array[1][0]]['weight'] > 0, "Weighted edges not detected"


operation = jobs_array[0][0]
elg_machs = machine_assignment_algo.find_eligible_machines(operation, TG)
print(elg_machs, TG.nodes[elg_machs[0]])
assert len(elg_machs) > 0, "No eligible machines found"

mach_time = machine_assignment_algo.calculate_machining_time(operation, TG, elg_machs[0])
print(mach_time)