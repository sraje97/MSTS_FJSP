import sys
import os

from numpy import empty, nan, random

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
filepath = base_dir + 'data\Operations_Data_Testing.xlsx'
operation_df = import_data.import_job_data(filepath)
operation_df = operation_df.fillna('')

for i in range(num_jobs):
    job_number = 'J' + str(i+1)
    temp_job = operation_df[operation_df['Job'] == job_number]
    temp_job = temp_job.to_numpy()
    for temp_op in temp_job:
        # Convert parallel pre/succ operations to tuple
        if ',' in temp_op[6]:
            temp_op[6] = tuple(temp_op[6].split(','))
        if ',' in temp_op[7]:
            temp_op[7] = tuple(temp_op[7].split(','))
    jobs_array.append(temp_job)

## Assert Operations_Data.xlsx has been read
assert (len(jobs_array) > 0), "No Jobs detected"
print(jobs_array)

## IMPORT MACHINES ##
filepath = base_dir + 'data\Machines_Data_Testing.xlsx'
machine_df = import_data.import_machine_data(filepath)
machines_array = machine_df.to_numpy()
for i in range(len(machines_array)):
    machines_array[i][2] = machines_array[i][2].split(',')
    machines_array[i][2] = np.array(machines_array[i][2])
    machines_array[i][-1] = []

## Assert Machine_Data.xlsx has been read
assert len(machines_array) > 0, "No machines detected"

## IMPORT TRANISITION TIMES ##
t_times_df = import_data.import_transition_times(filepath)
col_names = list(t_times_df.columns)
t_times_array = t_times_df.to_numpy()

## Assert transition times from Machines_Data.xlsx has been read
assert len(t_times_array) > 0, "No transition time matrix detected"

## INITIALISE GRAPH ##
TG = nx.Graph()
for i in range(len(machines_array)):
    graph.add_node(TG, machines_array[i])
graph.add_edge_costs(TG, t_times_array, col_names)

## Assert nodes and weighted edges are added to graph
res = nx.get_node_attributes(TG, 'tasks')
assert len(res) > 0, "No nodes found with tasks"
assert TG.edges[machines_array[0][0], machines_array[1][0]]['weight'] == 30, "Weighted edges not detected"

operation = jobs_array[0][0]
elg_machs = list(machine_assignment_algo.find_eligible_machines(operation, TG))

## Assert find_eligible_machines() method finds at least one eligible machine
assert len(elg_machs) > 0, "No eligible machines found"
assert elg_machs[0] == 'M1', "Eligible machine not found"

mach_time = machine_assignment_algo.calculate_machining_time(operation, TG, elg_machs[0])
assert mach_time == 410, "Incorrect value"

edge_cost = machine_assignment_algo.get_transition_time(TG, elg_machs[0], machines_array[1][0])

algo_choice = "random"
x = machine_assignment_algo.run_random(jobs_array, TG)
random_op_sched = TG.nodes[elg_machs[0]]['op_schedule']
assert len(random_op_sched) > 0, "No schedule detected"

print("Random Algorithm:")
print(TG.nodes.data())

graph.clear_multiple_nodes_data(TG, 'op_schedule')
print("Clear:")
print(TG.nodes.data())
assert TG.nodes[elg_machs[0]]['op_schedule'] == [], "Clearing attribute data failed"

x = machine_assignment_algo.run_greedy(jobs_array, TG)
greedy_op_sched = TG.nodes[elg_machs[0]]['op_schedule']
assert greedy_op_sched[0] == ("O11", 0, 0), "Wrong operation assigned to machine"

print("Greedy Algorithm:")
print(TG.nodes.data())
