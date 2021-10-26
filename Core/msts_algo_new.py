############################################################

## IMPORT PYTHON PACKAGES
from posixpath import splitext
import sys
import os
import numpy as np
#import pandas as pd
import networkx as nx
#import plotly.graph_objects as go
#import plotly.express as px
import copy
from prettytable import PrettyTable

# Sets base directory one level higher than current file (@ X:\\..\\MSTS_FJSP)
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')

# Add base directory and the data directory to the system path
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'data'))

## IMPORT OUR MODULES
from operation import Operation
import preprocess
import graph
import machine_assignment

############################################################
def initial_solution(jobs_array, machine_graph, MA_algo, OS_algo):
    
    ### MACHINE ASSIGNMENT ###

    # Random machine assignment
    if MA_algo.lower() == 'random':
        print(MA_algo)
        x = machine_assignment.run_random(jobs_array, machine_graph)
        print("Return Random", x)
        print(graph.get_graph_info(machine_graph))
    # Greedy machine assignment
    elif MA_algo.lower() == 'greedy':
        print(MA_algo)
        x = machine_assignment.run_greedy(jobs_array, machine_graph, "FMT")
        print("Return Greedy", x)
        print(graph.get_graph_info(machine_graph))
    # Shortest Path machine assignment
    else:
        print(MA_algo)
        x = machine_assignment.run_shortest_path(jobs_array, machine_graph)
        print("Return Shortest Path:", x)
        print(graph.get_graph_info(machine_graph))

def msts():
    ############################################################
    #               INITIALIALISE JOBS + MACHINES              #
    ############################################################

    # TODO: Get as inputs
    instances_file = base_dir + 'data\Benchmarks\DAFJS\DAFJS28.txt'
    num_jobs = 3
    num_diff_machines = 4
    MA_algo_choice = "shortest path"
    OS_algo_choice = ""
    epochs = 100
    C_eni_max = 100
    p_exp_con = 1.0
    swap_methods = ["swap machine", "swap operation", "SPO", "MSPO"]

    eps_start = 1.0
    eps_end = 0.0
    eps_decay = 0.5

    # Load operation's information from instance's text file
    jobs_array, machines_array = preprocess.initialise_operations(instances_file)

    # If test instances are DAFJS/YFJS, jobs may have parallel operations
    # Hence, label the branches of the parallel paths
    if "DAFJS" in instances_file or "YFJS" in instances_file:
        jobs_array = preprocess.label_parallel_branches(jobs_array)

    machines_array.sort()
    G = nx.Graph()
    for machine in machines_array:
        #print(machine)
        graph.add_node(G, machine)
    
    if os.path.splitext(instances_file)[1] == '.txt':
        trans_times = np.zeros((len(machines_array), len(machines_array)))
        graph.add_edge_costs(G, trans_times, machines_array)
    
    table = PrettyTable(['Job', 'Operation', 'Pre', 'Succ', 'Series', 'Machines', 'Assigned Machine'])
    for job in jobs_array:
        for op in job:
            table.add_row([op.job_num, op.op_num, op.pre, op.succ, op.series, op.machines, op.mach_num])
    print(table)
    

    #print("Printing Graph")
    #print(graph.get_graph_info(G))

    #print("Printing Node")
    #print(graph.get_node_info(G, 'M1'))

    # If machine assignment algorithm not pre-specified, choose randomly
    if MA_algo_choice == "":
        MA_algo_choice = np.random.choice(['Random', 'Greedy', 'Shortest Path'], p=[0.35, 0.35, 0.3])
    if OS_algo_choice == '':
        OS_algo_choice = np.random.choice(['SMT', 'LRMT', 'ERT']) #, p=[0.35, 0.35, 0.3])

    curr_jobs = copy.deepcopy(jobs_array)
    curr_graph = copy.deepcopy(G)

    # TODO:
    ## Get initial solution ##
    curr_jobs = initial_solution(curr_jobs, curr_graph, MA_algo_choice, OS_algo_choice)


### BEGIN MAIN PROGRAM ###
if __name__ == '__main__':
    msts()