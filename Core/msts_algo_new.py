############################################################

## IMPORT PYTHON PACKAGES
from posixpath import splitext
import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import copy
from prettytable import PrettyTable
import timeit

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
import operation_scheduling

############################################################

def print_job_info(jobs_array):
    table = PrettyTable(['Job', 'Operation', 'Pre', 'Succ', 'Series', 'Machines', 'Assigned Machine'])
    for job in jobs_array:
        for op in job:
            table.add_row([op.job_num, op.op_num, op.pre, op.succ, op.series, op.machines, op.mach_num])
    print(table)

def initial_solution(jobs_array, machine_graph, MA_algo, OS_algo):
    
    ### MACHINE ASSIGNMENT ###

    # Random machine assignment
    if MA_algo.lower() == 'random':
        #print(MA_algo)
        x = machine_assignment.run_random(jobs_array, machine_graph)
        #print("Return Random", x)
        #print(graph.get_graph_info(machine_graph))
    # Greedy machine assignment
    elif MA_algo.lower() == 'greedy':
        #print(MA_algo)
        x = machine_assignment.run_greedy(jobs_array, machine_graph, "FMT")
        #print("Return Greedy", x)
        #print(graph.get_graph_info(machine_graph))
    # Shortest Path machine assignment
    else:
        #print(MA_algo)
        x = machine_assignment.run_shortest_path(jobs_array, machine_graph)
        #print("Return Shortest Path:", x)
        #rint(graph.get_graph_info(machine_graph))

    ### OPERATION SCHEDULING ###
    # Shortest Machining Time
    if OS_algo == "SMT":
        print(OS_algo)
        y = operation_scheduling.schedule_SMT(jobs_array, machine_graph)
        print("Return SMT Schedule:", y)
        print(graph.get_graph_info(machine_graph))
    # Largest Remaining Machining Time
    elif OS_algo == "LRMT":
        print(OS_algo)
        y = operation_scheduling.schedule_LRMT(jobs_array, machine_graph)
        print("Return LRMT Schedule:", y)
        print(graph.get_graph_info(machine_graph))
    # Earliest Release Time
    else:
        print(OS_algo)
        y = operation_scheduling.schedule_ERT(jobs_array, machine_graph)
        print("Return ERT Schedule:", y)
        print(graph.get_graph_info(machine_graph))
    
    return jobs_array

def create_gantt_chart(machine_graph):
    op_schedules = []
    res = nx.get_node_attributes(machine_graph, 'op_schedule')

    for mach, schedule in res.items():
        for tupl in schedule:
            temp_dict = dict(Job="J"+tupl[0][1], Start=tupl[1], Finish=tupl[2], Machine=mach, Details=tupl[0])
            op_schedules.append(temp_dict)

    #print(op_schedules)
    gantt_df = pd.DataFrame(op_schedules)
    gantt_df['Delta'] = gantt_df['Finish'] - gantt_df['Start']
    
    fig = px.timeline(gantt_df, x_start="Start", x_end="Finish", y="Machine", color="Job", text="Details")
    fig.update_yaxes(autorange="reversed")
    fig.layout.xaxis.type = 'linear'
    for d in fig.data:
        filt = gantt_df['Job'] == d.name
        d.x = gantt_df[filt]['Delta'].tolist()
    fig.show()

def msts(instances_file):
    ############################################################
    #               INITIALIALISE JOBS + MACHINES              #
    ############################################################

    # TODO: Get as inputs
    #instances_file = base_dir + 'data\Benchmarks\YFJS\YFJS17.txt'
    #instances_file = base_dir + 'data\Benchmarks\DAFJS\DAFJS03.txt'
    #instances_file = base_dir + 'data\Benchmarks\FMJ\mfjs01.txt'
    num_jobs = 3
    num_diff_machines = 4
    MA_algo_choice = "SHORTEST PATH"
    OS_algo_choice = "ERT"
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
    
    #print_job_info(jobs_array)

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
    create_gantt_chart(curr_graph)


### BEGIN MAIN PROGRAM ###
if __name__ == '__main__':
    """
    starttime = timeit.default_timer()
    filename = "data\Benchmarks\DAFJS\DAFJS01.txt"
    msts(filename)
    print("Time taken for", filename, ":", timeit.default_timer() - starttime)
    """
    """
    print("## YFJS: ##")
    for i in range(18, 20):
        if i < 9:
            file_num = "0" + str(i+1)
        else:
            file_num = str(i+1)
        filename = "data\Benchmarks\YFJS\YFJS" + file_num + ".txt"
        starttime = timeit.default_timer()
        msts(filename)
        print("Time taken for", filename, ":", timeit.default_timer() - starttime)
    """
    
    print("## DAFJS: ##")
    for i in range(30):
        if i < 9:
            file_num = "0" + str(i+1)
        else:
            file_num = str(i+1)
        filename = "data\Benchmarks\DAFJS\DAFJS" + file_num + ".txt"
        starttime = timeit.default_timer()
        msts(filename)
        print("Time taken for", filename, ":", timeit.default_timer() - starttime)
    
    """
    print("## YFJS: ##")
    for i in range(18, 20):
        file_num = str(i+1)
        filename = "data\Benchmarks\YFJS\YFJS" + file_num + ".txt"
        starttime = timeit.default_timer()
        msts(filename)
        print("Time taken for", filename, ":", timeit.default_timer() - starttime)
    """