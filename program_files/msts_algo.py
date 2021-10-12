import sys
import os

# Sets base directory one level higher than current file (@ X:\\..\\MSTS_FJSP)
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')

# Add base directory and the data directory to the system path
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'data'))

## IMPORT PYTHON PACKAGES
import numpy as np
import pandas as pd
import networkx as nx

## IMPORT OUR MODULES
import import_data
import machine_assignment_algo
import graph
import operation_scheduling
from operation import Operation

def msts():

    ############################################################
    #               INITIALIALISE JOBS + MACHINES              #
    ############################################################

    # TODO: Get as inputs
    num_jobs = 2
    num_diff_machines = 4
    machine_types = {'Mill' : None, 'Lathe' : None, 'Drill' : None, 'Bore' : None, 'Hone' : None, 'Broach': None}
    MA_algo_choice = "Shortest Path"
    OS_algo_choice = "SMT"

    # Initialise as empty array/list
    jobs_array = []

    ## IMPORT JOBS ##
    # Read data from excel file and store in DataFrame
    filepath = base_dir + 'data\Operations_Data.xlsx'
    operation_df = import_data.import_job_data(filepath)

    # Split each job to store independently in jobsArray
    for i in range(num_jobs):
        new_job = []
        job_number = 'J' + str(i+1)
        temp_job = operation_df[operation_df['Job'] == job_number]
        temp_job = temp_job.to_numpy()

        # Convert parallel pre/succ operations to tuple
        for j in range(len(temp_job)):
            if ',' in temp_job[j][6]:
                temp_job[j][6] = tuple(temp_job[j][6].split(','))
            if ',' in temp_job[j][7]:
                temp_job[j][7] = tuple(temp_job[j][7].split(','))
            op = Operation(temp_job[j])
            new_job.append(op)
            
        jobs_array.append(temp_job)

    ## IMPORT MACHINES ##
    filepath = base_dir + 'data\Machines_Data.xlsx'
    machine_df = import_data.import_machine_data(filepath)
    
    # Convert dataframe to numpy array
    machines_array = machine_df.to_numpy()

    print("Printing Machines")
    # Convert machine capabilities into array of strings
    # And set operation schedule to be an array
    for i in range(len(machines_array)):
        machines_array[i][2] = machines_array[i][2].split(',')
        machines_array[i][2] = np.array(machines_array[i][2])
        machines_array[i][-1] = []
        print(machines_array[i])

    ## IMPORT TRANISITION TIMES ##
    #global t_times_df
    t_times_df = import_data.import_transition_times(filepath)
    col_names = list(t_times_df.columns)
    t_times_array = t_times_df.to_numpy()
    #print(t_times_df)

    ## INITIALISE GRAPH ##
    G = nx.Graph()
    for i in range(len(machines_array)):
        graph.add_node(G, machines_array[i])
    
    graph.add_edge_costs(G, t_times_array, col_names)

    print("Printing Graph")
    print(graph.get_graph_info(G))

    print("Printing Node")
    print(graph.get_node_info(G, 'M1'))

    
    ############################################################
    #                   INITIAL SOLUTION                       #
    ############################################################

    # If machine assignment algorithm not pre-specified, choose randomly
    if MA_algo_choice == "":
        MA_algo_choice = np.random.choice(['Random', 'Greedy', 'Shortest Path'], p=[0.35, 0.35, 0.3])

    # Random machine assignment
    if MA_algo_choice.lower() == 'random':
        print(MA_algo_choice)
        x = machine_assignment_algo.run_random(jobs_array, G)
        print("Return Random", x)
        print(graph.get_graph_info(G))
    # Greedy machine assignment
    elif MA_algo_choice.lower() == 'greedy':
        print(MA_algo_choice)
        x = machine_assignment_algo.run_greedy(jobs_array, G, "FMT")
        print("Return Greedy", x)
        print(graph.get_graph_info(G))
    # Shortest Path machine assignment
    else:
        print(MA_algo_choice)
        x = machine_assignment_algo.run_shortest_path(jobs_array, G)
        print("Return Shortest Path:", x)
        print(graph.get_graph_info(G))
    
    ############################################################
    #                   OPERATION SCHEDULING                   #
    ############################################################

    if OS_algo_choice == '':
        OS_algo_choice = np.random.choice(['SMT', 'LRMT', 'ERT']) #, p=[0.35, 0.35, 0.3])

    if OS_algo_choice == "SMT":
        print(OS_algo_choice)
        y = operation_scheduling.schedule_SMT
        print("Return Shortest Path:", y)
        print(graph.get_graph_info(G))

# Begin program here
if __name__ == '__main__':
    msts()