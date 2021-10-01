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

def msts():

    ############################################################
    #               INITIALIALISE JOBS + MACHINES              #
    ############################################################

    # TODO: Get as inputs
    num_jobs = 2
    num_diff_machines = 4
    machine_types = {'Mill' : None, 'Lathe' : None, 'Drill' : None, 'Bore' : None, 'Hone' : None, 'Broach': None}
    algo_choice = "random"

    # Initialise as empty array/list
    jobs_array = []

    ## IMPORT JOBS ##
    # Read data from excel file and store in DataFrame
    filepath = base_dir + 'data\Operations_Data.xlsx'
    operation_df = import_data.import_job_data(filepath)

    # Split each job to store independently in jobsArray
    for i in range(num_jobs):
        job_number = 'J' + str(i+1)
        temp_job = operation_df[operation_df['Job'] == job_number]
        temp_job = temp_job.to_numpy()
        #print(temp_job)
        jobs_array.append(temp_job)

    #print("Printing Job Array:")
    #print(jobs_array)

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

    """
    ############################################################
    #                   INITIAL SOLUTION                       #
    ############################################################

    # If machine assignment algorithm not pre-specified, choose randomly
    if algo_choice == "":
        algo_choice = np.random.choice(['Random', 'Greedy', 'Shortest Path'], p=[0.35, 0.35, 0.3])

    # Random machine assignment
    if algo_choice.lower() == 'random':
        print(algo_choice)
        x = machine_assignment_algo.run_random(jobs_array, G, t_times_df)
    
    # Greedy machine assignment
    elif algo_choice.lower() == 'greedy':
        print(algo_choice)
    
    # Shortest Path machine assignment
    else:
        print(algo_choice)

    """


# Begin program here
if __name__ == '__main__':
    msts()