############################################################

## IMPORT PYTHON PACKAGES
import sys
import os
import numpy as np
import networkx as nx
import copy

from numpy import random

# Sets base directory one level higher than current file (@ X:\\..\\MSTS_FJSP)
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')

# Add base directory and the data directory to the system path
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'data'))

## IMPORT OUR MODULES
import import_data
import machine_assignment_algo
import graph
import operation_scheduling
from operation import Operation

############################################################

def initial_solution(jobs_array, machine_graph, MA_algo, OS_algo):
    
    ### MACHINE ASSIGNMENT ###

    # Random machine assignment
    if MA_algo.lower() == 'random':
        print(MA_algo)
        x = machine_assignment_algo.run_random(jobs_array, machine_graph)
        print("Return Random", x)
        print(graph.get_graph_info(machine_graph))
    # Greedy machine assignment
    elif MA_algo.lower() == 'greedy':
        print(MA_algo)
        x = machine_assignment_algo.run_greedy(jobs_array, machine_graph, "FMT")
        print("Return Greedy", x)
        print(graph.get_graph_info(machine_graph))
    # Shortest Path machine assignment
    else:
        print(MA_algo)
        x = machine_assignment_algo.run_shortest_path(jobs_array, machine_graph)
        print("Return Shortest Path:", x)
        print(graph.get_graph_info(machine_graph))
    
    ### OPERATION SCHEDULING ###

    if OS_algo == '':
        OS_algo_choice = np.random.choice(['SMT', 'LRMT', 'ERT']) #, p=[0.35, 0.35, 0.3])
    
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

def calculate_makespan(machine_graph):
    res = nx.get_node_attributes(machine_graph, 'op_schedule')
    makespan = 0
    op_num = ''
    mach_num = ''
    for mach, schedule in res.items():
        for tupl in schedule:
            if tupl[2] > makespan:
                op_num = tupl[0]
                mach_num = mach
                makespan = tupl[2]
    return op_num, mach_num, makespan

def get_random_operation(jobs_array):
    op_list = []
    for job in jobs_array:
        for op in job:
            op_list.append(op.op_num)
    return np.random.choice(op_list)

def msts():

    ############################################################
    #               INITIALIALISE JOBS + MACHINES              #
    ############################################################

    # TODO: Get as inputs
    num_jobs = 3
    num_diff_machines = 4
    MA_algo_choice = ""
    OS_algo_choice = ""
    epochs = 100
    C_eni_max = 100
    p_exp_con = 1.0
    swap_methods = ["swap machine", "swap operation", "SPO", "MSPO"]

    eps_start = 1.0
    eps_end = 0.0
    eps_decay = 0.5


    # Initialise as empty array/list
    jobs_array = []

    ## IMPORT JOBS ##
    # Read data from excel file and store in DataFrame
    filepath = base_dir + 'data\Operations_Data.xlsx'
    operation_df = import_data.import_job_data(filepath)
    operation_df = operation_df.fillna('')

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

        jobs_array.append(new_job)

    ## IMPORT MACHINES ##
    filepath = base_dir + 'data\Machines_Data.xlsx'
    machine_df = import_data.import_machine_data(filepath)

    # Convert dataframe to numpy array
    machines_array = machine_df.to_numpy()

    #print("Printing Machines")
    # Convert machine capabilities into array of strings
    # And set operation schedule to be an array
    for i in range(len(machines_array)):
        machines_array[i][2] = machines_array[i][2].split(',')
        machines_array[i][2] = np.array(machines_array[i][2])
        machines_array[i][-1] = []
        #print(machines_array[i])

    ## IMPORT TRANISITION TIMES ##
    t_times_df = import_data.import_transition_times(filepath)
    col_names = list(t_times_df.columns)
    t_times_array = t_times_df.to_numpy()
    #print(t_times_df)

    ## INITIALISE GRAPH ##
    G = nx.Graph()
    for i in range(len(machines_array)):
        graph.add_node(G, machines_array[i])
    
    graph.add_edge_costs(G, t_times_array, col_names)

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

    ## Get initial solution ##
    curr_jobs = initial_solution(curr_jobs, curr_graph, MA_algo_choice, OS_algo_choice)

    # Create a copy of the current solution (jobs and machine graph)
    curr_solution = (copy.deepcopy(curr_jobs), copy.deepcopy(curr_graph))
    #curr_jobs = copy.deepcopy(jobs_array)
    #curr_graph = copy.deepcopy(G)

    _, _, local_best_makespan = calculate_makespan(curr_solution[1])
    global_best_makespan = local_best_makespan


    ############################################################
    #                  BEGIN MSTS ALGORITHM                    #
    ############################################################
    e_cnt = 0
    C_eni = 0
    tabu_list = []

    while e_cnt < epochs:
        # Keep a copy of the previous solution
        #prev_solution = (copy.deepcopy(curr_solution[0]), copy.deepcopy(curr_solution[1]))
        prev_solution = copy.deepcopy(curr_solution)
        #prev_jobs = curr_jobs
        #prev_graph = curr_graph

        # Get random operation and swap method to operate on
        operation = get_random_operation(curr_solution[0])
        swap_method = np.random.choice(swap_methods)

        ### TODO: CONDUCT SWAP ###
        # tabu_tuple = SWAP_OPERATION(operation, swap_method, curr_jobs, curr_graph)
        tabu_tuple = (operation, swap_method, 10)       # Tabu_Tuple : (operation, method, tabu tenure)

        last_op, last_mach, makespan = calculate_makespan(curr_solution[1])

        if makespan < local_best_makespan:
            local_best_solution = copy.deepcopy(curr_solution)
            local_best_makespan = makespan

            if tabu_tuple in tabu_list:
                tabu_list.remove(tabu_tuple)
            tabu_list.append(tabu_tuple)

            C_eni = 0
        else:
            C_eni += 1
            curr_solution = prev_solution

        for tabu_tuple in tabu_list:
            tabu_tuple[2] -= 1
            if tabu_tuple[2] == 0:
                tabu_list.remove(tabu_tuple)
        
        if C_eni == C_eni_max:
            p = random.random()

            if local_best_makespan < global_best_makespan:
                global_best_solution = copy.deepcopy(local_best_solution)
                global_best_makespan = local_best_solution
            
            if p < p_exp_con:
                curr_jobs = copy.deepcopy(jobs_array)
                curr_graph = copy.deepcopy(G)

                ## Get initial solution ##
                curr_jobs = initial_solution(curr_jobs, curr_graph, MA_algo_choice, OS_algo_choice)
            else:
                curr_solution = global_best_solution
            
            p_exp_con = eps_end + (eps_start - eps_end) * np.exp(-1.0 * e_cnt / epochs / eps_decay)
            e_cnt += 1


### Begin program here ###
if __name__ == '__main__':
    msts()