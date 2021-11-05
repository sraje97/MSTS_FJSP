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
import timeit
from prettytable import PrettyTable
from criticalpath import Node



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
import tabu_search

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
        #print(OS_algo)
        y = operation_scheduling.schedule_SMT(jobs_array, machine_graph)
        #print("Return SMT Schedule:", y)
        #print(graph.get_graph_info(machine_graph))
    # Largest Remaining Machining Time
    elif OS_algo == "LRMT":
        #print(OS_algo)
        y = operation_scheduling.schedule_LRMT(jobs_array, machine_graph)
        #print("Return LRMT Schedule:", y)
        #print(graph.get_graph_info(machine_graph))
    # Earliest Release Time
    else:
        #print(OS_algo)
        y = operation_scheduling.schedule_ERT(jobs_array, machine_graph)
        #print("Return ERT Schedule:", y)
        #print(graph.get_graph_info(machine_graph))
    
    return jobs_array, machine_graph

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

def check_tabu_status(oper, mach, tabu_list):
    for tupl in tabu_list:
        if tupl[0] == oper and tupl[1] == mach:
            # Operation + Machine in tabu list
            return True
    return False

def calculate_tenure(num_job, num_mach):
    Lmin = 10 + (num_job / num_mach)
    Lmax = 1.8 * Lmin
    return np.random.randint(low=Lmin, high=Lmax)


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
    MA_algo_choice = "greedy"
    OS_algo_choice = "ERT"

    epochs = 1
    TS_cnt_max = 1
    p_exp_con = 1.0
    swap_methods = ["Random MA", "LPT MA", "HMW MA", "Random OS", "HMW OS"]

    eps_start = 1.0
    eps_end = 0.0
    eps_decay = 0.5

    # Load operation's information from instance's text file
    jobs_array, machines_array = preprocess.initialise_operations(instances_file)

    # If test instances are DAFJS/YFJS, jobs may have parallel operations
    # Hence, label the branches of the parallel paths
    if "DAFJS" in instances_file or "YFJS" in instances_file:
        jobs_array, op_df = preprocess.label_parallel_branches(jobs_array)

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
    curr_jobs, curr_graph = initial_solution(curr_jobs, curr_graph, MA_algo_choice, OS_algo_choice)

    # Create a copy of the current solution (jobs and machine graph)
    curr_solution = (copy.deepcopy(curr_jobs), copy.deepcopy(curr_graph))
    _, _, local_best_mks = calculate_makespan(curr_solution[1])
    global_best_mks = local_best_mks

    # Sort operation schedule
    graph.sort_op_schedule(curr_graph)
    op_schedule = graph.get_op_schedule(curr_graph)

    # Create Gantt Chart
    # create_gantt_chart(curr_graph)
    print(local_best_mks)


    """
    sorted_jobs = tabu_search.flatten_job(curr_jobs)
    for job in sorted_jobs:
        for oper in job:
            print(oper.op_num, oper.job_num, oper.series, oper.processing_time)
    """
    

    ############################################################
    #                  BEGIN MSTS ALGORITHM                    #
    ############################################################

    e_cnt = 0           # Epoch counter
    TS_cnt = 0          # Tabu Search counter
    global_TS_cnt = 0   # Global Tabu Search counter
    tabu_list = []

    TS_start_time = timeit.default_timer()

    while e_cnt < epochs:
        # Terminate program after 1 hour
        if (timeit.default_timer() - TS_start_time) > 600: #TODO: Change to 3600
            break

        # Keep a copy of the previous solution
        prev_solution = copy.deepcopy(curr_solution)

        # Get random swap method
        #swap_method = np.random.choice(swap_methods)
        swap_method = "Critical Path"


        ## TODO: CONDUCT SWAP ##
        # Neighbourhood Tuple : (jobs_array, machine_graph, oper, mach, mks)
        neighbourhood = tabu_search.tabu_move(curr_solution[0], curr_solution[1], op_df, "Critical Path")
        best_neighbourhood = neighbourhood[0]
        # Tabu Tuple outline: (operation, machine, tabu_tenure)
        tabu_tuple = (best_neighbourhood[2], best_neighbourhood[3])

        # If there is an improvement in the makespan, keep the best neighbourhood solution
        if best_neighbourhood[-1] < local_best_mks:
            if tabu_list:
                for tupl in tabu_list:
                    if tupl[0] == tabu_tuple[0] and tupl[1] == tabu_tuple[1]:
                        tabu_list.remove(tupl)
                        break
            
            tenure = calculate_tenure( len(best_neighbourhood[0]) , graph.get_number_of_nodes(best_neighbourhood[1]) )

            tabu_tuple = (best_neighbourhood[2], best_neighbourhood[3], tenure)

            tabu_list.append(tabu_tuple)

            curr_solution = (copy.deepcopy(best_neighbourhood[0]), copy.deepcopy(best_neighbourhood[1]))
            local_best_mks = best_neighbourhood[-1]
            local_best_sln = (copy.deepcopy(best_neighbourhood[0]), copy.deepcopy(best_neighbourhood[1]))

            TS_cnt = 0
        else:
            # If the best_neighbour solution doesn't improve makespan, check this or other neighbours
            if tabu_list:
                ## TODO: Fix check_tabu_status
                # If tabu list exists check for non-tabu neighbourhood solution
                if check_tabu_status(tabu_tuple[0], tabu_tuple[1], tabu_list):
                    all_tabu = False
                    # Go through neighbours until we find a non-tabu solution
                    for neighbour in neighbourhood:
                        if check_tabu_status(neighbour[2], neighbour[3], tabu_list):
                            if neighbour == neighbourhood[-1]:
                                all_tabu = True
                            continue
                        tenure = calculate_tenure( len(neighbour[0]) , graph.get_number_of_nodes(neighbour[1]) )
                        tabu_tuple = (neighbour[2], neighbour[3], tenure)
                        curr_solution = (copy.deepcopy(neighbour[0]), copy.deepcopy(neighbour[1]))
                        break
                    if all_tabu:
                        # Store the best neighbour in case all neighbours are tabooed
                        tenure = calculate_tenure( len(best_neighbourhood[0]) , graph.get_number_of_nodes(best_neighbourhood[1]) )
                        tabu_tuple = (best_neighbourhood[2], best_neighbourhood[3], tenure)
                        curr_solution = (copy.deepcopy(best_neighbourhood[0]), copy.deepcopy(best_neighbourhood[1]))
                else:
                    # If best neighbourhood solution is non-tabu solution store it as the solution
                    tenure = calculate_tenure( len(best_neighbourhood[0]) , graph.get_number_of_nodes(best_neighbourhood[1]) )
                    tabu_tuple = (best_neighbourhood[2], best_neighbourhood[3], tenure)
                
                for tupl in tabu_list:
                    if tupl[0] == tabu_tuple[0] and tupl[1] == tabu_tuple[1]:
                        tabu_list.remove(tupl)
                tabu_list.append(tabu_tuple)

            else:
                # If no tabu_list then just add this straight away
                tenure = calculate_tenure( len(best_neighbourhood[0]) , graph.get_number_of_nodes(best_neighbourhood[1]) )
                tabu_tuple = (best_neighbourhood[2], best_neighbourhood[3], tenure)
                tabu_list.append(tabu_tuple)
                curr_solution = (copy.deepcopy(best_neighbourhood[0]), copy.deepcopy(best_neighbourhood[1]))

            TS_cnt += 1

        for tabu_tuple in tabu_list:
            tenure = tabu_tuple[-1] - 1
            tabu_tuple = (tabu_tuple[0], tabu_tuple[1], tenure)
            if tabu_tuple[-1] == 0:
                tabu_list.remove(tabu_tuple)

        if TS_cnt == TS_cnt_max:
            p = np.random.random()

            if local_best_mks < global_best_mks:
                global_best_mks = local_best_mks
                global_best_sln = copy.deepcopy(local_best_sln)
                global_TS_cnt = 0
            else:
                global_TS_cnt += 1
            
            if p < p_exp_con:
                curr_jobs = copy.deepcopy(jobs_array)
                curr_graph = copy.deepcopy(G)

                # Empty tabu list
                tabu_list = []

                ## Get initial solution ##
                curr_jobs = initial_solution(curr_jobs, curr_graph, MA_algo_choice, OS_algo_choice)
            else:
                curr_solution = global_best_sln

            p_exp_con = eps_end + (eps_start - eps_end) * np.exp(-1.0 * e_cnt / epochs / eps_decay)
            e_cnt += 1
    
    return global_best_sln, global_best_mks


### BEGIN MAIN PROGRAM ###
if __name__ == '__main__':
    
    starttime = timeit.default_timer()
    filename = "data\Benchmarks\DAFJS\DAFJS01.txt"
    sln, mks = msts(filename)
    print(mks)
    print("Time taken for", filename, ":", timeit.default_timer() - starttime)
    
    """
    print("## YFJS: ##")
    for i in range(20):
        if i < 9:
            file_num = "0" + str(i+1)
        else:
            file_num = str(i+1)
        filename = "data\Benchmarks\YFJS\YFJS" + file_num + ".txt"
        starttime = timeit.default_timer()
        sln, mks = msts(filename)
        print("Time taken for", filename, ":", timeit.default_timer() - starttime)
    """
    """
    print("## DAFJS: ##")
    for i in range(30):
        if i < 9:
            file_num = "0" + str(i+1)
        else:
            file_num = str(i+1)
        filename = "data\Benchmarks\DAFJS\DAFJS" + file_num + ".txt"
        starttime = timeit.default_timer()
        sln, mks = msts(filename)
        print("Time taken for", filename, ":", timeit.default_timer() - starttime)
    """
    """
    print("## YFJS: ##")
    for i in range(18, 20):
        file_num = str(i+1)
        filename = "data\Benchmarks\YFJS\YFJS" + file_num + ".txt"
        starttime = timeit.default_timer()
        sln, mks = msts(filename)
        print("Time taken for", filename, ":", timeit.default_timer() - starttime)
    """