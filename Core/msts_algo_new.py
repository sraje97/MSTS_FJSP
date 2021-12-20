############################################################

## IMPORT PYTHON PACKAGES
import os
import csv
import sys
import pickle
import pstats
import random
import timeit
import cProfile
import operator

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

from copy import deepcopy
from datetime import datetime
from prettytable import PrettyTable

## IMPORT OUR MODULES
import graph
import crossover
import preprocess
import tabu_search
import machine_assignment
import operation_scheduling

# Sets base directory one level higher than current file (@ X:\\..\\MSTS_FJSP)
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
# Add base directory and the data directory to the system path
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'data'))

############################################################

def get_time_stamp():
    now = datetime.now()
    year = now.strftime('%Y')
    month = now.strftime('%m')
    day = now.strftime('%d')
    hour = now.strftime('%H')
    minute = now.strftime('%M')
    second = now.strftime('%S')
    return '{}-{}-{}-{}-{}-{}'.format(day, month, year, hour, minute, second)

# Check if the particular tabu tuple is in tabu list
def check_tabu_status(oper, mach, tabu_list):
    for tupl in tabu_list:
        if tupl[0] == oper and tupl[1] == mach:
            # Operation + Machine in tabu list
            return True
    return False

# Generate a random tabu tenure between the bounds
def generate_tenure(num_job, num_mach):
    Lmin = 10 + (num_job / num_mach)
    Lmax = 1.8 * Lmin
    return np.random.randint(low=Lmin, high=Lmax)

# Generate random pairs of individuals for crossover
def generate_random_pairs(pop_list, n=2):
    random.shuffle(pop_list)
    result = []
    for i in range(0, len(pop_list), n):
        result.append(pop_list[i:i+n])
    return result

# Calculate the makespan from the machine schedules
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

def print_job_info(jobs_array):
    table = PrettyTable(['Job', 'Operation', 'Pre', 'Succ', 'Series', 'Machines', 'Assigned Machine'])
    for job in jobs_array:
        for op in job:
            table.add_row([op.job_num, op.op_num, op.pre, op.succ, op.series, op.machines, op.mach_num])
    print(table)

# Create a gantt chart of the current solution (machine schedules)
def create_gantt_chart(machine_graph):
    op_schedules = []
    res = nx.get_node_attributes(machine_graph, 'op_schedule')

    op_schedules = []
    mach_names = []
    job_names = []

    for mach, schedule in res.items():
        if mach not in mach_names:
            mach_names.append(mach)
        for tupl in schedule:
            op_job = "J" + tupl[0][tupl[0].find("O") + 1 : tupl[0].find("_")]
            if op_job not in job_names:
                job_names.append(op_job)
            temp_dict = dict(Job=op_job, Start=tupl[1], Finish=tupl[2], Machine=mach, Details=tupl[0])
            op_schedules.append(temp_dict)

    mach_names = sorted(mach_names, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    job_names = sorted(job_names, key=lambda x: int("".join([i for i in x if i.isdigit()])))

    gantt_df = pd.DataFrame(op_schedules)
    gantt_df['Delta'] = gantt_df['Finish'] - gantt_df['Start']

    # Colorscale for colors.sequential and colors.qualitative https://plotly.com/python/builtin-colorscales/
    # Good colours: Teal, Tealgrn, darkcyan, slategray, Plotly
    fig = px.timeline(gantt_df, x_start="Start", x_end="Finish", y="Machine", color="Job", text="Details",
                    color_discrete_sequence= px.colors.qualitative.Plotly,
                    #color_discrete_sequence=["lightslategray"],
                    category_orders= { "Machine" : mach_names, "Job" : job_names }
                    )
    #fig.update_yaxes(autorange="reversed")
    fig.layout.xaxis.type = 'linear'
    for d in fig.data:
        filt = gantt_df['Job'] == d.name
        d.x = gantt_df[filt]['Delta'].tolist()
    fig.show()

def mydeepcopy(obj):
    try:
        #print("pickling")
        return pickle.loads(pickle.dumps(obj, -1))
    except pickle.PicklingError:
        #print("deepcopy")
        return deepcopy(obj)

def initial_solution(jobs_array, machine_graph, MA_algo, OS_algo):
    ### MACHINE ASSIGNMENT ###
    # Random machine assignment
    if MA_algo.lower() == 'random':
        #print(MA_algo)
        x = machine_assignment.assign_random(jobs_array, machine_graph)
        #print("Return Random", x)
        #print(graph.get_graph_info(machine_graph))
    # Greedy machine assignment
    elif MA_algo.lower() == 'greedy':
        #print(MA_algo)
        x = machine_assignment.assign_greedy(jobs_array, machine_graph, "FMT")
        #print("Return Greedy", x)
        #print(graph.get_graph_info(machine_graph))
    # Shortest Path machine assignment
    # Greedy machine assignment
    elif MA_algo.lower() == 'lum':
        #print(MA_algo)
        x = machine_assignment.assign_LUM(jobs_array, machine_graph)
        #print("Return LUM", x)
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

def msts(instances_file, save_dir):
    ############################################################
    #               INITIALIALISE JOBS + MACHINES              #
    ############################################################

    # TODO: Get as inputs
    MA_algo_choice = "LUM"
    OS_algo_choice = "LRMT"
    print(OS_algo_choice)

    # Add test instance name to the save directory name
    basename = os.path.basename(instances_file)
    save_dir = os.path.join(save_dir, basename[:-4])
    try:
        # Create subdirectory for test instance
        os.makedirs(save_dir, exist_ok=True)
    except OSError:
        pass

    #swap_methods = ["Random MA", "LPT MA", "HMW MA", "Random OS", "HMW OS"]
    swap_methods = ["Critical Path MA", "Critical Path OS"]
    TS_cnt_max = 10
    p_exp_con = 1.0
    p_MA_OS = 1.0
    epochs = 50
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 0.5
    pop_size = 50
    population = []

    TS_mks = []
    local_mks = []
    pop_mks = []
    TS_itrs = 0

    #random.seed(1)
    #np.random.seed(1)

    # Load operation's information from instance's text file
    jobs_array, machines_array, trans_times = preprocess.initialise_operations(instances_file)

    # Label the branches of the parallel paths and get precedence order dataframe
    jobs_array, op_df = preprocess.label_parallel_branches(jobs_array)

    machines_array.sort()
    G = nx.Graph()
    for machine in machines_array:
        #print(machine)
        graph.add_node(G, machine)
    
    if trans_times is None:
        trans_times = np.zeros((len(machines_array), len(machines_array)))
        graph.add_edge_costs(G, trans_times, machines_array)
    else:
        graph.add_edge_costs(G, trans_times, machines_array)
    
    # If machine assignment algorithm not pre-specified, choose randomly
    if MA_algo_choice == "":
        MA_algo_choice = np.random.choice(['Random', 'Greedy', 'Shortest Path'], p=[0.35, 0.35, 0.3])
    if OS_algo_choice == '':
        OS_algo_choice = np.random.choice(['SMT', 'LRMT', 'ERT']) #, p=[0.35, 0.35, 0.3])

    # Generate a population of random individuals
    pop_mks_sum = 0
    for i in range(pop_size):
        curr_jobs = mydeepcopy(jobs_array)
        curr_graph = mydeepcopy(G)
        curr_jobs, curr_graph = initial_solution(curr_jobs, curr_graph, "Random", OS_algo_choice)
        individual = crossover.convert_to_str_seq(curr_graph)
        population.append(individual)
        _, _, mks = calculate_makespan(curr_graph)
        pop_mks_sum += mks
    pop_mks.append( pop_mks_sum / pop_size)

    curr_jobs = mydeepcopy(jobs_array)
    curr_graph = mydeepcopy(G)

    ## Get initial solution ##
    curr_jobs, curr_graph = initial_solution(curr_jobs, curr_graph, MA_algo_choice, OS_algo_choice)

    # Create a copy of the current solution (jobs and machine graph)
    curr_solution = (mydeepcopy(curr_jobs), mydeepcopy(curr_graph))
    local_best_sln = mydeepcopy(curr_solution)
    global_best_sln = mydeepcopy(curr_solution)
    _, _, local_best_mks = calculate_makespan(curr_solution[1])
    global_best_mks = local_best_mks
    local_mks.append(local_best_mks)
    TS_mks.append(local_best_mks)

    # Sort operation schedule
    graph.sort_op_schedule(curr_graph)

    ############################################################
    #                  BEGIN MSTS ALGORITHM                    #
    ############################################################

    e_cnt = 0           # Epoch counter
    TS_cnt = 0          # Tabu Search counter
    global_improved = 0 # Global improvement counter
    tabu_list = []

    # Initialise saves folders and files
    #fp_log = open(os.path.join(save_dir,  'log.txt'), 'w')
    #fp_log.write(MA_algo_choice, "\t", OS_algo_choice)
    #fp_log.close()

    MSTS_start_time = timeit.default_timer()
    
    while e_cnt < epochs:
        # Terminate program after 1 hour
        if (timeit.default_timer() - MSTS_start_time) > 3600: #TODO: Change to 3600
            print("Overtime! Epochs:", e_cnt)
            break

        ## LOCAL OPTIMIZATION ##

        # Get swap method using probability
        p = np.random.random()
        if p < p_MA_OS:
        #if TS_cnt < 6:
            swap_method = "Critical Path MA"
        else:
            swap_method = "Critical Path OS"

        # Neighbourhood Tuple : (jobs, graph, oper, mach, mks)
        neighbourhood = tabu_search.tabu_move(curr_solution[0], curr_solution[1], op_df, swap_method)
        TS_itrs += 1
        if neighbourhood:
            best_neighbourhood = neighbourhood[0]

            if TS_itrs % 5 == 0:
                TS_mks.append(best_neighbourhood[-1])
            
            # Place solution in population if makespan better than a current individual
            for i in range(pop_size):
                individual = max(population[i], key=operator.itemgetter(5))
                if best_neighbourhood[-1] < individual[-1]:
                    new_ind = crossover.convert_to_str_seq(best_neighbourhood[1])
                    population[i] = new_ind
                    break

            # Tabu Tuple outline: (operation, machine, tabu_tenure)
            tabu_tuple = (best_neighbourhood[2], best_neighbourhood[3])

            # If there is an improvement in the makespan, keep the best neighbourhood solution
            if best_neighbourhood[-1] < local_best_mks:
                if tabu_list:
                    for tupl in tabu_list:
                        if tupl[0] == tabu_tuple[0] and tupl[1] == tabu_tuple[1]:
                            tabu_list.remove(tupl)
                            break
                
                tenure = generate_tenure( len(best_neighbourhood[0]), graph.get_number_of_nodes(best_neighbourhood[1]) )

                tabu_tuple = (best_neighbourhood[2], best_neighbourhood[3], tenure)

                tabu_list.append(tabu_tuple)

                curr_solution = (mydeepcopy(best_neighbourhood[0]), mydeepcopy(best_neighbourhood[1]))
                local_best_mks = best_neighbourhood[-1]
                local_best_sln = (mydeepcopy(best_neighbourhood[0]), mydeepcopy(best_neighbourhood[1]))

                TS_cnt = 0
            else:
                # If the best_neighbour solution doesn't improve makespan, check this or other neighbours
                if tabu_list:
                    # If tabu list exists check for non-tabu neighbourhood solution
                    if check_tabu_status(tabu_tuple[0], tabu_tuple[1], tabu_list):
                        all_tabu = False
                        # Go through neighbours until we find a non-tabu solution
                        for neighbour in neighbourhood:
                            if check_tabu_status(neighbour[2], neighbour[3], tabu_list):
                                if neighbour == neighbourhood[-1]:
                                    all_tabu = True
                                continue
                            tenure = generate_tenure( len(neighbour[0]), graph.get_number_of_nodes(neighbour[1]) )
                            tabu_tuple = (neighbour[2], neighbour[3], tenure)
                            curr_solution = (mydeepcopy(neighbour[0]), mydeepcopy(neighbour[1]))
                            break
                        if all_tabu:
                            #print("All solutions tabooed!")
                            # Store the best neighbour in case all neighbours are tabooed
                            tenure = generate_tenure( len(best_neighbourhood[0]), graph.get_number_of_nodes(best_neighbourhood[1]) )
                            tabu_tuple = (best_neighbourhood[2], best_neighbourhood[3], tenure)
                            curr_solution = (mydeepcopy(best_neighbourhood[0]), mydeepcopy(best_neighbourhood[1]))
                    else:
                        # If best neighbourhood solution is non-tabu solution store it as the solution
                        tenure = generate_tenure( len(best_neighbourhood[0]), graph.get_number_of_nodes(best_neighbourhood[1]) )
                        tabu_tuple = (best_neighbourhood[2], best_neighbourhood[3], tenure)
                    
                    for tupl in tabu_list:
                        if tupl[0] == tabu_tuple[0] and tupl[1] == tabu_tuple[1]:
                            tabu_list.remove(tupl)
                            break
                    tabu_list.append(tabu_tuple)

                else:
                    # If no tabu_list then just add this straight away
                    tenure = generate_tenure( len(best_neighbourhood[0]), graph.get_number_of_nodes(best_neighbourhood[1]) )
                    tabu_tuple = (best_neighbourhood[2], best_neighbourhood[3], tenure)
                    tabu_list.append(tabu_tuple)
                    curr_solution = (mydeepcopy(best_neighbourhood[0]), mydeepcopy(best_neighbourhood[1]))

                TS_cnt += 1
                # If no improve with MA lower the probability of choosing MA
                # Else, increase the probability of choosing MA
                if swap_method == "Critical Path MA":
                    p_MA_OS -= 0.25
                else:
                    p_MA_OS += 0.25
                # p_MA_OS must be within the bounds of 0 and 1
                p_MA_OS = max(0, min(p_MA_OS, 1))
        else:
            TS_cnt += 1
            curr_solution = (mydeepcopy(local_best_sln[0]), mydeepcopy(local_best_sln[1]))
        
        # Decrement the tenure of each tabu solution
        i=0
        while i < (len(tabu_list)):
            tabu_tuple = list(tabu_list[i])
            tabu_tuple[-1] -= 1
            tabu_list[i] = tuple(tabu_tuple)
            if tabu_list[i][-1] == 0:
                tabu_list.remove(tabu_list[i])
                i -= 1
            i += 1

        if TS_cnt == TS_cnt_max:
            p = np.random.random()
            local_mks.append(local_best_mks)

            if local_best_mks < global_best_mks:
                global_best_mks = local_best_mks
                global_best_sln = mydeepcopy(local_best_sln)
                global_improved = 0
            else:
                global_improved += 1
                #print("Local:", global_best_mks)
            
            ## GLOBAL OPTIMIZATION ##
            # POX Crossover of individuals in the population
            pop_mks_sum = 0
            shuffled_pairs = generate_random_pairs(population, 2)
            for P1, P2 in shuffled_pairs:
                O1, O2 = crossover.POX_crossover(P1, P2, jobs_array)

                O1_jobs, O1_graph, O1_mks = crossover.set_schedules(O1, mydeepcopy(jobs_array), mydeepcopy(G))
                O2_jobs, O2_graph, O2_mks = crossover.set_schedules(O2, mydeepcopy(jobs_array), mydeepcopy(G))

                # If both offspring received deadlock when setting schedule then skip
                # Else, insert the offspring with the lower makespan to the population
                if (O1_mks == -1) and (O2_mks == -1):
                    continue
                elif (O1_mks == -1):
                    # Set O2 as offspring in population
                    O2 = crossover.update_times(O2, O2_graph)
                    population.append(O2)
                    pop_mks_sum += O2_mks
                elif (O2_mks == -1):
                    # Set O1 as offspring in population
                    O1 = crossover.update_times(O1, O1_graph)
                    population.append(O1)
                    pop_mks_sum += O1_mks
                else:
                    if O1_mks < O2_mks:
                        # Set O1 as offspring in population
                        O1 = crossover.update_times(O1, O1_graph)
                        population.append(O1)
                        pop_mks_sum += O1_mks
                    else:
                        # Set O2 as offspring in population
                        O2 = crossover.update_times(O2, O2_graph)
                        population.append(O2)
                        pop_mks_sum += O2_mks

                # If the new individual's makespan is better than the global best, save this
                if (O1_mks < global_best_mks) and (O1_mks > 0):
                    global_best_sln = (O1_jobs, O1_graph)
                    global_best_mks = O1_mks
                    #print("O1:", global_best_mks)
                if (O2_mks < global_best_mks) and (O2_mks > 0):
                    global_best_sln = (O2_jobs, O2_graph)
                    global_best_mks = O2_mks
                    #print("O2:", global_best_mks)

                P1_mks = max(P1, key=operator.itemgetter(5))
                P2_mks = max(P2, key=operator.itemgetter(5))

                # Delete the higher makespan parent
                if P1_mks[-1] < P2_mks[-1]:
                    population.remove(P2)
                    pop_mks_sum += P1_mks[-1]
                    # NOTE: Seems redundant but confirm below code again
                    if (P1_mks[-1] < global_best_mks) and (P1_mks[-1] > 0):
                        P1_jobs, P1_graph, global_best_mks = crossover.set_schedules(P1, mydeepcopy(jobs_array), mydeepcopy(G))
                        global_best_sln = (P1_jobs, P1_graph)
                        #global_best_sln[0], global_best_sln[1], global_best_mks = crossover.set_schedules(P1, mydeepcopy(jobs_array), mydeepcopy(G))
                else:
                    population.remove(P1)
                    pop_mks_sum += P2_mks[-1]
                    # NOTE: Seems redundant but confirm again
                    if (P2_mks[-1] < global_best_mks) and (P2_mks[-1] > 0):
                        P2_jobs, P2_graph, global_best_mks = crossover.set_schedules(P2, mydeepcopy(jobs_array), mydeepcopy(G))
                        global_best_sln = (P2_jobs, P2_graph)
                        #global_best_sln[0], global_best_sln[1], global_best_mks = crossover.set_schedules(P2, mydeepcopy(jobs_array), mydeepcopy(G))

            pop_mks.append(pop_mks_sum / pop_size)

            # Choose new current solution to be one of:
            # 1. Best global solution (likelihood increases as time passes)
            # 2. New random initial solution (static 5% chance)
            # 3. Random solution from the population (high chance early but lower as time passes)
            if p > p_exp_con:
                # Choose the global best solution
                curr_solution = mydeepcopy(global_best_sln)
            elif p < 0.05:
                # Generate new initial solution
                curr_jobs, curr_graph = initial_solution(mydeepcopy(jobs_array), mydeepcopy(G), MA_algo_choice, OS_algo_choice)
                curr_solution = (curr_jobs, curr_graph)
            else:
                # Choose random individual from the population
                individual = random.choice(population)
                curr_jobs, curr_graph, _ = crossover.set_schedules(individual, mydeepcopy(jobs_array), mydeepcopy(G))
                curr_solution = (curr_jobs, curr_graph)

            p_exp_con = eps_end + (eps_start - eps_end) * np.exp(-1.0 * e_cnt / epochs / eps_decay)
            TS_cnt = 0
            e_cnt += 1
            # If no improvement on the global solution after epochs/2 iterations, then terminate
            if global_improved >= epochs/2:
                print("No improvement... Breaking...")
                break

    # Get the best found solution and log it's information
    _, _, curr_best_mks = calculate_makespan(curr_solution[1])
    _, _, local_best_mks = calculate_makespan(local_best_sln[1])
    _, _, global_best_mks = calculate_makespan(global_best_sln[1])
    if curr_best_mks < local_best_mks:
        if curr_best_mks < global_best_mks:
            global_best_sln = curr_solution
            global_best_mks = curr_best_mks
    else:
        if local_best_mks < global_best_mks:
            global_best_sln = local_best_sln
            global_best_mks = local_best_mks
    
    # Create CSV to store best solution
    design_csv_path = os.path.join(save_dir, 'best_design.csv')
    fp_csv = open(design_csv_path, 'w', newline='')
    writer = csv.writer(fp_csv)

    writer.writerow(['Makespan', repr(global_best_mks)])
    writer.writerow(['OS Algo Choice', repr(OS_algo_choice), 'MS Algo Choice', repr(MA_algo_choice)])
    writer.writerow(['Epochs', repr(e_cnt), 'Time', round(timeit.default_timer() - MSTS_start_time, 3), \
                    'Probability', repr(p_exp_con)])

    writer.writerow(["Machine", "Machine_schedule"])

    #schedule = nx.get_node_attributes(global_best_sln[1], 'op_schedules')
    schedule = graph.get_op_schedule(global_best_sln[1])
    for key, val in schedule.items():
        writer.writerow([repr(key), repr(val)])
    
    writer.writerow(['Job', 'Operation'])
    
    sorted_jobs = tabu_search.flatten_job(global_best_sln[0])
    #for job in sorted_jobs:
        #for oper in job:
    for oper in sorted_jobs:
        writer.writerow([oper.job_num, oper.op_num, oper.pre, oper.succ, oper.series, oper.machines, \
                        oper.mach_num, oper.processing_time, oper.transport_time, oper.finish_time])
    
    fp_csv.close()

    # Keep log of TS iteration's makespan
    design_csv_path = os.path.join(save_dir, 'log_TS.csv')
    fp_csv = open(design_csv_path, 'w', newline='')
    writer = csv.writer(fp_csv)

    writer.writerow(['Iteration', 'Best Neighbourhood Makespan'])
    for i in range(len(TS_mks)):
        writer.writerow([i*5, TS_mks[i]])
    fp_csv.close()

    # Keep log of local and population's makespans
    design_csv_path = os.path.join(save_dir, 'log_epoch.csv')
    fp_csv = open(design_csv_path, 'w', newline='')
    writer = csv.writer(fp_csv)

    writer.writerow(['Epoch', 'Local Best Makespan', 'Population Average Makespan'])
    for i in range(len(local_mks)):
        writer.writerow([i, local_mks[i], pop_mks[i]])
    fp_csv.close()
    
    return global_best_sln, global_best_mks


### BEGIN MAIN PROGRAM ###
if __name__ == '__main__':

    PCname = os.environ['COMPUTERNAME']

    # Output folder to save to
    save_dir = os.path.join(base_dir, 'output_models/', get_time_stamp() + '_' + PCname)

    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError:
        pass

    # Create output argument text file
    #fp = open(os.path.join(save_dir, 'args.txt'), 'w')
    #fp.write(str("THESE WERE THE INPUT ARGUMENTS"))
    #fp.close()
    task_dict = dict()

    # Enable Stats profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # List of test instances
    YFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
    DAFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', \
             '17', '18', '19', '20', '21', '23', '24', '28', '29', '30']
    SFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    MFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    MK = ['01', '02', '03', '04', '05']

    """"""
    test_name = "MK111.txt"
    starttime = timeit.default_timer()
    filename = "data\Benchmarks\BR\\" + test_name
    sln, mks = msts(filename, save_dir)
    task_dict[test_name] = (mks, timeit.default_timer() - starttime)

    print("Time taken for", filename, ":", timeit.default_timer() - starttime, "Makespan:", mks)
    """"""
    """
    print("## YFJS: ##")
    for file_num in YFJS:
        test_name = "YFJS" + file_num + ".txt"
        filename = "data\Benchmarks\YFJS\\" + test_name
        starttime = timeit.default_timer()
        print(starttime)
        sln, mks = msts(filename, save_dir)
        task_dict[test_name] = (mks, timeit.default_timer() - starttime)
        print("Time taken for", filename, ":", timeit.default_timer() - starttime, "Makespan:", mks)
    """
    """
    print("## DAFJS: ##")
    for file_num in DAFJS:
        test_name = "DAFJS" + file_num + ".txt"
        filename = "data\Benchmarks\DAFJS\\" + test_name
        starttime = timeit.default_timer()
        sln, mks = msts(filename, save_dir)
        task_dict[test_name] = (mks, timeit.default_timer() - starttime)
        print("Time taken for", filename, ":", timeit.default_timer() - starttime, "Makespan:", mks)
    """
    """
    print("## SFJS: ##")
    for file_num in SFJS:
        test_name = "sfjs" + file_num + ".txt"
        filename = "data\Benchmarks\FMJ\\" + test_name
        starttime = timeit.default_timer()
        sln, mks = msts(filename, save_dir)
        task_dict[test_name] = (mks, timeit.default_timer() - starttime)
        print("Time taken for", filename, ":", timeit.default_timer() - starttime, "Makespan:", mks)
    """
    """
    print("## MFJS: ##")
    for file_num in MFJS:
        test_name = "mfjs" + file_num + ".txt"
        filename = "data\Benchmarks\FMJ\\" + test_name
        starttime = timeit.default_timer()
        sln, mks = msts(filename, save_dir)
        task_dict[test_name] = (mks, timeit.default_timer() - starttime)
        print("Time taken for", filename, ":", timeit.default_timer() - starttime, "Makespan:", mks)
    """
    """
    print("## MK: ##")
    for file_num in MK:
        test_name = "MK" + file_num + ".txt"
        filename = "data\Benchmarks\BR\\" + test_name
        starttime = timeit.default_timer()
        sln, mks = msts(filename, save_dir)
        task_dict[test_name] = (mks, timeit.default_timer() - starttime)
        print("Time taken for", filename, ":", timeit.default_timer() - starttime, "Makespan:", mks)
    """

    # Keep log of test cases makespan and times
    design_csv_path = os.path.join(save_dir, 'TestCases.csv')
    fp_csv = open(design_csv_path, 'w', newline='')
    writer = csv.writer(fp_csv)

    writer.writerow(['Test Instance', 'Makespan', 'Runtime'])
    for key, val in task_dict.items():
        writer.writerow([key, val[0], val[1]])
    fp_csv.close()

    # Disable Stats profiler
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()