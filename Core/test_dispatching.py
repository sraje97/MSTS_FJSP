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
    elif OS_algo == "ERT":
        #print(OS_algo)
        y = operation_scheduling.schedule_ERT(jobs_array, machine_graph)
        #print("Return ERT Schedule:", y)
        #print(graph.get_graph_info(machine_graph))
    else:
        #print(OS_algo)
        y = operation_scheduling.schedule_random(jobs_array, machine_graph)
        #print("Return ERT Schedule:", y)
        #print(graph.get_graph_info(machine_graph))
    
    return jobs_array, machine_graph

def msts(instances_file, save_dir):
    ############################################################
    #               INITIALIALISE JOBS + MACHINES              #
    ############################################################

    # TODO: Get as inputs
    MA_algo_choice = "LUM"
    OS_algo_choice = "lrmt"
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

    #machines_array.sort()
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

    curr_jobs = mydeepcopy(jobs_array)
    curr_graph = mydeepcopy(G)

    MSTS_start_time = timeit.default_timer()

    ## Get initial solution ##
    curr_jobs, curr_graph = initial_solution(curr_jobs, curr_graph, MA_algo_choice, OS_algo_choice)

    _, _, best_mks = calculate_makespan(curr_graph)
    
    # Create CSV to store best solution
    design_csv_path = os.path.join(save_dir, 'best_design.csv')
    fp_csv = open(design_csv_path, 'w', newline='')
    writer = csv.writer(fp_csv)

    writer.writerow(['Makespan', repr(best_mks)])
    writer.writerow(['OS Algo Choice', repr(OS_algo_choice), 'MS Algo Choice', repr(MA_algo_choice)])
    writer.writerow(['Epochs', repr(0), 'Time', round(timeit.default_timer() - MSTS_start_time, 3), \
                    'Probability', repr(p_exp_con)])

    writer.writerow(["Machine", "Machine_schedule"])

    schedule = graph.get_op_schedule(curr_graph)
    for key, val in schedule.items():
        writer.writerow([repr(key), repr(val)])
    
    writer.writerow(['Job', 'Operation'])
    
    sorted_jobs = tabu_search.flatten_job(curr_jobs)
    for oper in sorted_jobs:
        writer.writerow([oper.job_num, oper.op_num, oper.pre, oper.succ, oper.series, oper.machines, \
                        oper.mach_num, oper.processing_time, oper.transport_time, oper.finish_time])
    
    fp_csv.close()
    
    return (curr_jobs, curr_graph), best_mks


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
    """
    YFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
    DAFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', \
             '17', '18', '19', '20', '21', '23', '24', '28', '29', '30']
    SFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    MFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    MK = ['01', '02', '03', '04', '05']
    """

    YFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    DAFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', \
            '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
    SFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    MFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    MK = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    """
    test_name = "YFJS14.txt"
    starttime = timeit.default_timer()
    filename = "data\Benchmarks\T_Times\YFJS\\" + test_name
    sln, mks = msts(filename, save_dir)
    task_dict[test_name] = (mks, timeit.default_timer() - starttime)

    print("Time taken for", filename, ":", timeit.default_timer() - starttime, "Makespan:", mks)
    """
    """"""
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
    """"""

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