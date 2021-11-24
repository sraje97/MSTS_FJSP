import random
import operator
import networkx as nx

from graph import get_op_schedule
from msts_algo_new import calculate_makespan
from tabu_search import get_operation_job, recompute_times

# Convert a machine schedule to a sequence string containing a topological order of operations
def convert_to_str_seq(machine_graph):
    str_seq = []
    res = nx.get_node_attributes(machine_graph, 'op_schedule')

    # TODO: MAKE SURE PARALLEL PRECEDENCE CONSTRAINTS ARE NOT BEING VIOLATED
    # ESPECIALLY AFTER SORTING BY ST & MACH_NUM
    
    # Tuple : (job, op, mach, PT, ST, FT)
    # Then sort by starting time and mach, then finishing time
    for mach, schedule in res.items():
        for item in schedule:
            op_num = item[0]
            job_num = op_num[op_num.find("O") + 1 : op_num.find("_")]
            PT = item[2] - item[1]
            str_seq.append( [job_num, op_num, mach, PT, item[1], item[2]] )

    # Sort the sequence by starting time, then machine number, then finishing time (redundant)
    ret_seq = sorted(str_seq, key=operator.itemgetter(4, 2, 5))

    #print("String sequence:", ret_seq)

    return ret_seq

# Conduct Precedence-preserving crossover [see references]
def POX_crossover(P1, P2, num_jobs):
    # Create two jobsets
    jobset = list(range(1, num_jobs+1))
    random.shuffle(jobset)
    jobset1 = jobset[ : num_jobs // 2]
    jobset2 = jobset[num_jobs // 2 : ]

    # Create two offspring
    O1 = [None] * len(P1)
    O2 = [None] * len(P2)
    O1_idx = 0
    O2_idx = 0
    mach_dict = dict()
    
    # Tuple : (job, op, mach, PT, ST, FT)
    # Add jobset1 operations into offspring in position
    for i in range(len(P1)):
        if int(P1[i][0]) in jobset1:
            O1[i] = [ P1[i][0], P1[i][1], P1[i][2], P1[i][3], 0, 0 ]

        if int(P2[i][0]) in jobset1:
            O2[i] = [ P2[i][0], P2[i][1], P2[i][2], P2[i][3], 0, 0 ]

        if P1[i][2] not in mach_dict:
            mach_dict[P1[i][2]] = 0

    # Add jobset2 operations to fill remaining positions
    for i in range(len(P1)):
        while(O1[O1_idx] != None):
            O1_idx += 1
            if O1_idx == len(P1):
                O1_idx -= 1
                break
        while(O2[O2_idx] != None):
            O2_idx += 1
            if O2_idx == len(P1):
                O2_idx -= 1
                break

        if int(P2[i][0]) not in jobset1:
            O1[O1_idx] = [ P2[i][0], P2[i][1], P2[i][2], P2[i][3], 0, 0 ]
        if int(P1[i][0]) not in jobset1:
            O2[O2_idx] = [ P1[i][0], P1[i][1], P1[i][2], P1[i][3], 0, 0 ]

    return O1, O2

# Converts sequence string to jobs_array and machine graph with their attributes
def set_schedules(seq_str, jobs_array, machine_graph):
    for oper in seq_str:
        op_schedule = machine_graph.nodes[oper[2]]['op_schedule']
        op_schedule.append( ( oper[1], oper[4], oper[5] ) )

        op, _ = get_operation_job(jobs_array, oper[1])
        op.mach_num = oper[2]
        op.processing_time = oper[3]
    
    oper_schedules = get_op_schedule(machine_graph)

    # Set the operation's start/finish times given the set in machine schedules
    jobs_array, machine_graph = recompute_times(jobs_array, machine_graph, oper_schedules)

    # Deadlock occured hence skip this offspring
    if jobs_array == -1:
        makespan = -1
    else:
        _, _, makespan = calculate_makespan(machine_graph)

    return jobs_array, machine_graph, makespan

# Update the start and finish times in the sequence string
def update_times(seq_str, machine_graph):
    for oper in seq_str:
        op_schedule = machine_graph.nodes[oper[2]]['op_schedule']
        idx = [i for i, tupl in enumerate(op_schedule) if tupl[0] == oper[1]][-1]

        oper[4] = op_schedule[idx][1]
        oper[5] = op_schedule[idx][2]

    return seq_str