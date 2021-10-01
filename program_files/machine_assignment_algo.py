import numpy as np
import networkx as nx

import msts_algo
import graph
from program_files.graph import get_node_info

# Find all machines capable of doing this operation
def find_eligible_machines(operation, machines_graph):
    # Gets nodes with attributes 'tasks'
    res = nx.get_node_attributes(machines_graph, 'tasks')
    #print(res)

    # Finds machines with 
    for machine, tasks in res.items():
        idx = np.where(tasks == operation[2])
        if idx[0].size != 0:
            #print("This machine", key, "has value", val[idx] , "with index", idx[0])
            print(graph.get_node_info(machines_graph, machine))
            yield machine

# Calculate machining time for specific operation-machine combination
# (MT = (PT * alpha) + (ST * beta))
def calculate_machining_time(operation, machine):
    return (operation[3] * machine[3]) + (operation[4] * machine[4])

# Get transition time from previous operation's machine to current
# operation's machine
def get_transition_time(operation, machine, trans_time):
    pass

# Append operation to the machine's schedule and,
# Store machine in operation's mach_num variable    <~ REWORD?
def assign_machine_to_operation(operation, machine, trans_time):
    # TODO: Add operation to machine's schedule
    pass


    #machining_time = calculate_machining_time(operation, machine)
    """     
    if len(machine[-1]) == 0:
        mach_ST = 0
        mach_FT = 0
    else:
        mach_ST = machine[-1][1]
        mach_FT = machine[-1][2]
    """

############################################################
#               MACHINE ASSIGNMENT ALGORITHMS              #
############################################################

# Randomly assign machine to all operations
def run_random(jobs_array, machine_graph, trans_time):
    for job in jobs_array:
        print(job)
        for operation in job:
            #eligible_machines = []
            eligible_machines = list(find_eligible_machines(operation, machine_graph))
            #print("Eligible Machine for Operation:", operation[1], operation[2])
            #print(eligible_machines)
            if len(eligible_machines) > 1:
                #print("length > 2")
                rnd_idx = np.random.choice(len(eligible_machines),size=1)
                machine = eligible_machines[int(rnd_idx)]
            else:
                #print("Length = 1")
                machine = eligible_machines[0]


            # TODO: CALL ASSIGN_MACHINE_TO_OPERATION(operation, machine, trans_time)

            #operation[-1] = machine[0]

    return 1

# Greedily assign machine to all operations
# TODO
def run_greedy(jobs_array, machines_array):
    pass

# Use adapted Dijkstra's algorithm to assign machine with
# shortest path for each job
# TODO
def run_shortest_path(jobs_array, machines_array):
    pass