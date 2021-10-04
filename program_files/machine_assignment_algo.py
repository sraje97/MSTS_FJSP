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
            yield machine

# Calculate machining time for specific operation-machine combination
# (MT = (PT * alpha) + (ST * beta))
def calculate_machining_time(operation, machine_graph, machine):
    return ((operation[3] * machine_graph.nodes[machine]['alpha']) + (operation[4] * machine_graph.nodes[machine]['beta']))

# Get transition time from previous operation's machine to current
# operation's machine
def get_transition_time(machine_graph, machA, machB):
    return machine_graph.edges[machA, machB]['weight']

# Append operation to the machine's schedule and,
# Store machine in operation's "mach_num" variable
def assign_machine_to_operation(operation, machine_graph, machine):
    # Add machine label to operation
    operation[-1] = str(machine)

    # Add operation to machine/node's attribute
    # Don't assign starting and finishing time yet
    op_schedule = machine_graph.nodes[machine]['op_schedule']
    op_schedule.append((operation[1], 0, 0))

    print("New:", op_schedule)

    nx.set_node_attributes(machine_graph, {machine: {'op_schedule': op_schedule } } )


############################################################
#               MACHINE ASSIGNMENT ALGORITHMS              #
############################################################

# Randomly assign machine to all operations
def run_random(jobs_array, machine_graph):
    for job in jobs_array:
        print(job)
        for operation in job:
            # Get list of all eligible machines
            eligible_machines = list(find_eligible_machines(operation, machine_graph))
            #print("Eligible Machines for Operation (", operation[1], operation[2], "):", eligible_machines)

            if len(eligible_machines) > 1:
                # Choose any random machine if more than one compatible machine
                rnd_idx = np.random.choice(len(eligible_machines),size=1)
                machine = eligible_machines[int(rnd_idx)]
            else:
                # Choose the only eligible machine
                machine = eligible_machines[0]
            #print(machine)

            # Assign the machine to the operation (and vice versa)
            assign_machine_to_operation(operation, machine_graph, machine)
            print(machine_graph.nodes[machine])
            print(operation)

    return 1

# Greedily assign machine to all operations using either 
# FMT - Fastest Machining Time (default) or LMT - Longest Machining Time
def run_greedy(jobs_array, machine_graph, greedy_type = "FMT"):
    for job in jobs_array:
        print(job)
        for operation in job:
            # Get list of all eligible machines
            eligible_machines = list(find_eligible_machines(operation, machine_graph))
            #print("Eligible Machines for Operation (", operation[1], operation[2], "):", eligible_machines)

            if greedy_type == "FMT":
                # Use default FMT
                minFMT = 999999
                for machine in eligible_machines:
                    # Calculate machining time for operation on each eligible machine
                    machining_time = calculate_machining_time(operation, machine_graph, machine)
                    print(machining_time)
                    if machining_time < minFMT:
                        # Assign the machine with FMT for operation
                        minFMT = machining_time
                        best_machine = machine
            else:
                # Use LMT
                maxFMT = 0
                for machine in eligible_machines:
                    machining_time = calculate_machining_time(operation, machine_graph, machine)
                    print(machining_time)
                    if machining_time > maxFMT:
                        # Assign the machine with LMT for operation
                        maxFMT = machining_time
                        best_machine = machine
            
            assign_machine_to_operation(operation, machine_graph, best_machine)
            print("Best machine...")
            print(machine_graph.nodes[best_machine])

    return 1

# Use adapted Dijkstra's algorithm to assign machine with
# shortest path for each job
# TODO
def run_shortest_path(jobs_array, machines_array):
    pass