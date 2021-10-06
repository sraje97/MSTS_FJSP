import numpy as np
import networkx as nx
import operator

import msts_algo
import graph
from program_files.graph import get_node_info

bigM = 999_999

# Find all machines capable of doing this operation
def find_eligible_machines(operation, machine_graph):
    # Gets nodes with attributes 'tasks'
    res = nx.get_node_attributes(machine_graph, 'tasks')
    #print(res)

    # Finds machines with 
    for machine, tasks in res.items():
        idx = np.where(tasks == operation[2])
        if idx[0].size != 0:
            yield machine

# Calculate machining time for specific operation-machine combination
# (MT = (PT * alpha) + (ST * beta))
def calculate_machining_time(operation, machine_graph, machine):
    return round(((operation[3] * machine_graph.nodes[machine]['alpha']) + (operation[4] * machine_graph.nodes[machine]['beta'])), -1)

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

    nx.set_node_attributes(machine_graph, {machine: {'op_schedule': op_schedule } } )

def get_best_node(priority_queue):
    return min(priority_queue, key=operator.itemgetter(2))

def make_path_node(operation, machine_graph, machine, prev_machine, node):
    # Add machine to list of machines in path
    path_seq = node[1]
    path_seq.append(machine)

    # Calculate total machining and transition time
    #edge_cost = get_transition_time(machine_graph, prev_machine, machine)
    total_time = calculate_machining_time(operation, machine_graph, machine) + \
                get_transition_time(machine_graph, prev_machine, machine)

    path_node = (operation, path_seq, total_time)
    return path_node

def deal_with_parallel_P2_ops(job_array, op_num, machine_graph):
    # Get operation using the operation number
    operation = [item for item in job_array if item[1] == op_num][0]
    eligible_machines = find_eligible_machines(operation, machine_graph)

    # Large number
    minFMT = 999_999

    for machine in eligible_machines:
        machining_time = calculate_machining_time(operation, machine_graph, machine)
        if machining_time < minFMT:
            minFMT = machining_time
            best_machine = machine
    assign_machine_to_operation(operation, machine_graph, best_machine)


    while operation[7] != '':
        #not(np.isnan(operation)):
        minFMT = 999_999
        
        op_P2_succ = operation[7]
        operation = [item for item in job_array if item[1] == op_P2_succ]
        if operation.size() == 0:
            break
        if operation[8] == "P2":
            eligible_machines = find_eligible_machines(operation, machine_graph)

            for machine in eligible_machines:
                machining_time = calculate_machining_time(operation, machine_graph, machine)
                if machining_time < minFMT:
                    minFMT = machining_time
                    best_machine = machine
            assign_machine_to_operation(operation, machine_graph, best_machine)
        else:
            break
    


############################################################
#               MACHINE ASSIGNMENT ALGORITHMS              #
############################################################

# Randomly assign machine to all operations
def run_random(jobs_array, machine_graph):
    for job in jobs_array:
        #print(job)
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
            #print(machine_graph.nodes[machine])
            #print(operation)

    return 1

# Greedily assign machine to all operations using either 
# FMT - Fastest Machining Time (default) or LMT - Longest Machining Time
def run_greedy(jobs_array, machine_graph, greedy_type = "FMT"):
    for job in jobs_array:
        #print(job)
        for operation in job:
            # Get list of all eligible machines
            eligible_machines = list(find_eligible_machines(operation, machine_graph))
            #print("Eligible Machines for Operation (", operation[1], operation[2], "):", eligible_machines)

            if greedy_type == "FMT":
                # Use default FMT
                minFMT = bigM
                for machine in eligible_machines:
                    # Calculate machining time for operation on each eligible machine
                    machining_time = calculate_machining_time(operation, machine_graph, machine)
                    #print(machining_time)
                    if machining_time < minFMT:
                        # Assign the machine with FMT for operation
                        minFMT = machining_time
                        best_machine = machine
            else:
                # Use LMT
                maxFMT = 0
                for machine in eligible_machines:
                    machining_time = calculate_machining_time(operation, machine_graph, machine)
                    #print(machining_time)
                    if machining_time > maxFMT:
                        # Assign the machine with LMT for operation
                        maxFMT = machining_time
                        best_machine = machine
            
            assign_machine_to_operation(operation, machine_graph, best_machine)
            #print("Best machine...")
            #print(machine_graph.nodes[best_machine])

    return 1

# Use adapted Dijkstra's algorithm to assign machine with shortest path for each job
def run_shortest_path(jobs_array, machine_graph):
    for job in jobs_array:
        priority_queue = []
        final_best_node = (None, None, bigM)

        # Get first operation
        operation = job[0]
        eligible_machines = find_eligible_machines(operation, machine_graph)
        
        for machine in eligible_machines:
            # For each eligible machine add tuple of (operation, machince, total time)
            total_time = calculate_machining_time(operation, machine_graph, machine)
            path_node = (operation, [machine], total_time)
            priority_queue.append(path_node)

        while priority_queue.size() > 0:
            # Get node with the smallest total time so far
            best_node = get_best_node(priority_queue)
            priority_queue.remove(best_node)
            
            # Get next operation in sequence and it's assigned machine
            operation = best_node[0][7]
            prev_machine = best_node[1][-1]
            

            if type(operation) is tuple:
                op_P2 = [item for item in operation if item[8] == "P2"][0]

                deal_with_parallel_P2_ops(job, op_P2, machine_graph)
            
            # Add operations - bar the final operations of each job - to the priority queue
            # Insert the tuple of operation, path of machines taken, and total machining time so far
            elif operation[7] != '':
                # Get list of all eligible machines
                eligible_machines = find_eligible_machines(operation, machine_graph)
                for machine in eligible_machines:
                    # Create a path node (tuple) for operation+machine
                    path_node = make_path_node(operation, machine_graph, machine, prev_machine, best_node)

                    # Add path node (tuple) to the priority queue
                    priority_queue.append(path_node)
            else:
                # Get list of all eligible machines
                eligible_machines = find_eligible_machines(operation, machine_graph)
                for machine in eligible_machines:
                    # Create a path node (tuple) for operation+machine
                    path_node = make_path_node(operation, machine_graph, machine, prev_machine, best_node)
                    
                    # Store node with lowest total time
                    if path_node[2] < final_best_node[2]:
                        final_best_node = path_node
                    
                temp_list = [(final_best_node[0], final_best_node[1][-1])]
                final_best_node.pop()

