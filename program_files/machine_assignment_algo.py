import numpy as np
import networkx as nx
import operator
import copy

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

def get_min_FMT(operation, machine_graph, eligible_machines):
    minFMT = bigM

    for machine in eligible_machines:
        # Calculate machining time for operation on each eligible machine
        machining_time = calculate_machining_time(operation, machine_graph, machine)
        if machining_time < minFMT:
            # Assign the machine with FMT for operation
            minFMT = machining_time
            best_machine = machine
    # Assign machine to operation
    assign_machine_to_operation(operation, machine_graph, best_machine)

# Get node with lowest total time
def get_best_node(priority_queue):
    return min(priority_queue, key=operator.itemgetter(2))

# Create a node/tuple (operation, path sequence, total time)
def make_path_node(operation, machine_graph, machine, prev_machine, node):
    # Add machine to list of machines in path
    path_seq = node[1]
    path_seq.append(machine)

    # Calculate total machining and transition time and previous time
    total_time = calculate_machining_time(operation, machine_graph, machine) + \
                get_transition_time(machine_graph, prev_machine, machine) + node[2]

    # Create node/tuple
    path_node = (operation, path_seq, total_time)
    return path_node

# Assign all succeeding P2 parallel operations using Greedy algorithm
def assign_parallel_operation_succ(job_array, op_num, machine_graph):
    # Get operation using the operation number
    operation = [item for item in job_array if item[1] == op_num][0]
    eligible_machines = find_eligible_machines(operation, machine_graph)

    # Assign operation to machine using Greedy algorithm
    get_min_FMT(operation, machine_graph, eligible_machines)

    # Assign any succeeding P2 parallel operations as well
    while operation[7] != '':
        op_P2_succ = operation[7]
        operation = [item for item in job_array if item[1] == op_P2_succ][0]

        # Only assign successor operations which are P2
        if operation[8] == "P2":
            eligible_machines = find_eligible_machines(operation, machine_graph)

            # Assign operation to machine using Greedy algorithm
            get_min_FMT(operation, machine_graph, eligible_machines)
        else:
            break

# Assign all preceding P2 parallel operations using Greedy algorithm
def assign_parallel_operation_pre(job_array, op_num, machine_graph):
    # Get operation using the operation number
    operation = [item for item in job_array if item[1] == op_num][0]
    eligible_machines = find_eligible_machines(operation, machine_graph)

    # Assign operation to machine using Greedy algorithm
    get_min_FMT(operation, machine_graph, eligible_machines)

    # Assign any preceding P2 parallel operations as well
    while operation[6] != '':
        op_P2_pre = operation[6]
        operation = [item for item in job_array if item[1] == op_P2_pre][0]
        
        # Only assign successor operations which are P2
        if operation[8] == "P2":
            eligible_machines = find_eligible_machines(operation, machine_graph)

            # Assign operation to machine using Greedy algorithm
            get_min_FMT(operation, machine_graph, eligible_machines)
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
                get_min_FMT(operation, machine_graph, eligible_machines)
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
        # Create priority queue for node/tuple (operation, path sequence, total time)
        priority_queue = []
        # Use to store best final node
        best_final_node = (None, None, bigM)

        # Get first operation of job
        operation = job[0]
        eligible_machines = find_eligible_machines(operation, machine_graph)
        
        for machine in eligible_machines:
            # For each eligible machine add tuple of (operation, machince, total time) 
            # to the priority queue
            total_time = calculate_machining_time(operation, machine_graph, machine)
            path_node = (operation, [machine], total_time)
            priority_queue.append(path_node)

        # Continuously loop through the priority queue until we get to the final operation
        while len(priority_queue) > 0:
            # Get node with the smallest total time in queue
            best_node = get_best_node(priority_queue)
            idx = [i for i, tupl in enumerate(priority_queue) if tupl[1] == best_node[1]][-1]
            # Pop node from queue
            priority_queue.pop(idx)
            
            # Get next operation in sequence and the current operation's assigned machine
            op_succ = best_node[0][7]
            prev_machine = best_node[1][-1]
            
            # Assign any parallel P2 operations (after disassembly/split)
            if type(op_succ) is tuple:
                # Get the P2 operation - P2 operation after P1 operation in tuple
                op_P2_succ = op_succ[1]

                # Assign the P2 operation to a machine
                assign_parallel_operation_succ(job, op_P2_succ, machine_graph)
                # Get the P! operation of the tuple
                op_succ = op_succ[0]

            operation = [item for item in job if item[1] == op_succ][0]
            
            # Get previous operation and check if tuple
            op_pre = operation[6]
            # Assign any previous parallel P2 operations (before assembly/join)
            if type(op_pre) is tuple:
                op_P2_pre = op_pre[1]

                # Assign the P2 operation to a machine
                assign_parallel_operation_pre(job, op_P2_pre, machine_graph)

            # Add operations - bar the final operations of each job - to the priority queue
            if operation[7] != '':
                eligible_machines = find_eligible_machines(operation, machine_graph)
                for machine in eligible_machines:
                    temp_node = copy.deepcopy(best_node)
                    # Create a path node (tuple) for operation + machine/path sequence + total time
                    path_node = make_path_node(operation, machine_graph, machine, prev_machine, temp_node)

                    # Add path node (tuple) to the priority queue
                    priority_queue.append(path_node)
            else:
                eligible_machines = find_eligible_machines(operation, machine_graph)
                for machine in eligible_machines:
                    # Create a path node (tuple) for operation + machine/path sequence + total time
                    path_node = make_path_node(operation, machine_graph, machine, prev_machine, best_node)
                    
                    # Store node with lowest total time
                    if path_node[2] < best_final_node[2]:
                        best_final_node = path_node
                
                # Add best operation + machine to assignment list
                assignment_list = [(best_final_node[0], best_final_node[1][-1])]
                best_final_node[1].pop()

                # Backtrack through the machine/path sequence and add operation + machine 
                # to the assignment list for all P1 & S operations in job
                for idx in range(len(best_final_node[1])):
                    last_machine = best_final_node[1][-1]
                    last_operation = best_final_node[0][6]

                    # If preceding parallel operations, append only the P1 or S operations
                    if type(last_operation) is tuple:
                        last_operation = last_operation[0]
                        operation = [item for item in job if item[1] == last_operation][0]
                        assignment_list.append((operation, last_machine))

                        # Remove machine from path sequence and get preceding operation
                        temp_best_node = list(best_final_node)
                        temp_best_node[0] = operation
                        best_final_node = tuple(temp_best_node)
                        best_final_node[1].pop()
                    elif last_operation != '':
                        operation = [item for item in job if item[1] == last_operation][0]
                        assignment_list.append((operation, last_machine))
                        
                        # Remove machine from path sequence and get preceding operation                        
                        temp_best_node = list(best_final_node)
                        temp_best_node[0] = operation
                        best_final_node = tuple(temp_best_node)
                        best_final_node[1].pop()
                    else:
                        # Finished
                        break
                
                # Assign machine and operation through the back propagated assignment list
                for element in reversed(assignment_list):
                    assign_machine_to_operation(element[0], machine_graph, element[1])

                break
    return 1



