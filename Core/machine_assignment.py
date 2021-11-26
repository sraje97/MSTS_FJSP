import random
import operator
import networkx as nx
from copy import deepcopy

bigM = 999_999

############################################################

# Get machining time for specific operation-machine combination [MT = (PT * alpha) + (ST * beta)]
def calculate_machining_time(machine_graph, machine):
    return round( machine[1] * machine_graph.nodes[machine[0]]['alpha'] )

# Get transition time from previous operation's machine to current operation's machine
def get_transition_time(machine_graph, machA, machB):
    return int(machine_graph.edges[machA, machB]['weight'])

def get_best_node(p_queue):
    return min(p_queue, key=operator.itemgetter(2))

def getIndexOfTuple(l, index, value):
    for pos,t in enumerate(l):
        if t[index] == value:
            return pos

def find_first_op(job_array):
    operation = job_array[-1]
    op_found_flag = False
    while not op_found_flag:
        op_pre = operation.pre
        if type(op_pre) is list:
            for op in op_pre:
                operation = [item for item in job_array if item.op_num == op][0]
                if operation.series[0] == 'S':
                    break
        elif op_pre == None:
            op_found_flag = True
            return operation
        else:
            operation = [item for item in job_array if item.op_num == op_pre][0]
    return -1
        



# Add operation to the machine's schedule and store machine in operation's 'mach_num' variable
def assign_operation_to_machine(operation, machine_graph, machine):
    # Add machine label to operation
    operation.mach_num = machine[0]
    operation.processing_time = machine[1]

    # Add operation to machine/node's attribute
    # Don't assign starting and finishing time yet
    op_schedule = machine_graph.nodes[machine[0]]['op_schedule']
    op_schedule.append( (operation.op_num, 0, 0) )

    nx.set_node_attributes(machine_graph, {machine[0]: {'op_schedule': op_schedule } } )

def get_min_FMT(operation, machine_graph, eligible_machines):
    minFMT = bigM

    for machine in eligible_machines:
        # Get machining time for specific operation-machine combination [MT = (PT * alpha) + (ST * beta)]
        #machining_time = round( operation.processing_time * machine_graph.nodes[machine[0]]['alpha'] )
        machining_time = calculate_machining_time(machine_graph, machine)

        if machining_time < minFMT:
            # Assign the machine with FMT for operation
            minFMT = machining_time
            best_machine = machine
    # Assign machine to operation
    assign_operation_to_machine(operation, machine_graph, best_machine)

# Create a node/tuple (operation, path sequence, total time)
def make_path_node(operation, machine_graph, machine, prev_machine, node):
    # Add machine to list of machines in path
    path_seq = node[1]
    path_seq.append(machine[0])

    # Calculate total machining and transition time and previous time
    total_time = calculate_machining_time(machine_graph, machine) + \
                get_transition_time(machine_graph, prev_machine, machine[0]) + node[2]

    # Create node/tuple
    path_node = (operation, path_seq, total_time)
    return path_node

# Assign all succeeding P2 parallel operations using Greedy algorithm
def assign_parallel_operation_succ(job_array, operation, machine_graph):
    # Get operation using the operation number
    #operation = [item for item in job_array if item.op_num == op_num][0]
    eligible_machines = operation.machines

    # Assign operation to machine using Greedy algorithm
    get_min_FMT(operation, machine_graph, eligible_machines)

    # Assign any succeeding P2 parallel operations as well
    while operation.succ != None:
        op_P2_succ = operation.succ
        operation = [item for item in job_array if item.op_num == op_P2_succ][0]

        # Only assign successor operations which are P2
        if operation.series[0] == "P":
            eligible_machines = operation.machines

            # Assign operation to machine using Greedy algorithm
            get_min_FMT(operation, machine_graph, eligible_machines)
        else:
            break

# Assign all preceding P2 parallel operations using Greedy algorithm
def assign_parallel_operation_pre(job_array, operation, machine_graph):
    # Get operation using the operation number
    #operation = [item for item in job_array if item.op_num == op_num][0]
    eligible_machines = operation.machines

    # Assign operation to machine using Greedy algorithm
    get_min_FMT(operation, machine_graph, eligible_machines)

    # Assign any preceding P2 parallel operations as well
    while operation.pre != None:
        op_P2_pre = operation.pre
        operation = [item for item in job_array if item.op_num == op_P2_pre][0]
        
        # Only assign successor operations which are P2
        if operation.series[0] == "P":
            eligible_machines = operation.machines

            # Assign operation to machine using Greedy algorithm
            get_min_FMT(operation, machine_graph, eligible_machines)
        else:
            break

############################################################
#               MACHINE ASSIGNMENT ALGORITHMS              #
############################################################

# Randomly assign machine to all operations
def assign_random(jobs_array, machine_graph):
    for job in jobs_array:
        for operation in job:
            # Get list of all eligible machines
            eligible_machines = operation.machines

            if len(eligible_machines) > 1:
                # Choose any random machine if more than one compatible machine
                machine = random.choice(eligible_machines)
            else:
                # Choose the only eligible machine
                machine = eligible_machines[0]
            # Assign the machine to the operation (and vice versa)
            assign_operation_to_machine(operation, machine_graph, machine)
    return 1

# Greedily assign machine to all operations using either 
# FMT - Fastest Machining Time (default) or LMT - Longest Machining Time
def assign_greedy(jobs_array, machine_graph, greedy_type="FMT"):
    for job in jobs_array:
        for operation in job:
            # Get list of all eligible machines
            eligible_machines = operation.machines

            if greedy_type == "FMT":
                # Use default FMT
                get_min_FMT(operation, machine_graph, eligible_machines)
            else:
                # Use LMT
                maxFMT = 0

                for machine in eligible_machines:
                    machining_time = calculate_machining_time(machine_graph, machine)

                    if machining_time > maxFMT:
                        # Assign the machine with LMT for operation
                        maxFMT = machining_time
                        best_machine = machine
                # Assign the machine to the operation (and vice versa)
                assign_operation_to_machine(operation, machine_graph, best_machine)
    return 1

# Assign the operation to the machine with the least number of operations already assigned
# i.e., the Least Utilised Machine (LUM)
def assign_LUM(jobs_array, machine_graph):
    for job in jobs_array:
        for operation in job:
            # Get list of all eligible machines
            eligible_machines = operation.machines

            # Get the least utilised machine
            least_length = bigM
            for machine in eligible_machines:
                len_op_schedule = len(machine_graph.nodes[machine[0]]['op_schedule'])
                if len_op_schedule < least_length:
                    least_length = len_op_schedule
                    LUM = machine

            # Assign the machine to the operation (and vice versa)
            assign_operation_to_machine(operation, machine_graph, LUM)
    return 1

# Use adapted Dijkstra's algorithm to assign machine with shortest path for each job
def run_shortest_path(jobs_array, machine_graph):
    total_jobs = len(jobs_array)
    jobs = []
    for i in range(total_jobs):
        jobs.append("J"+str(i+1))

    while jobs:
        # Select any arbitrary job to assign machines using Shortest Path
        random_job = random.choice(jobs)
        #print(random_job)
        # Remove job from list of jobs to be chosen
        jobs.remove(random_job)
        job = [item for item in jobs_array if item[0].job_num == random_job][0]

        # Create priority queue for node/tuple (operation, path sequence, total time)
        priority_queue = []
        # Keep list of processed parallel operations
        processed_parallel_ops = []
        # Use to store best final node
        best_final_node = (None, None, bigM)

        operation = job[0]
        if operation.series[0] != 'S':
            operation = find_first_op(job)
        eligible_machines = operation.machines

        if len(eligible_machines) > 3:
            eligible_machines.sort(key=lambda a: a[1])
            eligible_machines = eligible_machines[0:3]

        for machine in eligible_machines:
            # For each eligible machine add tuple of (operation, machince, total time) 
            # to the priority queue
            total_time = calculate_machining_time(machine_graph, machine)
            path_node = (operation, [machine[0]], total_time)
            priority_queue.append(path_node)
        
        while priority_queue:
            # Get node with the smallest total time in queue
            best_node = get_best_node(priority_queue)
            idx = getIndexOfTuple(priority_queue, 1, best_node[1])
            #idx = [index for (index, item) in enumerate(priority_queue) if item[1] == best_node[1]][-1]
            #print(idx)
            #idx = [i for i, tupl in enumerate(priority_queue) if tupl[1] == best_node[1]][-1]
            # Pop node from queue
            priority_queue.pop(idx)

            # Get next operation in sequence and the current operation's assigned machine
            op_succ = best_node[0].succ
            prev_machine = best_node[1][-1]

            if type(op_succ) is list:
                for op in op_succ:
                    oper = [item for item in job if item.op_num == op][0]
                    if oper.series[0] == 'P':
                        if oper.op_num not in processed_parallel_ops:
                            #print("Parallel successor", oper.op_num, oper.job_num)
                            processed_parallel_ops.append(oper.op_num)

                            # Assign the parallel operation to a machine
                            assign_parallel_operation_succ(job, oper, machine_graph)
                    if oper.series[0] == 'S':
                        seq_op = oper.op_num
                op_succ = seq_op

            operation = [item for item in job if item.op_num == op_succ][-1]
            #operation = [item for item in job if item.op_num in op_succ and item.series == "S"][-1]
            
            # Get previous operation and check if tuple
            op_pre = operation.pre

            if type(op_pre) is list:
                for op in op_pre:
                    oper = [item for item in job if item.op_num == op][0]
                    if oper.series[0] == 'P':
                        if oper.op_num not in processed_parallel_ops:
                            #print("Parallel successor", oper.op_num, oper.job_num)
                            processed_parallel_ops.append(oper.op_num)

                            # Assign the parallel operation to a machine
                            assign_parallel_operation_pre(job, oper, machine_graph)


            # Add operations (bar the final operations of each job) to the priority queue
            if operation.succ != None:
                eligible_machines = operation.machines

                if len(eligible_machines) > 3:
                    eligible_machines.sort(key=lambda a: a[1])
                    eligible_machines = eligible_machines[0:3]

                for machine in eligible_machines:
                    temp_node = deepcopy(best_node)
                    # Create a path node (tuple) for operation + machine/path sequence + total time
                    path_node = make_path_node(operation, machine_graph, machine, prev_machine, temp_node)

                    # Add path node (tuple) to the priority queue
                    priority_queue.append(path_node)
            else:
                #print("Final operation", operation.op_num, operation.job_num)
                eligible_machines = operation.machines
                for machine in eligible_machines:
                    temp_node = deepcopy(best_node)
                    # Create a path node (tuple) for operation + machine/path sequence + total time
                    path_node = make_path_node(operation, machine_graph, machine, prev_machine, temp_node)
                    
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
                    last_operation = best_final_node[0].pre

                    # If preceding parallel operations, append only the S operation
                    if type(last_operation) is list:
                        #last_operation = last_operation[0]
                        #operation = [item for item in job if item.op_num == last_operation][0]
                        for op in last_operation:
                            oper = [item for item in job if item.op_num == op][0]
                            if oper.series[0] == 'S':
                                operation = oper

                        #operation = [item for item in job if item.op_num in last_operation and item.series == "S"][0]
                        assignment_list.append( (operation, last_machine) )

                        # Remove machine from path sequence and get preceding operation
                        temp_best_node = list(best_final_node)
                        temp_best_node[0] = operation
                        best_final_node = tuple(temp_best_node)
                        best_final_node[1].pop()
                    elif last_operation != None:
                        operation = [item for item in job if item.op_num == last_operation][0]
                        assignment_list.append( (operation, last_machine) )
                        
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
                    machine = [item for item in element[0].machines if item[0] == element[1]][0]
                    assign_operation_to_machine(element[0], machine_graph, machine)
                break
    return 1