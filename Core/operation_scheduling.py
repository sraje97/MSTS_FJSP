import operator
from os import stat_result
import networkx as nx
from Core.machine_assignment import calculate_machining_time

import machine_assignment
import graph

############################################################

# TODO:
## CAN REMOVE ALL THREE AND REPLACE WITH ONE SINCE THEY ARE ALL THE SAME
## GET_SMT, GET_LRMT, GET_ERT
# Get operation with shortest machining time (SMT)
def get_SMT(next_operations):
	return min(next_operations, key=operator.itemgetter(1))

# Get operation with largest remaining machining time (LRMT)
def get_LRMT(next_operations):
    return max(next_operations, key=operator.itemgetter(1))

# Get operation with the earliest release time (ERT)
def get_ERT(next_operations):
    return min(next_operations, key=operator.itemgetter(1))

# Get the finish time of an operation in the machine's schedule
def get_machine_finish_time(op_schedule):
	return max(op_schedule, key=operator.itemgetter(2))

# Check if operation is in the executable operations list
def get_flag(operation, executable_operations):
    for item in executable_operations:
        if operation == item[0]:
            return True
    return False

# Get the the earliest start time for an operation
def get_start_time(operation, prev_operation, machine_graph):
	machine = operation.mach_num
	
	# Get machine's schedule
	op_schedule = machine_graph.nodes[machine]['op_schedule']
	machine_finish_time = get_machine_finish_time(op_schedule)
	
	# Find the latest time an operation may start
	if prev_operation == None:
		return machine_finish_time[-1]
	else:
		return max(prev_operation.finish_time, machine_finish_time[-1])

def get_total_remaining_time(job_array, op=None):
    total_time = 0

    if op == None:
        # Sum up all operation's processing time in a job
        for op in job_array:
            total_time += op.processing_time
    else:
        # If operation is specified, sum up total machining time from operation
        while op.succ != None:
            total_time += op.processing_time
            if type(op.succ) is tuple:
                for oper in op.succ:
                    op = [item for item in job_array if item.op_num == oper][0]
                    if op.series[0] == 'S':
                        continue
            else:
                op = [item for item in job_array if item.op_num == op.succ][0]
        total_time += op.processing_time
    return total_time


# Helper function to append operation, operation's machine, and specified time to executable operations list
def append_operation_tuple(job_array, operation, machine_graph, executable_operations, OS_algo):
    # Get processing time for current operation
    machining_time = operation.processing_time

    # Depending on the scheduling algorithm, add either machining/remaining/release time
    if OS_algo == "SMT":
        executable_operations.append( (operation, machining_time) )
    elif OS_algo == "LRMT":
        remaining_time = get_total_remaining_time(job_array, operation)
        remaining_time -= machining_time
        executable_operations.append( (operation, remaining_time) )
    else:
        #release_time = operation.finish_time
        #executable_operations.append( (operation, release_time) )
        executable_operations.append( (operation, operation.finish_time) )
    return executable_operations


def schedule_operation(job_array, operation, machine_graph, scheduled_operations):
    # Get the start time depending on previous operation or current machine's finishing time
    if operation.pre == None:
        start_time = get_start_time(operation, None, machine_graph)
        prev_machine = ''
    elif operation.pre in scheduled_operations:
        prev_operation = [item for item in job_array if item.op_num == operation.pre][0]
        start_time = get_start_time(operation, prev_operation, machine_graph)
        prev_machine = prev_operation.mach_num
    elif type(operation.pre) is list:
        # If previous operations are parallel branches, must wait till both have been scheduled
        branch_cnt = 0
        for op in operation.pre:
            if op in scheduled_operations:
                branch_cnt += 1

        if branch_cnt == len(operation.pre):
            finish_time = 0
            for op in operation.pre:
                oper = [item for item in job_array if item.op_num == op][0]
                # Get the operation which finished the latest
                if oper.finish_time > finish_time:
                    prev_operation = oper
            
            start_time = get_start_time(operation, prev_operation, machine_graph)
            prev_machine = prev_operation.mach_num
        else:
            pass
    else:
        print("Previous operation not found")

    # Get current operation's machine schedule
    machine = operation.mach_num
    op_schedule = machine_graph.nodes[machine]['op_schedule']
    idx = [i for i, tupl in enumerate(op_schedule) if tupl[0] == operation.op_num][-1]

    # Calculate the transition time between previous and current machine
    if prev_machine == '':
        transition_time = 0
    elif prev_machine == machine:
        transition_time = 0
    else:
        transition_time = machine_assignment.get_transition_time(machine_graph, machine, prev_machine)
    
    # Add transition time to the start time
    start_time += transition_time
    # Finish time is the starting time plus the processing time (ST + PT)
    finish_time = operation.processing_time + start_time

    # Update the tuple in the machine's schedule
    op_tuple = (operation.op_num, start_time, finish_time)
    op_schedule[idx] = op_tuple
    nx.set_node_attributes(machine_graph, {machine: {'op_schedule': op_schedule} } )
    
    # Update the operation's finish time
    operation.finish_time = finish_time

    scheduled_operations.append(operation.op_num)
    return scheduled_operations

def add_next_executable_operation(job, operation, machine_graph, executable_operations, scheduled_operations, OS_algo):
    # Get the index of the operation from the executable's list and remove it
    idx = [i for i, tupl in enumerate(executable_operations) if tupl[0].op_num == operation.op_num][0]
    remaining_time = executable_operations[idx][1]
    executable_operations.pop(idx)

    # Get the next operation in the job array
    op_succ = operation.succ

    if op_succ == None:
        return executable_operations
    elif type(op_succ) is list:
        # If multiple successive operations, add both operations to the list
        for op in op_succ:
            succ_operation = [item for item in job if item.op_num == op][0]
            executable_operations = append_operation_tuple(job, succ_operation, machine_graph, executable_operations, OS_algo)
    else:
        succ_operation = [item for item in job if item.op_num == op_succ][0]

        # In case of parallel previous operations, check if all have been scheduled
        if type(succ_operation.pre) is list:
            for op in succ_operation.pre:
                if op not in scheduled_operations:
                    # Get the previous operation which hasn't been scheduled
                    operation = [item for item in job if item.op_num == op][0]

                    # Check if this operation is not available for execution
                    if not(get_flag(operation, executable_operations)):

                        # Keep following the operation path back until the first scheduled operation is found
                        while operation.pre != None:
                            pre_operation = [item for item in job if item.op_num == operation.pre][0]
                            # If a previous operation has been scheduled, add this operation to executable's list
                            if operation.pre in scheduled_operations:
                                executable_operations = append_operation_tuple(job, operation, machine_graph, executable_operations, OS_algo)
                                return executable_operations
                            elif get_flag(pre_operation, executable_operations):
                                return executable_operations
                            else:
                                operation = pre_operation
                        
                        # Add the last found operation to executable operations list
                        executable_operations = append_operation_tuple(job, operation, machine_graph, executable_operations, OS_algo)
                        return executable_operations
        executable_operations = append_operation_tuple(job, succ_operation, machine_graph, executable_operations, OS_algo)
    
    return executable_operations



############################################################
#             OPERATION SCHEDULING ALGORITHMS          	   #
############################################################

def schedule_SMT(jobs_array, machine_graph):
    # Initialise lists to store already scheduled and next available operations
    scheduled_operations = []
    next_executable_operations = []

    # For each starting operation (of a job), enable it to be available for scheduling
    for i in range(len(jobs_array)):
        operation = jobs_array[i][0]
        machining_time = operation.processing_time
        next_executable_operations.append( (operation, machining_time) )
    
    while next_executable_operations:
        # Get the operation with shortest machining time
        operation = get_SMT(next_executable_operations)[0]
        job_array = jobs_array[ int(operation.job_num[1:]) - 1 ]

        # Schedule that operation onto it's assigned machine
        scheduled_operations = schedule_operation(job_array, operation, machine_graph, scheduled_operations)

        # Add operation's successors to the executable operations list
        next_executable_operations = add_next_executable_operation(job_array, operation, \
                    machine_graph, next_executable_operations, scheduled_operations, "SMT")

    return 1


# Schedule operations using the Largest Remaining Machining Time (LRMT) algorithm
def schedule_LRMT(jobs_array, machine_graph):
    # Initialise lists to store already scheduled and next available operations
    scheduled_operations = []
    next_executable_operations = []

    # For each starting operation (of a job), enable it to be available for scheduling
    for i in range(len(jobs_array)):
        operation = jobs_array[i][0]
        total_remaining_time = get_total_remaining_time(jobs_array[i])
        next_executable_operations.append( (operation, total_remaining_time) )
    
    while next_executable_operations:
        # Get the operation with shortest machining time
        operation = get_LRMT(next_executable_operations)[0]
        job_array = jobs_array[ int(operation.job_num[1:]) - 1 ]

        # Schedule that operation onto it's assigned machine
        scheduled_operations = schedule_operation(job_array, operation, machine_graph, scheduled_operations)

        # Add operation's successors to the executable operations list
        next_executable_operations = add_next_executable_operation(job_array, operation, \
                    machine_graph, next_executable_operations, scheduled_operations, "LRMT")

    return 1


# Schedule operations using the Earliest Release Time (ERT) algorithm
def schedule_ERT(jobs_array, machine_graph):
    # Initialise lists to store already scheduled and next available operations
    scheduled_operations = []
    next_executable_operations = []

    # For each starting operation (of a job), enable it to be available for scheduling
    for i in range(len(jobs_array)):
        operation = jobs_array[i][0]
        next_executable_operations.append( (operation, operation.finish_time) )
    
    while next_executable_operations:
        # Get the operation with shortest machining time
        operation = get_ERT(next_executable_operations)[0]
        job_array = jobs_array[ int(operation.job_num[1:]) - 1 ]

        # Schedule that operation onto it's assigned machine
        scheduled_operations = schedule_operation(job_array, operation, machine_graph, scheduled_operations)

        # Add operation's successors to the executable operations list
        next_executable_operations = add_next_executable_operation(job_array, operation, \
                    machine_graph, next_executable_operations, scheduled_operations, "ERT")

    return 1
