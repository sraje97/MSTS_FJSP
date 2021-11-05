from copy import deepcopy
import random
from criticalpath import Node
import operator
import networkx as nx

import graph
import machine_assignment
from msts_algo_new import calculate_makespan
from operation_scheduling import get_start_time

############################################################

def flatten_job(jobs_array):
    sorted_jobs = []
    for job in jobs_array:
        sorted_op_nums = sorted(job, key=lambda op: op.op_num)
        sorted_jobs.append(sorted_op_nums)

    return sorted_jobs

# Get the finish time of an operation in the machine's schedule
def get_machine_finish_time(op_schedule):
	return max(op_schedule, key=operator.itemgetter(2))

def find_mach(op_num, schedules):
    for key, val in schedules.items():
        for i in range(len(val)):
            if val[i][0] == op_num:
                return key, i

def reset_times(schedules):
    for key, schedule in schedules.items():
        for i in range(len(schedule)):
            schedule[i] = (schedule[i][0], 0, 0)
    return schedules

def check_op_precedence(oper, left_op, op_df):
    if op_df.at[left_op, oper] == 1:
        return False
    return True

def get_operation_job(jobs_array, oper):
    oper_job = oper[oper.find("O") + 1 : oper.find("_")]
    job_array = jobs_array[ int(oper_job) - 1]
    operation = [item for item in job_array if item.op_num == oper][0]
    return operation, job_array
"""
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
"""
############################################################

def get_critical_path(jobs_array, op_schedule):
    dj_graph = Node('DJ Graph')

    for job in jobs_array:
        for op in job:
            print("Add Node:", op.op_num, op.processing_time)
            op_num = op.op_num
            op_num = dj_graph.add(Node(op.op_num, duration=op.processing_time))
        for op in job:
            if type(op.succ) is list:
                for succ in op.succ:
                    print("Link Node (list):", op.op_num, succ)
                    dj_graph.link(op.op_num, succ)
            elif op.succ == None:
                continue
            else:
                print("Link Node (single):", op.op_num, op.succ)
                dj_graph.link(op.op_num, op.succ)
            #print(op.op_num, op.succ, op.processing_time)

    for key, val in op_schedule.items():
        print(key, val)
        for i in range(len(val)-1):
            print("Link Node (op_schedule):", val[i][0], val[i+1][0])
            dj_graph.link(val[i][0], val[i+1][0])
    
    dj_graph.update_all()
    CP = dj_graph.get_critical_path()
    print(CP)
    print(dj_graph.duration)
    return CP, dj_graph.duration

def schedule_operation(job_array, operation, machine_graph, scheduled_operations):
    # TODO:
    # PREVIOUS OPERATION'S MACHINE MAY NOT BE CORRECT DUE TO SWAP
    # MIGHT HAVE TO FIND ANOTHER WAY TO GET PREVIOUS OP'S MACHINE
    # HAVE TO REASSIGN MACH_NUM IF OPERATION SWAPS TO ANOTHER MACHINE IN TABU_MOVE()

    if operation.pre == None:
        # If no previous operation then get machine's latest finishing time
        start_time = get_start_time(operation, None, machine_graph)
        prev_machine = ''
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
            # Previous operations were not scheduled, hence return
            return scheduled_operations, -1
    elif operation.pre in scheduled_operations:
        # If previous operation is scheduled, find the latest start time
        prev_operation = [item for item in job_array if item.op_num == operation.pre][0]
        start_time = get_start_time(operation, prev_operation, machine_graph)
        prev_machine = prev_operation.mach_num
    else:
        # Previous operation was not scheduled, hence return
        return scheduled_operations, -1
    
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
    #op_schedule.sort(key=lambda a: a[2])
    nx.set_node_attributes(machine_graph, {machine: {'op_schedule': op_schedule} } )
    
    # Update the operation's finish time
    operation.finish_time = finish_time

    scheduled_operations.append(operation.op_num)
    return scheduled_operations, 1
    ###########################

def recompute_times(jobs_array, machine_graph, schedules):
    scheduled_operations = []
    next_executable_operations = []

    for val in schedules.values():
        operation = val[0][0]
        next_executable_operations.append(operation)

    prev_exec_flag = False

    while next_executable_operations:
        if not prev_exec_flag:
            operation = next_executable_operations[0]
        else:
            operation = random.choice(next_executable_operations)
            prev_exec_flag = False

        operation, job_array = get_operation_job(jobs_array, operation)

        scheduled_operations, unscheduled = schedule_operation(job_array, operation, machine_graph, scheduled_operations)
        
        if unscheduled == -1:
            prev_exec_flag = True
            continue

        # Add the next executable op from the schedule
        mach, idx = find_mach(operation.op_num, schedules)
        oper_schedule = schedules[mach]
        if idx == (len(oper_schedule)-1):
            prev_exec_flag = True
            next_executable_operations.remove(operation.op_num)
            continue
        next_executable_operations.append(oper_schedule[idx+1][0])

        next_executable_operations.remove(operation.op_num)


    return jobs_array, machine_graph

def tabu_move(jobs_array, machine_graph, op_df, swap_method):
    solutions_list = []
    schedules = graph.get_op_schedule(machine_graph)

    for schedule in schedules.values():
        schedule.sort(key=lambda a: a[-1])

    if swap_method == "Critical Path":
        eligible_ops, mks = get_critical_path(jobs_array, schedules)
        for i in range(len(eligible_ops)):
            eligible_ops[i] = eligible_ops[i].name

    # TODO: Maybe get a random handful of eligible_ops instead of all?
    for oper in eligible_ops:
        # Create a deepcopy for each unique operation
        oper_schedules = deepcopy(schedules)
        oper_jobs_array = deepcopy(jobs_array)
        oper_machine_graph = deepcopy(machine_graph)

        # Find the machine and index of the operation in the machine schedule
        mach, idx = find_mach(oper, schedules)

        # Get the machine of the particular operation
        oper_schedule = oper_schedules[mach]

        # Can't swap to the left if operation is in first position
        if idx == 0:
            continue

        # Get the neighbouring operation (from the left)
        left_op = oper_schedule[idx-1][0]
        left_job = left_op[left_op.find("O") + 1 : left_op.find("_")]
        oper_job = oper[oper.find("O") + 1 : oper.find("_")]

        # If both operations belong to same job, ensure precedence constraints
        # are not being violated
        if left_job == oper_job:
            swap_isvalid = check_op_precedence(oper, left_op, op_df[int(oper_job) - 1])
            if not swap_isvalid:
                continue
            oper_schedule[idx-1] = (oper, 0, 0)
            oper_schedule[idx] = (left_op, 0, 0)
        else:
            oper_schedule[idx-1] = (oper, 0, 0)
            oper_schedule[idx] = (left_op, 0, 0)

        """ # WILL ONLY NEED THIS FOR OPERATION SWAP TO ANOTHER MACHINE
        # Change machine number for oper and left_job and update processing time
        oper_job_array = oper_jobs_array[ int(oper_job) - 1]
        op = [item for item in oper_job_array if item.op_num == oper][0]
        op.mach_num = mach
        for ele in op.machines:
            if ele[0] == mach:
                machining_time = machine_assignment.calculate_machining_time(oper_machine_graph, ele)
        op.processing_time = machining_time

        oper_job_array = oper_jobs_array[ int(left_job) - 1]
        op = [item for item in oper_job_array if item.op_num == left_op][0]
        op.mach_num = mach
        for ele in op.machines:
            if ele[0] == mach:
                machining_time = machine_assignment.calculate_machining_time(oper_machine_graph, ele)
        op.processing_time = machining_time
        """
        
        # TODO: Recompute Times
        oper_schedules = reset_times(oper_schedules)

        for machine, sched in oper_schedules.items():
            nx.set_node_attributes(oper_machine_graph, {machine: {'op_schedule': sched} } )
        
        oper_jobs_array, oper_machine_graph = recompute_times(oper_jobs_array, oper_machine_graph, oper_schedules)

        _, _, makespan = calculate_makespan(oper_machine_graph)

        solutions_list.append((oper_jobs_array, oper_machine_graph, oper, mach, makespan))

    # Sort neighbourhood by ascending makespan value
    solutions_list.sort(key=lambda a: a[-1])

    return solutions_list