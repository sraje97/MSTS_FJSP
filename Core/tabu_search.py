import pickle
import random
import timeit
import operator
import networkx as nx
from copy import deepcopy
#from criticalpath import Node
from critpath import Node


import graph
import machine_assignment
#from msts_algo_new import calculate_makespan
from operation_scheduling import get_start_time

############################################################

def mydeepcopy(obj):
    try:
        #print("pickling")
        return pickle.loads(pickle.dumps(obj, -1))
    except pickle.PicklingError:
        #print("deepcopy")
        return deepcopy(obj)

def flatten_job(jobs_array):
    sorted_jobs = []
    for job in jobs_array:
        sorted_op_nums = sorted(job, key=lambda op: op.op_num)
        for op in sorted_op_nums:
            sorted_jobs.append(op)

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
    for _, schedule in schedules.items():
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

def get_insertion_pos(job, operation, op_df, op_schedule, mach):
    valid_pos_list = []
    same_jobs = []

    # If the machine schedule is empty (i.e. no assigned operations)
    if not op_schedule:
        valid_pos_list.append( (0, mach) )
        return valid_pos_list

    # Get list of all operations in schedule which are from the same job
    for i in range(len(op_schedule)):
        op =  op_schedule[i][0]
        op_job = op[op.find("O") + 1 : op.find("_")]
        if op_job == operation.job_num[1:]:
            same_jobs.append((op, i))
    
    # If no operations from the same job, then curr operation can be 
    # inserted at any position
    if not same_jobs:
        for pos in range(len(op_schedule)+1):
            valid_pos_list.append( (pos, mach) )
        return valid_pos_list
    
    max_pos = len(op_schedule)
    min_pos = -1
    op_job = operation.job_num[1:]
    for op, pos in same_jobs:
        if not check_op_precedence(op, operation.op_num, op_df[ int(op_job)-1 ]):
            max_pos = pos
        if not check_op_precedence(operation.op_num, op, op_df[ int(op_job)-1 ]):
            min_pos = pos
    
    for i in range(min_pos, max_pos):
        valid_pos_list.append( (i+1, mach) )

    return valid_pos_list

############################################################

def get_critical_path(jobs_array, op_schedule):
    dj_graph = Node('DJ Graph')

    for job in jobs_array:
        for op in job:
            #print("Add Node:", op.op_num, op.processing_time)
            op_num = op.op_num
            op_num = dj_graph.add(Node(op.op_num, duration=op.processing_time))
        for op in job:
            if type(op.succ) is list:
                for succ in op.succ:
                    #print("Link Node (list):", op.op_num, succ)
                    dj_graph.link(op.op_num, succ)
            elif op.succ == None:
                continue
            else:
                #print("Link Node (single):", op.op_num, op.succ)
                dj_graph.link(op.op_num, op.succ)
            #print(op.op_num, op.succ, op.processing_time)

    for _, val in op_schedule.items():
        #print(key, val)
        for i in range(len(val)-1):
            #print("Link Node (op_schedule):", val[i][0], val[i+1][0])
            dj_graph.link(val[i][0], val[i+1][0])
    
    #print("Updating DJ Graph", timeit.default_timer())
    dj_graph.update_all()
    #print("Updated DJ Graph", timeit.default_timer())
    CP = dj_graph.get_critical_path()
    #print(CP)
    #print(dj_graph.duration)
    return CP, dj_graph.duration

def schedule_operation(job_array, operation, machine_graph, scheduled_operations):
    if operation.pre == None:
        # If no previous operation then get machine's latest finishing time
        prev_machine = ''
        start_time = get_start_time(operation, None, machine_graph)
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
                    finish_time = oper.finish_time
            prev_machine = prev_operation.mach_num
            start_time = get_start_time(operation, prev_operation, machine_graph)
        else:
            # Previous operations were not scheduled, hence return
            return scheduled_operations, -1
    elif operation.pre in scheduled_operations:
        # If previous operation is scheduled, find the latest start time
        prev_operation = [item for item in job_array if item.op_num == operation.pre][0]
        prev_machine = prev_operation.mach_num
        start_time = get_start_time(operation, prev_operation, machine_graph)
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
    # TODO: Can get rid of input argument schedules, and just get it inside the function
    scheduled_operations = []
    next_executable_operations = []

    for val in schedules.values():
        # If no operations assigned to this machine schedule
        if len(val) == 0:
            continue
        operation = val[0][0]
        next_executable_operations.append(operation)
        #print(val)

    prev_exec_flag = False
    deadlock_cnt = 0

    while next_executable_operations:
        if deadlock_cnt == 100:
            #print("Returning")
            return -1, -1
        if not prev_exec_flag:
            operation = next_executable_operations[0]
        else:
            operation = random.choice(next_executable_operations)
            prev_exec_flag = False
        #print(operation, next_executable_operations)

        operation, job_array = get_operation_job(jobs_array, operation)

        scheduled_operations, unscheduled = schedule_operation(job_array, operation, machine_graph, scheduled_operations)
        
        if unscheduled == -1:
            #print(operation.op_num, next_executable_operations)
            deadlock_cnt += 1
            prev_exec_flag = True
            continue

        # Add the next executable op from the schedule
        mach, idx = find_mach(operation.op_num, schedules)
        oper_schedule = schedules[mach]
        if idx == (len(oper_schedule)-1):
            #prev_exec_flag = True
            next_executable_operations.remove(operation.op_num)
            continue
        next_executable_operations.append(oper_schedule[idx+1][0])

        next_executable_operations.remove(operation.op_num)

        deadlock_cnt = 0


    return jobs_array, machine_graph

def tabu_move(jobs_array, machine_graph, op_df, swap_method):
    random.seed(1)
    best_solutions = []
    solutions_list = []
    #schedules = graph.get_op_schedule(machine_graph)
    schedules = nx.get_node_attributes(machine_graph, 'op_schedule')

    for schedule in schedules.values():
        schedule.sort(key=lambda a: a[-1])
        #print(schedule)

    if swap_method[0:13] == "Critical Path":
        eligible_ops, _ = get_critical_path(jobs_array, schedules)
        for i in range(len(eligible_ops)):
            eligible_ops[i] = eligible_ops[i].name

    TS_start_time = timeit.default_timer()
    #print("Start time Tabu move:", TS_start_time)

    # TODO: Maybe get a random handful of eligible_ops instead of all?
    for oper in eligible_ops:
        # If running for too long, return what you've got so far
        if (timeit.default_timer() - TS_start_time) > 600: #TODO: Change to 3600
            #print("Time overboard")
            best_solutions.sort(key=lambda a: a[-1])
            return best_solutions

            # solutions_list.sort(key=lambda a: a[-1])
            #return solutions_list
        
        if swap_method[-2:] == "OS":

            # Create a mydeepcopy for each unique operation
            oper_schedules = mydeepcopy(schedules)
            oper_jobs_array = mydeepcopy(jobs_array)
            oper_machine_graph = mydeepcopy(machine_graph)

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
            
            # Set all times to 0
            oper_schedules = reset_times(oper_schedules)

            for machine, sched in oper_schedules.items():
                nx.set_node_attributes(oper_machine_graph, {machine: {'op_schedule': sched} } )
            
            oper_jobs_array, oper_machine_graph = recompute_times(oper_jobs_array, oper_machine_graph, oper_schedules)

            # Deadlock occured hence skip this oper swap and continue to next one
            if oper_jobs_array == -1:
                continue

            _, _, makespan = calculate_makespan(oper_machine_graph)

            solutions_list.append((oper_jobs_array, oper_machine_graph, oper, mach, makespan))
        
        elif swap_method[-2:] == "MA":
            operation, job = get_operation_job(jobs_array, oper)
            eligible_machines = operation.machines
            insertion_positions = []

            for machine in eligible_machines:
                if machine[0] == operation.mach_num:
                    continue
                op_schedule = machine_graph.nodes[machine[0]]['op_schedule']

                # Get all valid positions of operation insertion into machine's schedule
                valid_poses = get_insertion_pos(job, operation, op_df, op_schedule, machine[0])
                insertion_positions += valid_poses
            
            # If no alternate eligible machines or valid positions, then skip operation
            if not insertion_positions:
                continue

            # If insertion positions is too large, get a maximum of 50 positions to swap
            if len(insertion_positions) > 50:
                insertion_positions = random.sample(insertion_positions, 50)
            
            for pos, mach in insertion_positions:
                oper_jobs_array = mydeepcopy(jobs_array)
                oper_machine_graph = mydeepcopy(machine_graph)
                oper_schedules = nx.get_node_attributes(oper_machine_graph, 'op_schedule')
                operation, job = get_operation_job(oper_jobs_array, oper)

                # Delete the operation from its old position
                old_mach, idx = find_mach(oper, schedules)
                old_oper_schedule = oper_schedules[old_mach]
                del old_oper_schedule[idx]

                oper_schedule = oper_schedules[mach]
                #op_schedule = oper_machine_graph.nodes[mach]['op_schedule']        # REPLACE WITH THIS IF WE REMOVE SCHEDULES
                oper_schedule.insert( pos, (oper, 0, 0) )

                # Change operation's assigned machine and update processing time for new machine
                operation.mach_num = mach
                for ele in operation.machines:
                    if ele[0] == mach:
                        machining_time = machine_assignment.calculate_machining_time(oper_machine_graph, ele)
                        break
                operation.processing_time = machining_time

                oper_schedules = reset_times(oper_schedules)

                for machine, sched in oper_schedules.items():
                    nx.set_node_attributes(oper_machine_graph, {machine: {'op_schedule': sched} } )

                oper_jobs_array, oper_machine_graph = recompute_times(oper_jobs_array, oper_machine_graph, oper_schedules)

                # Deadlock occured hence skip this oper swap and continue to next one
                if oper_jobs_array == -1:
                    continue

                _, _, makespan = calculate_makespan(oper_machine_graph)

                solutions_list.append((oper_jobs_array, oper_machine_graph, oper, mach, makespan))

        else:
            print("Incorrect swap method")

        solutions_list.sort(key=lambda a: a[-1])
        if len(solutions_list) >= 10:
            best_solutions += solutions_list[:10]
        else:
            best_solutions += solutions_list
        
        # Empty solutions list
        solutions_list = []

    # Sort neighbourhood by ascending makespan value
    best_solutions.sort(key=lambda a: a[-1])
    return best_solutions
    
    #solutions_list.sort(key=lambda a: a[-1])

    #return solutions_list
