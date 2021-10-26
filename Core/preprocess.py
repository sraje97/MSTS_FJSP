import os
from copy import deepcopy

from operation import Operation
import import_data

############################################################

def recurse_reverse(jobarray, pre_op, branch):
    op = [item for item in jobarray if item.op_num == pre_op][0]

    if op.pre == None:
        branch.append((op.op_num, "Stop"))
        return branch

    if type(op.pre) is list:
        branch.append((op.op_num, "Start"))
        for pre_op in op.pre:
            recurse_reverse(jobarray, pre_op, branch)
    else:
        branch.append(op.op_num)
        recurse_reverse(jobarray, op.pre, branch)
    
    
    return branch

def recurse_forward(jobarray, succ_op, branch):
    op = [item for item in jobarray if item.op_num == succ_op][0]

    if op.succ == None:
        branch.append((op.op_num, "Stop"))
        return branch

    if type(op.succ) is list:
        branch.append((op.op_num, "Start"))
        for succ_op in op.succ:
            recurse_forward(jobarray, succ_op, branch)
    else:
        branch.append(op.op_num)
        recurse_forward(jobarray, op.succ, branch)
    
    return branch

# Get operations from txt file, generate jobs list and initialise operation information
def initialise_operations(datafile):
    # Hold list of jobs
    jobs_array = []

    # Ensure file is '.txt'
    if os.path.splitext(datafile)[1] == '.txt':

        # Import precedence and machining information of each job
        operations, machines =  import_data.import_txt_data(datafile)

        ## JOBS ##
        #print(operations)

        # Initialise lists to hold seen operations and hold a job
        generated_ops = []
        temp_jobs = []

        # Change operations labels to standard labelling method
        operations = change_ops_labels(operations)

        for ops in operations:

            # Create new Operation object instance if operation not seen before
            # (For first operation in ops)
            if ops[0] not in generated_ops:
                op = Operation(ops[2], ops[0], succ_op=ops[1])
                temp_jobs.append(op)
                generated_ops.append(ops[0])
            # Else, update previously created Operation object's information
            else:
                # Find Operation object in jobs list
                op = [item for item in temp_jobs if item.op_num == ops[0]][0]
                if op == None:
                    raise Exception("Operation with this operation label not found!")
                
                # Get Operation's succeeding operation
                succ_op = op.succ

                # Add the current succeeding operation to its list
                # (Multiple, one, or None succeeding operations)
                if type(succ_op) == list:
                    succ_op.append(ops[1])
                    op.succ = succ_op
                elif type(succ_op) == str:
                    succ_ops = [succ_op, ops[1]]
                    op.succ = succ_ops
                else:
                    op.succ = ops[1]

            # Create new Operation object instance if operation not seen before
            # (For second operation in ops)
            if ops[1] not in generated_ops:
                op = Operation(ops[2], ops[1], pre_op=ops[0])
                temp_jobs.append(op)
                generated_ops.append(ops[1])
            # Else, update previously created Operation object's information
            else:
                # Find Operation object in jobs list
                op = [item for item in temp_jobs if item.op_num == ops[1]][0]
                if op == None:
                    raise Exception("Operation with this operation label not found!")
                
                # Get Operation's preceding operation
                pre_op = op.pre

                # Add the current preceding operation to its list
                # (Multiple, one, or None preceding operations)
                if type(pre_op) == list:
                    pre_op.append(ops[0])
                    op.pre = pre_op
                elif type(pre_op) == str:
                    pre_ops = [pre_op, ops[0]]
                    op.pre = pre_ops
                else:
                    op.pre = ops[0]

        ## MACHINES ##
        # Store unique machines
        unique_machines = []
        for i in range(len(machines)):
            # Initialise a list of operation's eligible machines
            eligible_machines = []
            j = 1

            # For each eligible machine for operation save label and processing time
            while j < len(machines[i]):
                machine = 'M' + machines[i][j]
                process_time = machines[i][j+1]
                eligible_machines.append((machine, process_time))

                # Store the different number of machines
                if machine not in unique_machines:
                    unique_machines.append(machine)
                j += 2
            
            # Get the operation
            #op = [item for item in temp_jobs if item.op_num == str(i)][0]
            op = temp_jobs[i]
            if op == None:
                raise Exception("Operation with this operation label not found!")
            
            # Add the eligible machines to the operation
            op.machines = eligible_machines
        
        #for op in temp_jobs:
        #    print(op.op_num, op.job_num, op.pre, op.succ, op.machines)

        # Get the number of jobs n, and create n nested lists in jobs_array
        max_jobs = int(op.job_num[1:])
        for i in range(max_jobs):
            jobs_array.append([])

        job_number = 'J1'
        # For each operation in temp_jobs add the operation to the 
        # corresponding nested jobs list in jobs_array
        for operation in temp_jobs:
            if operation.job_num == job_number:
                job_idx = int(job_number[1:]) - 1
                jobs_array[job_idx].append(operation)
            else:
                job_number = operation.job_num
                job_idx = int(job_number[1:]) - 1
                jobs_array[job_idx].append(operation)

        #print(jobs_array)
        #print(machines)
    else:
        raise Exception("Invalid file type")

    return jobs_array, unique_machines

def label_parallel_branches(jobs_array):
    for job in jobs_array:
        final_ops = []
        for op in job:
            if op.succ == None:
                final_ops.append(op)

        branchlist = []
        if len(final_ops) > 1:
            # Many end operations
            branchlist = recurse_forward(job, job[0].op_num, branchlist)
        else:
            #  Single end operation
            branchlist = recurse_reverse(job, final_ops[0].op_num, branchlist)

        #print(branchlist)

        sequences = []
        templist = []
        i = 0

        while i < len(branchlist):
            oper = branchlist[i]
            if type(oper) is tuple:
                if oper[1] == "Start":
                    templist.append(oper[0])
                    branched_copy = deepcopy(templist)
                    idx = i
                if oper[1] == "Stop":
                    templist.append(oper[0])
                    sequences.append(templist)
                    templist = deepcopy(branched_copy)
            else:
                templist.append(oper)

            i += 1

        if len(final_ops) == 1:
            for item in sequences:
                item = item.reverse()

        #print(sequences)

        longestseq = max(sequences, key=len)
        #print(longestseq)
        branch_label = 'S'

        for sequence in sequences:
            if sequence == longestseq:
                for op in longestseq:
                    oper = [item for item in job if item.op_num == op][0]
                    oper.series = branch_label
                branch_label = 'P1'
            else:
                for op in sequence:
                    if op not in longestseq:
                        oper = [item for item in job if item.op_num == op][0]
                        oper.series = branch_label
                branch_label = 'P' + str(int(branch_label[1]) + 1)

        #for op in job:
        #    print(op.op_num, op.series)

    return jobs_array

# Change the labels of operations from numbers to standard labelling method i.e. "O1_10"
def change_ops_labels(operations):
    label = ['O', 'J', 0, 1]
    base = 1

    for ops in operations:
        if ops[2] == "".join( [ str(label[1]), str(label[2]) ] ):
            ops[0] = label[0] + str(label[2]) + "_" + str( int(ops[0]) + base_diff )
            ops[1] = label[0] + str(label[2]) + "_" + str( int(ops[1]) + base_diff )
        else:
            base_diff = base - int(ops[0])
            label[2] += 1
            ops[0] = label[0] + str(label[2]) + "_" + str( int(ops[0]) + base_diff )
            ops[1] = label[0] + str(label[2]) + "_" + str( int(ops[1]) + base_diff )

    return operations