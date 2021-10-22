############################################################

## IMPORT PYTHON PACKAGES
from posixpath import splitext
import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import copy

# Sets base directory one level higher than current file (@ X:\\..\\MSTS_FJSP)
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')

# Add base directory and the data directory to the system path
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'data'))

## IMPORT OUR MODULES
import import_data
from operation import Operation

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

def initialise_operations(datafile):
    # Initialise as empty array/list
    jobs_array = []

    if os.path.splitext(datafile)[1] == '.txt':

        operations, machines =  import_data.import_txt_data(datafile)

        ## IMPORT JOBS ##
        print(operations)

        generated_ops = []
        temp_jobs = []
        for ops in operations:

            # Check if operation still part of this job
            if ops[0] not in generated_ops:
                op = Operation(ops[2], ops[0], succ_op=ops[1])
                temp_jobs.append(op)
                generated_ops.append(ops[0])
            else:
                op = [item for item in temp_jobs if item.op_num == ops[0]][0]
                if op == None:
                    raise Exception("Operation with this operation label not found!")
                succ_op = op.succ

                if type(succ_op) == list:
                    succ_op.append(ops[1])
                    op.succ = succ_op
                elif type(succ_op) == str:
                    succ_ops = [succ_op, ops[1]]
                    op.succ = succ_ops
                else:
                    op.succ = ops[1]

            if ops[1] not in generated_ops:
                op = Operation(ops[2], ops[1], pre_op=ops[0])
                temp_jobs.append(op)
                generated_ops.append(ops[1])
            else:
                op = [item for item in temp_jobs if item.op_num == ops[1]][0]
                if op == None:
                    raise Exception("Operation with this operation label not found!")
                pre_op = op.pre
                if type(pre_op) == list:
                    pre_op.append(ops[0])
                    op.pre = pre_op
                elif type(pre_op) == str:
                    pre_ops = [pre_op, ops[0]]
                    op.pre = pre_ops
                else:
                    op.pre = ops[0]

        ## IMPORT MACHINES ##
        for i in range(len(machines)):
            eligible_machines = []
            j = 1
            while j < len(machines[i]):
                machine = 'M' + machines[i][j]
                process_time = machines[i][j+1]
                eligible_machines.append((machine, process_time))
                j += 2
            op = [item for item in temp_jobs if item.op_num == str(i)][0]
            if op == None:
                raise Exception("Operation with this operation label not found!")
            op.machines = eligible_machines
        
        for op in temp_jobs:
            print(op.op_num, op.job_num, op.pre, op.succ, op.machines)


        max_jobs = int(op.job_num[1:])
        for i in range(max_jobs):
            jobs_array.append([])

        job_number = 'J1'
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

    return jobs_array



def msts():
    ############################################################
    #               INITIALIALISE JOBS + MACHINES              #
    ############################################################

    # TODO: Get as inputs
    instances_file = base_dir + 'data\Benchmarks\DAFJS\DAFJS01.txt'
    num_jobs = 3
    num_diff_machines = 4
    MA_algo_choice = ""
    OS_algo_choice = ""
    epochs = 100
    C_eni_max = 100
    p_exp_con = 1.0
    swap_methods = ["swap machine", "swap operation", "SPO", "MSPO"]

    eps_start = 1.0
    eps_end = 0.0
    eps_decay = 0.5

    jobs_array = initialise_operations(instances_file)

    if "DAFJS" in instances_file or "YFJS" in instances_file:
        for jobarray in jobs_array:
            final_ops = []
            for op in jobarray:
                if op.succ == None:
                    final_ops.append(op)

            branchlist = []
            if len(final_ops) > 1:
                print("Many end operations")
                branchlist = recurse_forward(jobarray, jobarray[0].op_num, branchlist)
            else:
                print("Single end operation")
                branchlist = recurse_reverse(jobarray, final_ops[0].op_num, branchlist)

            print(branchlist)

            sequences = []
            templist = []
            i = 0

            while i < len(branchlist):
                oper = branchlist[i]
                if type(oper) is tuple:
                    if oper[1] == "Start":
                        templist.append(oper[0])
                        branched_copy = copy.deepcopy(templist)
                        idx = i
                    if oper[1] == "Stop":
                        templist.append(oper[0])
                        sequences.append(templist)
                        templist = copy.deepcopy(branched_copy)
                else:
                    templist.append(oper)

                i += 1

            if len(final_ops) == 1:
                for item in sequences:
                    item = item.reverse()

            print(sequences)

            longestseq = max(sequences, key=len)
            print(longestseq)
            branch_label = 'S'

            for sequence in sequences:
                if sequence == longestseq:
                    for op in longestseq:
                        oper = [item for item in jobarray if item.op_num == op][0]
                        oper.series = branch_label
                    branch_label = 'P1'
                else:
                    for op in sequence:
                        if op not in longestseq:
                            oper = [item for item in jobarray if item.op_num == op][0]
                            oper.series = branch_label
                    branch_label = 'P' + str(int(branch_label[1]) + 1)

            for op in jobarray:
                print(op.op_num, op.series)


    # TODO: Add branch name to operations (i.e. S, P1, P2)
    # TODO: Setup Machine graph and nodes/edges attributes
        

### BEGIN MAIN PROGRAM ###
if __name__ == '__main__':
    msts()