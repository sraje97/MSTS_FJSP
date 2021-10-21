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

def msts():
    ############################################################
    #               INITIALIALISE JOBS + MACHINES              #
    ############################################################

    # TODO: Get as inputs
    instances_file = base_dir + 'data\Benchmarks\FMJ\sfjs01.txt'
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


    # Initialise as empty array/list
    jobs_array = []

    if os.path.splitext(instances_file)[1] == '.txt':

        operations, machines =  import_data.import_txt_data(instances_file)

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
                succ_ops = op.succ
                succ_ops.append(ops[1])
                op.succ = succ_ops
            print(op.op_num, op.job_num, op.pre, op.succ)

            if ops[1] not in generated_ops:
                op = Operation(ops[2], ops[1], pre_op=ops[0])
                temp_jobs.append(op)
                generated_ops.append(ops[1])
            else:
                op = [item for item in temp_jobs if item.op_num == ops[1]][0]
                if op == None:
                    raise Exception("Operation with this operation label not found!")
                pre_ops = op.pre
                pre_ops.append(ops[0])
                op.pre = pre_ops
            print(op.op_num, op.job_num, op.pre, op.succ)
        
        print(temp_jobs)

        ## IMPORT MACHINES ##
        for i in range(len(machines)):
            eligible_machines = []
            j = 1
            while j < len(machines[i]):
                machine = machines[i][j]
                process_time = machines[i][j+1]
                eligible_machines.append((machine, process_time))
                j += 2
            op = [item for item in temp_jobs if item.op_num == str(i)][0]
            if op == None:
                raise Exception("Operation with this operation label not found!")
            op.machines = eligible_machines
        
            print(op.op_num, op.job_num, op.machines)
        
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

        print(jobs_array)
        #print(machines)
    else:
        raise Exception("Invalid file type")
        

### BEGIN MAIN PROGRAM ###
if __name__ == '__main__':
    msts()