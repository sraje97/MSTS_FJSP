from operator import ne
import numpy

def schedule_SMT(jobs_array, machine_graph):
    unscheduled_operations = []
    schehedule_operations = []
    next_executable_operations = []

    for i in range(len(jobs_array)):
        next_executable_operations.append(jobs_array[i][0])
        
    