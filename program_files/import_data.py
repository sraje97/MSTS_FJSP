import pandas as pd
from pandas.core.indexes.base import ensure_index

# Show all columns when printing
pd.set_option('display.max_columns', None)

def import_job_data(filename):

    # Read only columns A,B,C,H,I,J,K,L,M,N
    operation_data = pd.read_excel(filename, usecols="A,B,C,H,I,J,K,L,M,N", engine='openpyxl')
    #print(operation_data)
    return operation_data

def import_machine_data(filename):

    # Read data from Sheet1 of Machine_Data file
    machine_data = pd.read_excel(filename, sheet_name='Sheet1', engine='openpyxl')
    #print(machine_data)
    return machine_data

def import_transition_times(filename):

    # Read data from Sheet2 of Machine_Data file
    t_times = pd.read_excel(filename, sheet_name='Sheet2', engine='openpyxl')
    return t_times