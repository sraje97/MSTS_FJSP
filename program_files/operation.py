# Import python packages
import os
import numpy as np


class Operation:
    def __init__(self, op_array):
        self.job_num = op_array[0]
        self.op_num = op_array[1]
        self.task = op_array[2]
        self.processing_time = op_array[3]
        self.setup_time = op_array[4]
        self.machining_time = self.processing_time + self.setup_time
        self.pre = op_array[6]
        self.succ = op_array[7]
        self.series = op_array[8]
        self.mach_num = None
        self.finish_time = 0