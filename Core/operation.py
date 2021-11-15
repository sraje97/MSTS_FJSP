############################################################

class Operation:
    def __init__(self, job_num, op_num, pre_op=None, succ_op=None, branch='S', \
                    eligible_machines='', assigned_machine=None, PT=0, ST=0, FT=0):
        
        self.job_num = job_num
        self.op_num = op_num
        self.pre = pre_op
        self.succ = succ_op
        self.series = branch
        self.machines = eligible_machines
        self.mach_num = assigned_machine
        self.processing_time = PT
        self.setup_time = ST
        self.finish_time = FT