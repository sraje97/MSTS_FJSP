from operation import Operation
import networkx as nx
import graph
import copy

jobarray = []
x=1

if x == 1:
    op = Operation(op_num='0', job_num='J1', pre_op=None, succ_op='1')
    jobarray.append(op)
    op = Operation(op_num='1', job_num='J1', pre_op='0', succ_op='2')
    jobarray.append(op)
    op = Operation(op_num='2', job_num='J1', pre_op='1', succ_op='3')
    jobarray.append(op)
    op = Operation(op_num='3', job_num='J1', pre_op='2', succ_op='8')
    jobarray.append(op)
    op = Operation(op_num='8', job_num='J1', pre_op=['3', '7'], succ_op='9')
    jobarray.append(op)
    op = Operation(op_num='4', job_num='J1', pre_op=None, succ_op='5')
    jobarray.append(op)
    op = Operation(op_num='5', job_num='J1', pre_op='4', succ_op='6')
    jobarray.append(op)
    op = Operation(op_num='6', job_num='J1', pre_op='5', succ_op='7')
    jobarray.append(op)
    op = Operation(op_num='7', job_num='J1', pre_op='6', succ_op='8')
    jobarray.append(op)
    op = Operation(op_num='9', job_num='J1', pre_op='8', succ_op=None)
    jobarray.append(op)
else:
    op = Operation(op_num='0', job_num='J1', pre_op=None, succ_op='1')
    jobarray.append(op)
    op = Operation(op_num='1', job_num='J1', pre_op='0', succ_op='2')
    jobarray.append(op)
    op = Operation(op_num='2', job_num='J1', pre_op='1', succ_op='3')
    jobarray.append(op)
    op = Operation(op_num='3', job_num='J1', pre_op='2', succ_op='4')
    jobarray.append(op)
    op = Operation(op_num='4', job_num='J1', pre_op='3', succ_op='5')
    jobarray.append(op)
    op = Operation(op_num='5', job_num='J1', pre_op='4', succ_op=['6', '10'])
    jobarray.append(op)
    op = Operation(op_num='6', job_num='J1', pre_op='5', succ_op='7')
    jobarray.append(op)
    op = Operation(op_num='7', job_num='J1', pre_op='6', succ_op='8')
    jobarray.append(op)
    op = Operation(op_num='8', job_num='J1', pre_op='7', succ_op='9')
    jobarray.append(op)
    op = Operation(op_num='9', job_num='J1', pre_op='8', succ_op=None)
    jobarray.append(op)
    op = Operation(op_num='10', job_num='J1', pre_op='5', succ_op='11')
    jobarray.append(op)
    op = Operation(op_num='11', job_num='J1', pre_op='10', succ_op='12')
    jobarray.append(op)
    op = Operation(op_num='12', job_num='J1', pre_op='11', succ_op='13')
    jobarray.append(op)
    op = Operation(op_num='13', job_num='J1', pre_op='12', succ_op=None)
    jobarray.append(op)


#for oper in jobarray:
#    print(oper.op_num, oper.pre, oper.succ)



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

final_ops = []
for op in jobarray:
    if op.succ == None:
        final_ops.append(op)

branchlist = []
if len(final_ops) > 1:
#    print("Many end operations")
    branchlist = recurse_forward(jobarray, jobarray[0].op_num, branchlist)
else:
#    print("Single end operation")
    branchlist = recurse_reverse(jobarray, final_ops[0].op_num, branchlist)

#print(branchlist)

breakflag = False
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
            templist = branched_copy
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
            oper = [item for item in jobarray if item.op_num == op][0]
            oper.series = branch_label
        branch_label = 'P1'
    else:
        for op in sequence:
            if op not in longestseq:
                oper = [item for item in jobarray if item.op_num == op][0]
                oper.series = branch_label
        branch_label = 'P' + str(int(branch_label[1]) + 1)

#for op in jobarray:
#    print(op.op_num, op.series)


TG = nx.Graph()
graph.add_node(TG, 'M1')
graph.add_node(TG, 'M2')

mach1 = [item for item in TG.nodes if item == "M1"][0]
print(mach1)
print(graph.get_node_info(TG, mach1))
op_schedle = TG.nodes[mach1]['op_schedule']
op_schedle.append(("O11", 0, 0))
nx.set_node_attributes(TG, {mach1: {'op_schedule': op_schedle}})
print(graph.get_node_info(TG, mach1))