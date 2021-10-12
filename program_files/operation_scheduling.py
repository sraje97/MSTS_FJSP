import operator
import networkx as nx

import machine_assignment_algo
import graph

def get_SMT(operations):
	return min(operations, key=operator.itemgetter(1))

def get_LRMT(operations):
	return max(operations, key=operator.itemgetter(1))


def get_machine_finish_time(op_schedule):
	return max(op_schedule, key=operator.itemgetter(2))

def get_start_time(operation, prev_operation, machine_graph):
	machine = operation.mach_num
	op_schedule = machine_graph.nodes[machine]['op_schedule']
	machine_finish_time = get_machine_finish_time(op_schedule)
	if prev_operation == '':
		return machine_finish_time[-1]
	else:
		return max(prev_operation.finish_time, machine_finish_time[-1])

def get_total_remaining_time(job_array):
	total_time = 0
	for op in job_array:
		total_time += op.machining_time
	return total_time


def schedule_operation(job_array, operation, machine_graph, scheduled_operations):
	if operation.pre == '':
		start_time = get_start_time(operation, '', machine_graph)
		prev_machine = ''
		#schedule_operation(operation, machine_graph, '', start_time)
	elif (operation.pre in scheduled_operations):
		prev_operation = [item for item in job_array if item.op_num == operation.pre][0]
		start_time = get_start_time(operation, prev_operation, machine_graph)
		prev_machine = prev_operation.mach_num
		#schedule_operation(operation, machine_graph, prev_operation.mach_num, start_time)
	elif type(operation.pre) is tuple:
		if (operation.pre[0] in scheduled_operations) and (operation.pre[1] in scheduled_operations):
			op_P1_pre = [item for item in job_array if item.op_num == operation.pre[0]][0]
			op_P2_pre = [item for item in job_array if item.op_num == operation.pre[1]][0]
			if op_P1_pre.finish_time > op_P2_pre.finish_time:
				prev_operation = op_P1_pre
			else:
				prev_operation = op_P2_pre
			start_time = get_start_time(operation, prev_operation, machine_graph)
			prev_machine = prev_operation.mach_num
		#	schedule_operation(operation, machine_graph, prev_operation.mach_num, start_time)
	else:
		print("Error")


	machine = operation.mach_num
	op_schedule = machine_graph.nodes[machine]['op_schedule']
	idx = [i for i, tupl in enumerate(op_schedule) if tupl[0] == operation.op_num][-1]

	if prev_machine == '':
		transition_time = 0
	elif prev_machine == machine:
		transition_time = 0
	else:
		transition_time = machine_assignment_algo.get_transition_time(machine_graph, machine, prev_machine)

	start_time += transition_time
	finish_time = machine_assignment_algo.calculate_machining_time(operation, machine_graph, machine) + start_time
	op_tuple = (operation.op_num, start_time, finish_time)
	op_schedule[idx] = op_tuple
	nx.set_node_attributes(machine_graph, {machine: {'op_schedule': op_schedule} } )
	operation.finish_time = finish_time

def add_next_executable_operation(job_array, operation, machine_graph, executable_operations_list, scheduled_operations, algo_choice):
	idx = [i for i, tupl in enumerate(executable_operations_list) if tupl[0].op_num == operation.op_num][0]
	remaining_time = executable_operations_list[idx][1]
	executable_operations_list.pop(idx)
	succ_operation = operation.succ
	
	if succ_operation == '':
		return executable_operations_list
	elif type(succ_operation) is tuple:
		for op in succ_operation:
			operation = [item for item in job_array if item.op_num == op][0]
			machining_time = machine_assignment_algo.calculate_machining_time(operation, machine_graph, operation.mach_num)
			
			if algo_choice == "SMT":
				executable_operations_list.append((operation, machining_time))
			elif algo_choice == "LRMT":
				remaining_time -= machining_time
				executable_operations_list.append((operation, remaining_time))
			else:
				return executable_operations_list	# TODO: ERT
	else:
		operation = [item for item in job_array if item.op_num == succ_operation][0]
		
		if type(operation.pre) is tuple:
			if not((operation.pre[0] in scheduled_operations) and (operation.pre[1] in scheduled_operations)):
				return executable_operations_list
		
		machining_time = machine_assignment_algo.calculate_machining_time(operation, machine_graph, operation.mach_num)
		if algo_choice == "SMT":
			executable_operations_list.append((operation, machining_time))
		elif algo_choice == "LRMT":
			remaining_time -=  machining_time
			executable_operations_list.append((operation, machining_time))
		else:
			return executable_operations_list	# TODO: ERT

	return executable_operations_list

############################################################
#             OPERATION SCHEDULING ALGORITHMS          	   #
############################################################

def schedule_SMT(jobs_array, machine_graph):
	scheduled_operations = []
	next_executable_operations = []

	for i in range(len(jobs_array)):
		operation = jobs_array[i][0]
		machining_time = machine_assignment_algo.calculate_machining_time(operation, machine_graph, operation.mach_num)
		next_executable_operations.append((operation, machining_time))

	while next_executable_operations:
		operation = get_SMT(next_executable_operations)[0]
		job_array = jobs_array[int(operation.job_num[1:]) - 1]
		
		schedule_operation(job_array, operation, machine_graph, scheduled_operations)

		scheduled_operations.append(operation.op_num)
		next_executable_operations = add_next_executable_operation(job_array, operation, machine_graph, next_executable_operations, scheduled_operations, "SMT")
	return 1

def schedule_LRMT(jobs_array, machine_graph):
	scheduled_operations = []
	next_executable_operations = []

	for i in range(len(jobs_array)):
		operation = jobs_array[i][0]
		total_remaining_time = get_total_remaining_time(jobs_array[i])
		next_executable_operations.append((operation, total_remaining_time))
	
	while next_executable_operations:
		operation = get_LRMT(next_executable_operations)[0]
		job_array = jobs_array[int(operation.job_num[1:]) - 1]

		schedule_operation(job_array, operation, machine_graph, scheduled_operations)
		scheduled_operations.append(operation.op_num)

		next_executable_operations = add_next_executable_operation(job_array, operation, machine_graph, next_executable_operations, scheduled_operations, "LRMT")
	return 1