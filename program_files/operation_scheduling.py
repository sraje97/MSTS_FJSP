import operator
import networkx as nx

import machine_assignment_algo
import graph


# Get operation with shortest machining time (SMT)
def get_SMT(operations):
	return min(operations, key=operator.itemgetter(1))

# Get operation with largest remaining machining time (LRMT)
def get_LRMT(operations):
	return max(operations, key=operator.itemgetter(1))

# Get operation with the earliest release time (ERT)
def get_ERT(operations):
	return min(operations, key=operator.itemgetter(1))

# Get the finish time of an operation in the machine's schedule
def get_machine_finish_time(op_schedule):
	return max(op_schedule, key=operator.itemgetter(2))

# Get the the earliest start time for an operation
def get_start_time(operation, prev_operation, machine_graph):
	machine = operation.mach_num
	
	# Get machine's schedule
	op_schedule = machine_graph.nodes[machine]['op_schedule']
	machine_finish_time = get_machine_finish_time(op_schedule)
	
	# Find the latest time an operation may start
	if prev_operation == '':
		return machine_finish_time[-1]
	else:
		return max(prev_operation.finish_time, machine_finish_time[-1])

# Get the total remaining workload of a job
def get_total_remaining_time(job_array, op=None):
	total_time = 0

	if op == None:		
		# Sum up all operation's machine times in a job
		for op in job_array:
			total_time += op.machining_time
		return total_time
	else:
		# If operation is specified, sum up total machining time from operation
		while op.succ != '':
			total_time += op.machining_time
			if type(op.succ) is tuple:
				succ_op = op.succ[0]
				op = [item for item in job_array if item.op_num == succ_op][0]
			else:
				op = [item for item in job_array if item.op_num == op.succ][0]
		total_time += op.machining_time
		return total_time

# Check if operation is in the executable operations list
def get_flag(operation, executable_operations_list):
	for item in executable_operations_list:
		if operation == item[0]:
			return True
	return False

# Helper function to append operation, operation's machine, and specified time to executable operations list
def append_operation_tuple(job_array, operation, machine_graph, executable_operations_list, algo_choice):
	# Get machining time for current operation
	machining_time = machine_assignment_algo.calculate_machining_time(operation, machine_graph, operation.mach_num)
	
	# Depending on the scheduling algorithm, add either machining/remaining/release time
	if algo_choice == "SMT":
		executable_operations_list.append((operation, machining_time))
	elif algo_choice == "LRMT":
		remaining_time = get_total_remaining_time(job_array, operation)
		remaining_time -=  machining_time
		executable_operations_list.append((operation, machining_time))
	else:
		release_time = operation.finish_time
		executable_operations_list.append((operation, release_time))
	return executable_operations_list


# Schedule an operation on the machine (add start and finish time on machine schedule)
def schedule_operation(job_array, operation, machine_graph, scheduled_operations):
	# Get the start time depending on previous operation or current machine's finishing time
	if operation.pre == '':
		start_time = get_start_time(operation, '', machine_graph)
		prev_machine = ''
	elif (operation.pre in scheduled_operations):
		prev_operation = [item for item in job_array if item.op_num == operation.pre][0]
		start_time = get_start_time(operation, prev_operation, machine_graph)
		prev_machine = prev_operation.mach_num
	elif type(operation.pre) is tuple:
		# If previous operations are parallel operations, must wait till both have been scheduled
		if (operation.pre[0] in scheduled_operations) and (operation.pre[1] in scheduled_operations):
			op_P1_pre = [item for item in job_array if item.op_num == operation.pre[0]][0]
			op_P2_pre = [item for item in job_array if item.op_num == operation.pre[1]][0]
			
			# Get the operation which finished the latest
			if op_P1_pre.finish_time > op_P2_pre.finish_time:
				prev_operation = op_P1_pre
			else:
				prev_operation = op_P2_pre
			
			start_time = get_start_time(operation, prev_operation, machine_graph)
			prev_machine = prev_operation.mach_num
		else:
			# Check if op_pre[0] in scheduled or op_pre[0] in next_executable
				# If neither
			pass
	else:
		print("Previous operation not found")

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
		transition_time = machine_assignment_algo.get_transition_time(machine_graph, machine, prev_machine)

	# Add transition time to the start time
	start_time += transition_time
	# Finish time is the starting time plus the machining time (ST + MT)
	finish_time = machine_assignment_algo.calculate_machining_time(operation, machine_graph, machine) + start_time
	
	# Update the tuple in the machine's schedule
	op_tuple = (operation.op_num, start_time, finish_time)
	op_schedule[idx] = op_tuple
	nx.set_node_attributes(machine_graph, {machine: {'op_schedule': op_schedule} } )
	
	# Update the operation's finish time
	operation.finish_time = finish_time

# Add operation to the list of next available/schedulable operations
def add_next_executable_operation(job_array, operation, machine_graph, executable_operations_list, scheduled_operations, algo_choice):
	# Get the index of the operation from the list and delete it from the list
	idx = [i for i, tupl in enumerate(executable_operations_list) if tupl[0].op_num == operation.op_num][0]
	remaining_time = executable_operations_list[idx][1]
	executable_operations_list.pop(idx)

	# Get the next operation in the job array
	op_succ = operation.succ
	
	# Add the next operation to the next executable list if all previous operations are scheduled
	if op_succ == '':
		return executable_operations_list
	elif type(op_succ) is tuple:
		# If multiple successive operations, add both operations to the list
		for op in op_succ:
			succ_operation = [item for item in job_array if item.op_num == op][0]
			executable_operations_list = append_operation_tuple(job_array, succ_operation, machine_graph, executable_operations_list, algo_choice)
	else:
		succ_operation = [item for item in job_array if item.op_num == op_succ][0]
		
		# In case of parallel previous operations, check if both have been scheduled
		if type(succ_operation.pre) is tuple:
			if not((succ_operation.pre[0] in scheduled_operations) and (succ_operation.pre[1] in scheduled_operations)):
				
				# Find which of the previous operation hasn't been scheduled
				if succ_operation.pre[0] not in scheduled_operations:
					operation = [item for item in job_array if item.op_num == succ_operation.pre[0]][0]
				elif succ_operation.pre[1] not in scheduled_operations:
					operation = [item for item in job_array if item.op_num == succ_operation.pre[1]][0]
				else:
					return executable_operations_list
				
				# Check if this operation is not available for execution
				if not(get_flag(operation, executable_operations_list)):
					
					# Keep following the operation path back until you find the first operation that has been scheduled
					while operation.pre != '':
						pre_operation = [item for item in job_array if item.op_num == operation.pre][0]
						# If a previous operation has been scheduled, then add this operation to executable operations list
						if operation.pre in scheduled_operations:
							executable_operations_list = append_operation_tuple(job_array, operation, machine_graph, executable_operations_list, algo_choice)
							return executable_operations_list
						elif get_flag(pre_operation, executable_operations_list):
							return executable_operations_list
						else:
							operation = pre_operation
					# Add the last found operation to executable operations list
					executable_operations_list = append_operation_tuple(job_array, operation, machine_graph, executable_operations_list, algo_choice)
					return executable_operations_list

				return executable_operations_list
		
		executable_operations_list = append_operation_tuple(job_array, succ_operation, machine_graph, executable_operations_list, algo_choice)

	return executable_operations_list

############################################################
#             OPERATION SCHEDULING ALGORITHMS          	   #
############################################################

# Schedule operations using the Shortest Machining Time (SMT) algorithm
def schedule_SMT(jobs_array, machine_graph):
	# Initialise lists to store already scheduled and next available operations
	scheduled_operations = []
	next_executable_operations = []

	# For each starting operation (of a job), enable it to be available for scheduling
	for i in range(len(jobs_array)):
		operation = jobs_array[i][0]
		machining_time = machine_assignment_algo.calculate_machining_time(operation, machine_graph, operation.mach_num)
		next_executable_operations.append((operation, machining_time))

	while next_executable_operations:
		# Get the operation with shortest machining time
		operation = get_SMT(next_executable_operations)[0]
		job_array = jobs_array[int(operation.job_num[1:]) - 1]

		# Schedule that operation onto it's assigned machine
		schedule_operation(job_array, operation, machine_graph, scheduled_operations)

		scheduled_operations.append(operation.op_num)
		# Add operation's successors to the executable operations list
		next_executable_operations = add_next_executable_operation(job_array, operation, machine_graph, next_executable_operations, scheduled_operations, "SMT")
	return 1

# Schedule operations using the Largets Remaining Machining Time (LRMT) algorithm
def schedule_LRMT(jobs_array, machine_graph):
	scheduled_operations = []
	next_executable_operations = []

	# For each starting operation (of a job), enable it to be available for scheduling
	for i in range(len(jobs_array)):
		operation = jobs_array[i][0]
		total_remaining_time = get_total_remaining_time(jobs_array[i])
		next_executable_operations.append((operation, total_remaining_time))
	
	while next_executable_operations:
		# Get the operation belonging to the job with the largest remaining machining time
		operation = get_LRMT(next_executable_operations)[0]
		job_array = jobs_array[int(operation.job_num[1:]) - 1]

		# Schedule that operation onto it's assigned machine
		schedule_operation(job_array, operation, machine_graph, scheduled_operations)
		scheduled_operations.append(operation.op_num)

		# Add operation's successors to the executable operations list
		next_executable_operations = add_next_executable_operation(job_array, operation, machine_graph, next_executable_operations, scheduled_operations, "LRMT")
	return 1

# Schedule operations using the Earliest Release Time (ERT) algorithm
def schedule_ERT(jobs_array, machine_graph):
	scheduled_operations = []
	next_executable_operations = []

	# For each starting operation (of a job), enable it to be available for scheduling
	for i in range(len(jobs_array)):
		operation = jobs_array[i][0]
		next_executable_operations.append((operation, operation.finish_time))
	
	while next_executable_operations:
		# Get the operation with the earliest release (starting) time
		operation = get_ERT(next_executable_operations)[0]
		job_array = jobs_array[int(operation.job_num[1:]) - 1]

		# Schedule that operation onto it's assigned machine
		schedule_operation(job_array, operation, machine_graph, scheduled_operations)
		scheduled_operations.append(operation.op_num)

		# Add operation's successors to the executable operations list
		next_executable_operations = add_next_executable_operation(job_array, operation, machine_graph, next_executable_operations, scheduled_operations, "ERT")
	return 1