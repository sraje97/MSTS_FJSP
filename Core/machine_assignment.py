import random
import networkx as nx
from networkx.algorithms.flow import maxflow


bigM = 999_999

############################################################

# Add operation to the machine's schedule and store machine in operation's 'mach_num' variable
def assign_operation_to_machine(operation, machine_graph, machine):
    # Add machine label to operation
    operation.mach_num = machine

    # Add operation to machine/node's attribute
    # Don't assign starting and finishing time yet
    op_schedule = machine_graph.nodes[machine]['op_schedule']
    op_schedule.append( (operation.op_num, 0, 0) )

    nx.set_node_attributes(machine_graph, {machine: {'op_schedule': op_schedule } } )

def get_min_FMT(operation, machine_graph, eligible_machines):
    minFMT = bigM

    for machine in eligible_machines:
        # Calculate machining time for specific operation-machine combination [MT = (PT * alpha) + (ST * beta)]
        machining_time = round( operation.processing_time * machine_graph.nodes[machine[0]]['alpha'] )

        if machining_time < minFMT:
            # Assign the machine with FMT for operation
            minFMT = machining_time
            best_machine = machine[0]
    # Assign machine to operation
    assign_operation_to_machine(operation, machine_graph, best_machine)

############################################################
#               MACHINE ASSIGNMENT ALGORITHMS              #
############################################################

# Randomly assign machine to all operations
def run_random(jobs_array, machine_graph):
    for job in jobs_array:
        for operation in job:
            # Get list of all eligible machines
            eligible_machines = operation.machines

            if len(eligible_machines) > 1:
                # Choose any random machine if more than one compatible machine
                machine = random.choice(eligible_machines)[0]
            else:
                # Choose the only eligible machine
                machine = eligible_machines[0][0]
            # Assign the machine to the operation (and vice versa)
            assign_operation_to_machine(operation, machine_graph, machine)
    return 1

# Greedily assign machine to all operations using either 
# FMT - Fastest Machining Time (default) or LMT - Longest Machining Time
def run_greedy(jobs_array, machine_graph, greedy_type="FMT"):
    for job in jobs_array:
        for operation in job:
            # Get list of all eligible machines
            eligible_machines = operation.machines

            if greedy_type == "FMT":
                # Use default FMT
                get_min_FMT(operation, machine_graph, eligible_machines)
            else:
                # Use LMT
                maxFMT = 0

                for machine in eligible_machines:
                    machining_time = round( operation.processing_time * machine_graph.nodes[machine[0]]['alpha'] )

                    if machining_time > maxFMT:
                        # Assign the machine with LMT for operation
                        maxFMT = machining_time
                        best_machine = machine[0]
                # Assign the machine to the operation (and vice versa)
                assign_operation_to_machine(operation, machine_graph, best_machine)
    return 1

# Use adapted Dijkstra's algorithm to assign machine with shortest path for each job
def run_shortest_path(jobs_array, machine_graph):
    for job in jobs_array:
        # Create priority queue for node/tuple (operation, path sequence, total time)
        priority_queue = []
        # Keep list of processed parallel operations
        processed_parallel_ops = []
        # Use to store best final node
        best_final_node = (None, None, bigM)

        operation = job[0]
        eligible_machines = operation.machines

        for machine in eligible_machines:
            # For each eligible machine add tuple of (operation, machince, total time) 
            # to the priority queue
            total_time = round( operation.processing_time * machine_graph.nodes[machine[0]]['alpha'] )
            path_node = (operation, [machine[0]], total_time)
            priority_queue.append(path_node)
        
        while priority_queue:
            pass    # TODO