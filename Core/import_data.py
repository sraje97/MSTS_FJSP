import numpy as np

def import_txt_data(filename):
    with open(filename) as f:
        line = f.readline()
        line = line.split()
        num_ops, num_edges, num_machs = int(line[0]), int(line[1]), int(line[2])

        op_precedence = []
        op_machines = []
        t_times = np.zeros([num_machs, num_machs], dtype=np.uint8)

        for i in range(num_edges):
            line = f.readline()
            line = line.split()
            op_precedence.append(line)
        
        for i in range(num_ops):
            line = f.readline()
            line = line.split()
            line.insert(0, str(i))
            op_machines.append(line)

        for i in range(num_machs):
            line = f.readline()
            if line == '':
                t_times = None
                break
            line = [int(x) for x in line.split()]
            t_times[i] = np.array(line)

    f.close
    return op_precedence, op_machines, t_times