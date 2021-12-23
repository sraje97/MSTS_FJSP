import random
import statistics
import numpy as np

random.seed(1)

def import_txt_data(filename):
    with open(filename) as f:
        line = f.readline()
        line = line.split()
        num_ops, num_edges, num_machs = int(line[0]), int(line[1]), int(line[2])

        op_precedence = []
        op_machines = []
        t_times = []

        for i in range(num_edges):
            line = f.readline()
            line = line.split()
            op_precedence.append(line)
        
        for i in range(num_ops):
            line = f.readline()
            line = line.split()
            op_machines.append(line[2::2])

        flat_list1 = [int(item) for sublist in op_machines for item in sublist]

        minP = min(flat_list1)
        maxP = max(flat_list1)
        medP = statistics.median(flat_list1)

        for i in range(num_machs):
            line = f.readline()
            if line == '':
                t_times = None
                break
            t_times.append( [int(x) for x in line.split()] )
        
        flat_list2 = [int(item) for sublist in t_times for item in sublist]
        flat_list2 = [i for i in flat_list2 if i != 0]
        
        minT = min(flat_list2)
        maxT = max(flat_list2)

    f.close
    #return flat_list1, flat_list2, num_machs
    return minP, maxP, medP, minT, maxT, num_machs


def generate_t_times(medP, num_machs):
    max_PT = medP
    #max_PT = max(fl1)

    ten_pct = round(0.1 * max_PT)
    thirty_pct = round(0.3 * max_PT)

    print("LB:", ten_pct, "\tUB:", thirty_pct)

    nparray = np.zeros([num_machs, num_machs], dtype=np.uint8)

    for i in range(num_machs):
        for j in range(num_machs):
            if i == j:
                nparray[i][j] = 0
            elif nparray[i][j] > 0:
                continue
            elif nparray[j][i] > 0:
                continue
            else:
                randnum = max(1, random.uniform(ten_pct, thirty_pct+1))
                nparray[i][j] = randnum
                nparray[j][i] = randnum

    for i in range(num_machs):
        print(*nparray[i], sep='\t')


YFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
DAFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', \
         '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
SFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
MFJS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
MK = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']

minmaxes = dict()
for file_num in MK:
    test_name = "MK" + file_num + ".txt"
    filename = "data\Benchmarks\T_Times\BR\\" + test_name
    #fl1, fl2, num_machs = import_txt_data(filename)
    a,b,c,d,e,f = import_txt_data(filename)
    x = generate_t_times(c, f)

    minmaxes[test_name] = (a,b,c,d,e,f)

for key, val in minmaxes.items():
    print(key, val)