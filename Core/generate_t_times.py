import random
import numpy as np

random.seed(1)

num_machs = 26
max_PT = 200

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
            randnum = random.randrange(ten_pct, thirty_pct+1)
            nparray[i][j] = randnum
            nparray[j][i] = randnum

for i in range(num_machs):
    print(*nparray[i], sep='\t')