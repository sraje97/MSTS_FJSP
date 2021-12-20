from collections import defaultdict
import csv

import pandas as pd
import plotly.express as px

# File name
filepath = ''
#filepath = input("Filename:")
if filepath == '':
    filepath = 'H:\\Documents\\Github\\MSTS_FJSP\\output_models\\21-12-2021-12-39-27_EN-403038\\MK111\\best_design.csv'

mach_dict = defaultdict()

with open(filepath, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        if line_count == 3:
            line_count += 1
        elif line_count < 3:
            line_count += 1
            continue
        elif row[0] == 'Job':
            line_count += 1
            break
        else:
            res = [idx for idx in eval(row[1]) if isinstance(idx, tuple)]

            mach_dict[row[0]] = res
            line_count += 1

op_schedules = []
mach_names = []
job_names = []

for mach, schedule in mach_dict.items():
    if mach not in mach_names:
        mach_names.append(mach)
    for tupl in schedule:
        op_job = "J" + tupl[0][tupl[0].find("O") + 1 : tupl[0].find("_")]
        if op_job not in job_names:
            job_names.append(op_job)
        temp_dict = dict(Job=op_job, Start=tupl[1], Finish=tupl[2], Machine=mach, Details=tupl[0])
        op_schedules.append(temp_dict)

mach_names = sorted(mach_names, key=lambda x: int("".join([i for i in x if i.isdigit()])))
job_names = sorted(job_names, key=lambda x: int("".join([i for i in x if i.isdigit()])))

gantt_df = pd.DataFrame(op_schedules)
gantt_df['Delta'] = gantt_df['Finish'] - gantt_df['Start']

# Colorscale for colors.sequential and colors.qualitative https://plotly.com/python/builtin-colorscales/
# Good colours: Teal, Tealgrn, darkcyan, slategray, Plotly
fig = px.timeline(gantt_df, x_start="Start", x_end="Finish", y="Machine", color="Job", text="Details",
                color_discrete_sequence= px.colors.qualitative.Plotly,
                #color_discrete_sequence=["lightslategray"],
                category_orders= { "Machine" : mach_names, "Job" : job_names }
                )
fig.layout.xaxis.type = 'linear'
fig.update_xaxes(rangemode="tozero")
fig.update_yaxes(zeroline=True)
for d in fig.data:
    filt = gantt_df['Job'] == d.name
    d.x = gantt_df[filt]['Delta'].tolist()
fig.show()