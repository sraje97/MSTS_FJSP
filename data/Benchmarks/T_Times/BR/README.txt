These are instances proposed in

    P. Brandimarte. Routing and scheduling in a flexible job shop by tabu 
    search. Annals of Operations Research, 41(3):157-183, 1993,

for the Flexible Job-Shop Scheduling Problem. The format is different 
from the original. In our format, the only information needed is about 
the operations of the jobs, precedence relations and machines.

Let N be the total number of operations of all jobs, A be the number of 
precedence relations and K be the number of machines. The operations are 
labeled from 0 to N-1 and the machines are labeled from 0 to K-1.

The description of the instance starts with a line that contains the 
numbers N, A and K. Each of the next A lines contains two distinct 
operations, U and V, indicating that U must be processed before V. 
Finally, the last N lines describe the operations: the (A + 2)-th
line corresponds to operation 0, the (A + 3)-th line corresponds to 
operation 1 and so on. The description of operations begins with the
number M of machines which can process the operation followed by
M pairs of numbers indicating, respectively, the label of a machine 
and the time that machine takes to process the operation.

Any line starting with a # is a comment.

