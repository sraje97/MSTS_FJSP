# MSTS_FJSP

This project utilises the Multi-Start Genetic Algorithm Tabu Search (MS-GATS) method for the classic Flexible Job-Shop Scheduling Problem (FJSP), Extended-FJSP (EFJSP), and EFJSP with Transportation Constraints.

* "msts_algo_new.py" is the main entry point for this algorithm, and used for running the simulation.
* Certain parameters can be set in this file including:
* TS_cnt_max - Tabu Search Iteration Counter
* p_exp_con - Probability for choosing in e-greedy policy 
* p_MA_OS - Probability of choosing MA or OS for critical path
* epochs - Maximum number of epochs
* eps_decay - Decay rate for e-greedy policy
* pop_size - Population size
* MA_algo_choice - Initial Dispatching Rules for machine assignment [Random, Greedy, LUM]
* OS_algo_choice - Initial Dispatching Rules for operation sequencing [Random, ERT, LRMT]


'data' directory has each dataset in individual directories.
Including modified datasets with transportation times (T_times)

Virtual environment - use MEVenv

Outputs are stored in 'output_models' named by date_PC-number.
