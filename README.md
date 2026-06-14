# Abstract

In the early stages of federated learning (FL), clients were commonly assumed to perform an identical number of local updates per communication round, leading researchers to adopt a simple weighted aggregation method (e.g., FedAvg) across clients' local models.
However, in real-world edge computing scenarios, device heterogeneity and data heterogeneity often cause clients to perform varying numbers of local updates.
Although numerous studies addressed the performance degradation due to heterogeneous local updates through algorithmic improvements (e.g., normalized aggregation methods like FedNova), these approaches do not inherently motivate clients to contribute sufficient computational effort. 
Designing effective incentives therefore requires a precise characterization of how clients' local effort and data heterogeneity jointly affect global model convergence. 
To this end, we derive a generalized convergence bound for FedNova that explicitly accounts for both data heterogeneity and heterogeneous numbers of local updates. 
Building on this characterization, we investigate the contract design problem for FL under clients' two-dimensional private information on computational cost and data heterogeneity.
We develop optimal contracts under complete, weakly incomplete, and strongly incomplete information scenarios.
Under complete and weakly incomplete information, we derive a closed-form solution showing that the server should only incentivize clients with both the lowest unit cost and the lowest data heterogeneity.
Under strongly incomplete information, we transform the combinatorial problem into a convex optimization problem via a practical assumption of positive probabilities for client types.
Our method achieves average cost reductions of 15\% and 5.5\% over the uniform contract and Stackelberg game benchmarks, respectively.


# Usage
Researchers can reproduce the simulation results by following the parameter settings in our main paper. 

## Code Structure

* `demo_compare_mechanisms.ipynb`: Implements the convex optimization tool used to generate Figure 3-6.
* `fednova_main.py`: The main training script for [FedNova](https://github.com/JYWa/FedNova). Customize parameters via the `args_parser()` function.
* `convergence_analysis.ipynb`: Performs convergence analysis using FedNova results to plot Figure 7-9.


We conducted 7 experiments based on code from [FedNova](https://github.com/JYWa/FedNova).
Figure 7-8 utilizes experiments 1-4 while Figure 9 utilizes expriments 5-6.
|Experiment|IID or alpha|local updates|
|---|---|---|
|1|0.5|U(0,20)|
|2|0.1|U(0,20)|
|3|IID|U(0,20)|
|4|1|U(0,20)|
|5|0.1|5|
|6|0.1|20|
|7|0.1|1|


## Requirements

The implementation runs on:
* Python 3.5
* Pytorch 1.0.0
* TorchVision 0.2.1

