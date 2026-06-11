Care Engine Analytics - Experiments
===================================

This directory contains scripts and configurations for running agent-based model (ABM) experiments using the CareEngine Java simulator.

Files
-----
* sensitivity.py: Performs sensitivity analysis by sweeping specific parameters across a range of values for multiple policies (e.g. Need-Prioritization, Risk-Stratification, FCFS).
* exploration.py: Conducts detailed temporal exploration of key state variables (e.g. expectations, unserved needs, care seeking behaviour, treatment delivery) for chosen scenarios.
* sensitivity_config.json.template: A template configuration for running sensitivity analyses.
* exploration_config.json.template: A template configuration for running detailed temporal explorations.

How to Run
----------
Both scripts require python 3 with dependencies (pandas, numpy, matplotlib, seaborn) and can be executed using the command line:

1. Sensitivity Analysis:
   python3 experiments/sensitivity.py --config <path_to_config_json> --output-root <output_directory_root>

   Example:
   python3 experiments/sensitivity.py --config experiments/sensitivity_config.json.template --output-root experiments/outputs

2. Detailed Exploration:
   python3 experiments/exploration.py --config <path_to_config_json> --output-root <output_directory_root>

   Example:
   python3 experiments/exploration.py --config experiments/exploration_config.json.template --output-root experiments/outputs

Configuration Structure
-----------------------
Configurations are specified in JSON format and organized as follows:

1. "global_settings": Contains execution-wide controls:
   - engine_path: Path to the ABM Server .jar file
   - port: Socket port to communicate with the Java server (default: 8383)
   - batch_size: Number of parallel simulations sent per batch (default: 100)
   - reps: Number of replicate simulations to run per configuration
   - reproduce_line: Flag indicating whether to use standard reproduction baseline (default: true)
   - state_variables: List of state variables to record and return from the simulator

2. "experiment_settings": Configurations specific to the type of experiment (e.g. OBS_PERIOD, initial_seed, range of sweep values, treatments to run, plot labels, and color palettes).

3. "baseline_parameters": Unified flat dictionary containing baseline parameters (N, W, totalCapacity, etc.) shared by all policies.

