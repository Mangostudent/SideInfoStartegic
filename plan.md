## Workflow
1.  You (the user) provide a coding problem or task.
.  I (GitHub Copilot) will interpret your problem statement and write a clear, coherent summary of my understanding below it.
3.  We discuss and refine the solution together. Once agreed, I will write a clear, structured plan for the proposed solution below the discussion.
4.  I implement the code, distributing logic across multiple files for modularity and readability.
5.  You review the code and provide feedback or approval.
6.  We iterate as needed until the solution is complete.



## Problem Statement

*(Below this line, you will describe the specific problem and your understanding of it. I will then write a coherent summary of my understanding below your description.)*

The user wants to refactor the existing Jupyter Notebook (`sideinfo_strategic.ipynb`) into a multi-file Python project. The goal is to improve code organization and maintainability by separating different components (models, data handling, experiments, utilities, and main execution logic) into individual files. Additionally, the user wants to enhance the analysis output to make it easier to understand the results for individual experiment runs, including displaying the parameter values and specific accuracy metrics for each seed, in addition to the aggregated results.

## Proposed Solution

*(Below this line, we will discuss and refine the solution. Once finalized, I will write a clear, structured plan for the solution here before coding begins.)*

The plan is to refactor the `sideinfo_strategic.ipynb` notebook into the following Python files:

1.  `models.py`: Contains the `BaseModel`, `StrategicModel`, `VanillaModel`, and `BayesianModel` classes.
2.  `data.py`: Contains the `Dataset` class.
3.  `experiment.py`: Contains the `Experiment` class and the `run_multiple_experiments` function.
4.  `utils.py`: Contains the `plot_results` function.
5.  `main.py`: Contains the main execution logic, including the control panel for parameters, running the batch experiments, printing aggregated results, and generating plots.

Improvements for analysis will include:
-   Ensuring necessary imports are added to all new files.
-   Modifying `main.py` to include a section that runs a single experiment with a specified seed and enables verbose output (`verbose=True`) for detailed analysis of that specific run, showing data parameters and evaluation results.
-   The existing batch run in `main.py` will continue to generate and print the aggregated results table for all seeds.
-   The plotting function will be updated to be imported and used from `utils.py`.
