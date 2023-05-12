

## TransOpt

This repository contains the code for the TransOpt model for generating embeddings of single-objective continuous optimization problems.
The transformer model is trained to classify the problem instances from the BBOB benchmark suite into the 24 problem classes.
The repository contains code for all of the experiments presented in the paper. 

### Transformer training

- The script static_problem_classification_parameters_exp.py contains the code for training the model with different model parameters.
- The script static_problem_classification_parameters_aggregations.py contains the code for training the model with different aggregation functions.
- The script static_problem_classification_parameters_downstream.py can be used to run the model with the hyperparameters which proofed to work well across all problem dimensions.

All of the previsously mentioned scripts take as input three arguments: 
- dimension - The dimension of the problems for which to run the problem classification task
- sample_count_dimension_factor - Defines the number of samples used for each problem instance. For each problem instance, sample_count_dimension_factor * problem_dimension samples 
instances_to_use - The number of problem instances to use to train the model

For the script to be run, the appropriate sample data should already be existing in the 'data' folder. The sample data can be provided in a file with the name 'data/lhs_samples_dimension_{dimension}_{sample_count_dimension_factor}_samples.p' (if the objective function values are not scaled, in case the script will scale them appropriately) or 'scaled_lhs_samples_dimension_{dimension}_{sample_count_dimension_factor}_samples.csv' (if the objective function values are scaled).


- The file utils_runner_universal.py contains the UniversalRunner class, which contains the main logic for running the experiments. The UniversalRunner class has several constructor parameters:

        task_name : str
        The name of the task to run the model on, such as 'problem_classification' or 'algorithm_classification'.
        extra_info : str
        An optional string to add extra information to the setting name and result directory.
        verbose : bool
        A flag to indicate whether to print verbose messages or not.
        lr_max : float
        The maximum learning rate to use for training the model.
        plot_training : bool
        A flag to indicate whether to plot the training metrics or not.
        n_heads : int
        The number of attention heads in the transformer model.
        n_layers : int
        The number of encoder layers in the transformer model.
        d_model : int
        The dimension of the input and output vectors in the transformer model.
        d_k : int
        The dimension of the query and key vectors in the attention mechanism.
        d_v : int
        The dimension of the value vector in the attention mechanism.
        n_epochs : int
        The number of epochs to train the model.
        batch_size : int
        The batch size for the data loaders.
        fold : int or None
        The fold number to use for cross-validation or None for no cross-validation.
        split_ids_dir : str or None
        The path to the directory where the split ids are stored or None for random splits.
        global_result_dir : str
        The path to the directory where all results are stored.
        aggregations : list of str or None
        The list of aggregation functions to use for aggregating the transformer embeddings. Can contain a list of the following aggregations "min", "max", "mean" or "std"


- The file model_stats.py contains the transformer model

### Comparison to ELA features

The file feature_calculation_per_problem.R is used for calculating the ELA features. It receives an input arguments the name of a file containing samples of optimization problem instances and the problem_id for which the ELA features should be calculated. It produces a single file containing the ELA features for a single problem id, and should be run for each problem id separately.

The file ela_static_problem_classification.py trains a Random Forest model based on the ELA features. Before running this file, the ELA features produced by the feature_calculation_per_problem.R should be merged into a single file.

The remaining notebooks are used for generating the visualizations for the paper.





