import numpy as np
import pandas as pd
from models import StrategicModel, VanillaModel, BayesianModel
from data import Dataset

# --- 3. Experiment Orchestration ---
class Experiment:
    def __init__(self, n_train, n_test, seed, optimizer_method='Nelder-Mead', optimizer_maxiter=500, optimizer_disp=False, parameter_mode='random', verbose=True):
        self.n_train = n_train; self.n_test = n_test; self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(self.seed)
        self.optimizer_method = optimizer_method; self.optimizer_maxiter = optimizer_maxiter; self.optimizer_disp = optimizer_disp
        self.dataset = Dataset(self.n_train, self.n_test, self.rng, parameter_mode, verbose=self.verbose)
        self.strategic_model = StrategicModel(optimizer_method, optimizer_maxiter, optimizer_disp)
        self.vanilla_model = VanillaModel(optimizer_method, optimizer_maxiter, optimizer_disp)
        self.bayesian_model = BayesianModel()
        self.bayesian_model.verbose = self.verbose # Pass verbose flag down

    def _print_data_parameters(self):
        print("\n--- Shared Data Generation Parameters (Seed: {}) ---".format(self.seed))
        probs_str = np.array2string(self.dataset.yz_probabilities, precision=4, suppress_small=True)
        print("P(Y,Z) for [(-1,-1), (-1,1), (1,-1), (1,1)]: {}".format(probs_str))
        print("Conditional Gaussian P(X | Y, Z):")
        for pair in self.dataset.yz_pairs:
            pair_tuple = tuple(pair)
            center, cov = self.dataset.centers[pair_tuple], self.dataset.covariances[pair_tuple]
            mean_str = np.array2string(center, precision=3, suppress_small=True, separator=',')
            cov_str = np.array2string(cov, prefix=' '*10, precision=3, suppress_small=True, separator=',').replace('\n', ' ').replace('  ', ' ')
            print(f"  (Y={pair[0]:>2}, Z={pair[1]:>2}) -> Mean: {mean_str}, Covariance: {cov_str}")

    def run_training_and_evaluation(self):
        if self.verbose: print(f"--- Generating Data for seed number {self.seed}---")
        self.dataset.generate_data()
        if self.verbose: self._print_data_parameters()

        if self.verbose: print("\n--- Training Models ---")
        self.strategic_model.train(self.dataset.X_train, self.dataset.Y_train, self.dataset.Z_train)
        self.vanilla_model.train(self.dataset.X_train, self.dataset.Y_train, self.dataset.Z_train)
        self.bayesian_model.learn_parameters_from_dataset(self.dataset)
        if self.verbose: print("Training complete.")

        if self.verbose: print("\n--- Evaluating Models ---")
        results = []
        for name, model in [('Strategic', self.strategic_model), ('Vanilla', self.vanilla_model)]:
            acc_ns = model.evaluate(self.dataset.X_test, self.dataset.Y_test, self.dataset.Z_test, strategic=False)
            acc_s = model.evaluate(self.dataset.X_test, self.dataset.Y_test, self.dataset.Z_test, strategic=True)
            results.append({'Model': name, 'Non-Strategic Acc': acc_ns, 'Strategic Acc': acc_s})

        acc_bayesian = self.bayesian_model.evaluate(self.dataset.X_test, self.dataset.Y_test, self.dataset.Z_test)
        results.append({'Model': 'Bayesian (Optimal)', 'Non-Strategic Acc': acc_bayesian, 'Strategic Acc': np.nan})

        results_df = pd.DataFrame(results).round(4)
        if self.verbose:
            print("\n--- Evaluation Results ---"); print(results_df.to_string(index=False))

        return results_df

def run_multiple_experiments(seeds, params):
    """
    Runs the experiment for a list of seeds and collects the key accuracy metrics and data parameters.
    """
    all_results = []
    all_data_params = []

    for seed in seeds:
        print(f"--- Running experiment for seed: {seed} ---")
        # Create a new experiment instance for each seed
        experiment = Experiment(
            n_train=params['n_train'],
            n_test=params['n_test'],
            seed=seed,
            optimizer_method=params['optimizer_method'],
            optimizer_maxiter=params['optimizer_maxiter'],
            optimizer_disp=params['optimizer_disp'],
            parameter_mode=params['parameter_mode'],
            verbose=False # Suppress detailed output for each run
        )

        # Generate data to get parameters
        experiment.dataset.generate_data()

        # Collect data parameters for this seed
        data_params_entry = {
            'seed': seed,
            'yz_probabilities': experiment.dataset.yz_probabilities.tolist(),
            'centers': {str(k): v.tolist() for k, v in experiment.dataset.centers.items()},
            'covariances': {str(k): v.tolist() for k, v in experiment.dataset.covariances.items()}
        }
        all_data_params.append(data_params_entry)

        # Run training and evaluation to get the results DataFrame for this seed
        results_df = experiment.run_training_and_evaluation()

        # Extract the 4 required values
        strat_row = results_df[results_df['Model'] == 'Strategic']
        vanilla_row = results_df[results_df['Model'] == 'Vanilla']

        result_entry = {
            'seed': seed,
            'Strategic_Non-Strategic_Acc': strat_row['Non-Strategic Acc'].iloc[0],
            'Strategic_Strategic_Acc': strat_row['Strategic Acc'].iloc[0],
            'Vanilla_Non-Strategic_Acc': vanilla_row['Non-Strategic Acc'].iloc[0],
            'Vanilla_Strategic_Acc': vanilla_row['Strategic Acc'].iloc[0],
        }
        all_results.append(result_entry)

    # Convert the lists of dictionaries to DataFrames
    final_results_df = pd.DataFrame(all_results)
    final_data_params_df = pd.DataFrame(all_data_params)

    return final_results_df, final_data_params_df
