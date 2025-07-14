import pandas as pd
from experiment import run_multiple_experiments
from utils import plot_results

if __name__ == '__main__':
    # --- Control Panel ---
    PARAMETER_MODE = 'random' # 'manual' or 'random'
    N_TRAIN = 5000
    N_TEST = 10000
    OPTIMIZER_METHOD = 'Nelder-Mead'
    OPTIMIZER_MAXITER = 2000
    OPTIMIZER_DISP = False

    # List of seeds to iterate over for batch run
    SEEDS_TO_RUN_BATCH = list(range(1, 21))

    # Package parameters into a dictionary
    exp_params = {
        'n_train': N_TRAIN,
        'n_test': N_TEST,
        'optimizer_method': OPTIMIZER_METHOD,
        'optimizer_maxiter': OPTIMIZER_MAXITER,
        'optimizer_disp': OPTIMIZER_DISP,
        'parameter_mode': PARAMETER_MODE
    }

    # --- Run the batch experiment and get the numerical results and data parameters ---
    print("\n\n" + "="*80)
    print(" " * 20 + "DETAILED RESULTS FOR EACH SEED")
    print("="*80)
    results_table, data_params_table = run_multiple_experiments(SEEDS_TO_RUN_BATCH, exp_params)

    pd.set_option('display.width', 120)
    pd.set_option('display.float_format', '{:.4f}'.format)

    # Print parameter values and accuracy results for each seed
    for idx, row in data_params_table.iterrows():
        seed = row['seed']
        print(f"\n{'='*40}\nSeed: {seed}")
        # yz_probabilities table
        print("\n[YZ Probabilities]")
        yz_df = pd.DataFrame({'Pair': ['(-1,-1)', '(-1,1)', '(1,-1)', '(1,1)'], 'Probability': row['yz_probabilities']})
        print(yz_df.to_string(index=False))
        # centers table
        print("\n[Centers]")
        centers_df = pd.DataFrame([(k, v[0], v[1]) for k, v in row['centers'].items()], columns=['Pair', 'Center X', 'Center Y'])
        print(centers_df.to_string(index=False))
        # covariances table
        print("\n[Covariances]")
        cov_rows = []
        for k, v in row['covariances'].items():
            cov_rows.append([k, v[0][0], v[0][1], v[1][0], v[1][1]])
        cov_df = pd.DataFrame(cov_rows, columns=['Pair', 'Cov[0,0]', 'Cov[0,1]', 'Cov[1,0]', 'Cov[1,1]'])
        print(cov_df.to_string(index=False))
        # accuracy table
        acc_row = results_table[results_table['seed'] == seed].iloc[0]
        acc_df = pd.DataFrame({
            'Accuracy Type': ['Strategic_Non-Strategic', 'Strategic_Strategic', 'Vanilla_Non-Strategic', 'Vanilla_Strategic'],
            'Value': [
                acc_row['Strategic_Non-Strategic_Acc'],
                acc_row['Strategic_Strategic_Acc'],
                acc_row['Vanilla_Non-Strategic_Acc'],
                acc_row['Vanilla_Strategic_Acc']
            ]
        })
        print("\n[Accuracy Results]")
        print(acc_df.to_string(index=False))
        print("="*40)

    # Add ConditionMet column with tick/cross emoji
    def calc_condition(row):
        if (
            row['Strategic_Non-Strategic_Acc'] < row['Vanilla_Non-Strategic_Acc'] and
            row['Strategic_Strategic_Acc'] > row['Vanilla_Strategic_Acc']
        ):
            return '✅'
        else:
            return '❌'
    results_table['ConditionMet'] = results_table.apply(calc_condition, axis=1)

    print("\n\n" + "="*80)
    print(" " * 20 + "AGGREGATED ACCURACY RESULTS ACROSS ALL SEEDS")
    print("="*80)
    print(results_table.set_index('seed'))
    print("="*80)

    # --- (Optional) Visualize the results as a scatter plot ---
    print("\nGenerating scatter plot of the results...")
    plot_results(results_table)
