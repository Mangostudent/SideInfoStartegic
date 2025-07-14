import matplotlib.pyplot as plt
import pandas as pd

def plot_results(df):
    """
    Generates a scatter plot of the results from the multiple experiments.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each of the four metrics
    ax.scatter(df['seed'], df['Strategic_Non-Strategic_Acc'], label='Strategic Model (Non-Strategic Eval)', marker='o', s=80)
    ax.scatter(df['seed'], df['Strategic_Strategic_Acc'], label='Strategic Model (Strategic Eval)', marker='^', s=80)
    ax.scatter(df['seed'], df['Vanilla_Non-Strategic_Acc'], label='Vanilla Model (Non-Strategic Eval)', marker='s', s=80)
    ax.scatter(df['seed'], df['Vanilla_Strategic_Acc'], label='Vanilla Model (Strategic Eval)', marker='x', s=80, c='red')

    ax.set_title('Model Accuracy Across Different Random Seeds', fontsize=16)
    ax.set_xlabel('Random Seed', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_xticks(df['seed'])
    ax.legend(fontsize=11)
    # Only use numeric columns for y-axis limits
    numeric_cols = ['Strategic_Non-Strategic_Acc', 'Strategic_Strategic_Acc', 'Vanilla_Non-Strategic_Acc', 'Vanilla_Strategic_Acc']
    ax.set_ylim(bottom=df[numeric_cols].min().min() - 0.05, top=1.0)

    plt.tight_layout()
    plt.show()
