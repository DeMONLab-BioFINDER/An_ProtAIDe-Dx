#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os 
import numpy as np 
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import label_binarize 
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def bootstrap_BCA(y_true,
                  y_pred,
                  nb_classes=4,
                  nb_bootstrap=1000,
                  random_seed=42):
    """
    Bootstrap balanced classification accuracy (BCA).

    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        nb_classes (int, optional): 
            Number of classes. Defaults to 4.
        nb_bootstrap (int, optional): 
            Number of bootstrap samples. Defaults to 1000.
        random_seed (int, optional): 
            Random seed for reproducibility. Defaults to 42.
    Returns:
        np.ndarray: Array of bootstrapped balanced classification accuracies
    """
    np.random.seed(random_seed)
    bootstrapped_BCAs = np.zeros((nb_bootstrap, nb_classes))

    # one-hot encode the true labels & predicted labels
    y_true_onehot = label_binarize(y_true, classes=np.arange(nb_classes))
    y_pred_onehot = label_binarize(y_pred, classes=np.arange(nb_classes))

    for i in range(nb_bootstrap):
        # resample with replacement
        indices = resample(np.arange(len(y_true)), 
                           replace=True, 
                           random_state=random_seed + i)
        y_true_boot, y_pred_boot = \
            y_true_onehot[indices], y_pred_onehot[indices]
        
        for j in range(nb_classes):
            bootstrapped_BCAs[i, j] = balanced_accuracy_score(
                y_true_boot[:, j], y_pred_boot[:, j])
        
    return bootstrapped_BCAs


def get_bootstrap_across_models(results_dir):
    """
    Get bootstrapped balanced classification accuracies (BCAs) across models.

    Args:
        results_dir (str): Directory containing model prediction CSV files

    Returns:
        pd.DataFrame: 
        DataFrame containing bootstrapped BCAs for each model and target
    """
    models = ['M0', 'M1', 'M2', 'M3']
    model_mapper = {'M0': 'Model 0',
                    'M1': 'Model 1',
                    'M2': 'Model 2',
                    'M3': 'Model 3'}
    target_list = ['AD', 'PD', 'FTD', 'Stroke']

    rows = []
    for model in models:
        model_df = pd.read_csv(
            os.path.join(results_dir, f'{model}_pred.csv'))
        
        model_bootstrapped_BCAs = bootstrap_BCA(
            model_df['true_label'].values, 
            model_df['predicted_label'].values)
        
        for j in range(len(target_list)):
            for k in range(1000):
                row = [
                    model_mapper[model],
                    target_list[j],
                    model_bootstrapped_BCAs[k, j]
                ]
                rows.append(row)
    bootstrap_df = pd.DataFrame(rows, columns=['Model', 'Target', 'BCA'])
    return bootstrap_df


def plot_bootstrap_pointplot(ax, df, vline=0.7):
    """
    Plot bootstrapped BCAs as a point plot.

    Args:
        ax (matplotlib.axes.Axes): Axes object to plot on
        df (pd.DataFrame): DataFrame containing bootstrapped BCAs
        vline (float, optional): Vertical line position. Defaults to 0.7.
    """
    sns.pointplot(
        data=df, 
        x="BCA", 
        y="Model", 
        hue="Target",
        dodge=0.4, 
        markers="o",
        palette={
            'AD': 'red',
            'PD': 'green',
            'FTD': 'orange',
            'Stroke': 'black',
            'Overall': 'darkviolet'
        },
        errorbar="sd",
        capsize=0.1,
        markersize=4,
        linestyles="none",
        ax=ax
    )

    ax.axvline(x=vline, color='gray', linestyle='--', linewidth=2)

    ax.set_title("")
    ax.set_xlabel("Bootstrapped BCA", fontsize=20)
    ax.set_ylabel("")
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("Diseases")
        leg.get_frame().set_linewidth(1)

    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)

    ax.tick_params(axis='x', length=5, width=3)
    ax.tick_params(axis='y', length=5, width=3)

    ax.set_xlim(0.4, 1.0)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xticklabels([str(tick) for tick in [0.5, 0.6, 0.7, 0.8, 0.9]], 
                       fontsize=16)

    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(['Model 0', 'Model 1', 'Model 2', 'Model 3'],
                       fontsize=16)

def plot_confusion_heatmap(ax, 
                           cm, 
                           annotations, 
                           labels,
                           model):
    """
    Plot a confusion matrix heatmap.

    Args:
        ax (matplotlib.axes.Axes): Axes object to plot on
        cm (np.ndarray): Confusion matrix array
        annotations (np.ndarray): Annotations for the heatmap
        labels (list): List of label names
        model (str): Model name
    """
    sns.heatmap(
        cm, 
        annot=annotations, 
        fmt="", 
        cmap="Greens",
        xticklabels=labels, 
        yticklabels=labels,
        square=True,
        vmin=0, vmax=100,
        cbar=False, linewidths=0.5,
        annot_kws={"size": 20},
        ax=ax
    )

    for text in ax.texts:
        x, y = text.get_position()
        col = int(round(x - 0.5))
        row = int(round(y - 0.5))
        if row == 0 and col == 0:
            text.set_color("white")
        else:
            text.set_color("black")

    # Thick internal grid lines
    for r in range(1, cm.shape[0]):
        ax.hlines(r, xmin=0, xmax=cm.shape[1], colors='black', linewidth=2)
    for c in range(1, cm.shape[1]):
        ax.vlines(c, ymin=0, ymax=cm.shape[0], colors='black', linewidth=2)

    # Outer box
    heatmap_box = Rectangle(
        (0, 0), cm.shape[1], cm.shape[0],
        fill=False, edgecolor='black', linewidth=3, clip_on=False
    )
    ax.add_patch(heatmap_box)

    ax.tick_params(axis='x', length=5, width=3)
    ax.tick_params(axis='y', length=5, width=3)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)

    ax.set_title(f"Confusion Matrix: {model}", fontsize=20)
    ax.set_xlabel("Predicted Label", fontsize=20)
    ax.set_ylabel("True Label", fontsize=20)



def compute_confusion_matrix(results_dir, model):
    """
    Compute confusion matrix and annotations for a given model.

    Args:
        results_dir (str): Directory containing model prediction CSV files
        model (str): Model name
    Returns:
        tuple: Confusion matrix and annotations
    """
    model_df = pd.read_csv(
        os.path.join(results_dir, f'{model}_pred.csv'))
    
    labels = sorted(model_df["true_label"].unique())
    cm = confusion_matrix(model_df["true_label"], 
                          model_df["predicted_label"], 
                          labels=labels)
    
    cm_percent = cm.astype(np.float64)
    cm_percent = cm_percent / cm_percent.sum(axis=1, keepdims=True) * 100
    annotations = np.array(
        [["{:.1f}%".format(value) for value in row] for row in cm_percent], 
        dtype=object)
    
    return cm, annotations 


def wrapper(results_dir):
    """
    Wrapper function for plotting bootstrapped BCAs and confusion matrices.

    Args:
        results_dir (str): Directory containing model prediction CSV files
    """
    # compute data for plotting 
    bootstrap_df = get_bootstrap_across_models(results_dir)
    model_mapper = {'M0': 'Model 0',
                    'M1': 'Model 1',
                    'M2': 'Model 2',
                    'M3': 'Model 3'}
    labels = ['AD', 'PD', 'FTD', 'Stroke']

    CM_model2, annotations_model2 = compute_confusion_matrix(results_dir, 'M2')
    CM_model3, annotations_model3 = compute_confusion_matrix(results_dir, 'M3')


    _, axes = plt.subplots(
        1, 3, figsize=(15, 8),
        gridspec_kw={"width_ratios": [0.8, 1.0, 1.0]}
    )

    plot_bootstrap_pointplot(axes[0], bootstrap_df)
    plot_confusion_heatmap(
        axes[1], CM_model2, annotations_model2, labels, model_mapper['M2'])
    plot_confusion_heatmap(
        axes[2], CM_model3, annotations_model3, labels, model_mapper['M3'])

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'Fig5_BF2_DiffDX.png'), dpi=400)
    plt.close()


if __name__ == '__main__':
    wrapper(
        results_dir='./results/replica/Fig5_BF2/DifferentialDiagnosis'
    )