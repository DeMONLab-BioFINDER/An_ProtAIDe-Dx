import os 
import copy
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openTSNE import TSNE
from src.utils.io import load_pkl
from src.ProtAIDeDx.misc.nn_helper import load_ProbaThresholds


def gen_predicted_proba_df(results_dir,
                           splits_dir,
                           probaThresholds_path,
                           nb_folds=10,
                           targets=['CU', 'AD', 'PD', 'FTD', 'ALS', 'StrokeTIA']):
    """
    _summary_

    Args:
        results_dir (_type_): _description_
        splits_dir (_type_): _description_
        probaThresholds_path (_type_): _description_
        nb_folds (int, optional): _description_. Defaults to 10.
        targets (list, optional): _description_. Defaults to ['CU', 'AD', 'PD', 'FTD', 'ALS', 'StrokeTIA'].

    Returns:
        _type_: _description_
    """
    gt_targets = [t + '-GT' for t in targets]
    prob_targets = [t + '-PredProb' for t in targets]

    cols2get = ['PersonGroup_ID', 
                'Contributor_Code',
                'Recruited Control'] + gt_targets  + prob_targets
    
    for fold in range(nb_folds):
        fold_dir = 'fold_' + str(fold)
        test_results_df = pd.read_csv(
            os.path.join(results_dir, fold_dir, 'test_results.csv'))
        threshold_list = load_ProbaThresholds(probaThresholds_path,
                                              'CV',
                                              fold_dir)
        threshold_list = np.array(threshold_list, dtype=np.float32)

        for i, prob_t in enumerate(prob_targets):
            test_results_df[prob_t] = (test_results_df[prob_t] / float(threshold_list[i]))
        
        # append Recruited Control column from data
        data_df = pd.read_csv(
            os.path.join(splits_dir, fold_dir, 'test.csv'),
            usecols=['PersonGroup_ID', 'Contributor_Code', 'Recruited Control'])
    
        test_results_df = test_results_df.merge(
            data_df[['PersonGroup_ID', 'Contributor_Code', 'Recruited Control']],
            on=['PersonGroup_ID', 'Contributor_Code'],
            how='left'
        )
        
        if fold == 0:
            df = copy.deepcopy(test_results_df)[cols2get]
        else:
            df = pd.concat([df, test_results_df], axis=0)[cols2get]
        
    df.reset_index(inplace=True, drop=True)

    return df 


def get_single_pos_subjects(proba_df,
                            targets=['CU', 'AD', 'PD', 'FTD', 'ALS', 'StrokeTIA']):
    """
    _summary_

    Args:
        proba_df (_type_): _description_
        targets (list, optional): _description_. Defaults to ['CU', 'AD', 'PD', 'FTD', 'ALS', 'StrokeTIA'].
    """
    prob_targets = [t + '-PredProb' for t in targets]
    prob_arr = proba_df[prob_targets].values

    selected_probaArr_list = []
    selected_DX_list = []
    for i in range(prob_arr.shape[0]):
        dx_values = proba_df.iloc[i][['Recruited Control', 
                                      'AD-GT', 
                                      'PD-GT', 
                                      'FTD-GT', 
                                      'ALS-GT', 
                                      'StrokeTIA-GT']]
        if np.nansum(dx_values) == 1:
            dx_label = targets[np.where(dx_values == 1)[0][0]]
            selected_DX_list.append(dx_label)
            selected_probaArr_list.append(prob_arr[i, :])
    
    return np.vstack(selected_probaArr_list), selected_DX_list


def tSNE_fit(selected_ProbaArr, 
             dim2proj=2,
             perplexity=1000):
    """
    _summary_

    Args:
        selected_ProbaArr (_type_): _description_
        dim2proj (int, optional): _description_. Defaults to 2.
        perplexity (int, optional): _description_. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    fitted_tSNE = TSNE(
        n_components=dim2proj,
        perplexity=perplexity,
        metric="euclidean",
        n_jobs=1,
        random_state=42,
        verbose=False,
    )
    fitted_tSNE = fitted_tSNE.fit(selected_ProbaArr)
    tsneArr = fitted_tSNE.transform(selected_ProbaArr)

    return fitted_tSNE, tsneArr


def tSNE_infer(fitted_tSNE, probaArr):
    """
    _summary_

    Args:
        fitted_tSNE (_type_): _description_
        probaArr (_type_): _description_

    Returns:
        _type_: _description_
    """
    return  fitted_tSNE.transform(probaArr)


def tSNE_scatter_plot(tsneArr,
                      DX_list,
                      fig_save_path,
                      fig_xlabel='tSNE x',
                      fig_ylabel='tSNE y',
                      fig_title='Individual probability map'):
    """
    _summary_

    Args:
        tsneArr (_type_): _description_
        DX_list (_type_): _description_
        fig_save_path (_type_): _description_
        fig_xlabel (str, optional): _description_. Defaults to 'tSNE x'.
        fig_ylabel (str, optional): _description_. Defaults to 'tSNE y'.
        fig_title (str, optional): _description_. Defaults to 'Individual probability map'.
    """
    plot_df = pd.DataFrame(tsneArr, 
                           columns=[fig_xlabel, fig_ylabel])
    plot_df['Label'] = DX_list

    plt.figure(figsize=(8, 8))
    color_mapper = {
        'CU': 'blue',
        'AD': 'red',
        'PD': 'green',
        'FTD': 'orange',
        'ALS': 'purple',
        'StrokeTIA': 'black'
    }

    ax = sns.scatterplot(
        data=plot_df, 
        x=fig_xlabel, 
        y=fig_ylabel, 
        hue='Label', 
        palette=color_mapper, 
        s=10)
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    
    # Add labels, and title
    plt.title(fig_title, fontsize=16)
    plt.xlabel(fig_xlabel, fontsize=14)
    plt.ylabel(fig_ylabel, fontsize=14)
    
    plt.tick_params(axis='x', length=7, width=4)
    plt.tick_params(axis='y', length=7, width=4)
    plt.xlim(-12, 12)
    plt.ylim(-12, 12)
    y_ticks_pos = [-10, -5, 0, 5, 10]
    x_ticks_pos = [-10, -5, 0, 5, 10]
    plt.xticks(x_ticks_pos, [])
    plt.yticks(y_ticks_pos, [])

    plt.tight_layout()
    if fig_save_path is None:
        plt.show()
    else:
        plt.savefig(fig_save_path, 
                    dpi=400)
        plt.cla()
        plt.close() 


def tSNE_contour_overlay_plot():
    pass 



def save_tSNE(fitted_tSNE,
              tsneArr,
              DX_list,
              tSNE_save_path):
    """
    _summary_

    Args:
        fitted_tSNE (_type_): _description_
        tsneArr (_type_): _description_
        DX_list (_type_): _description_
        tSNE_save_path (_type_): _description_
    """
    dict2save = {
        "tSNE": fitted_tSNE,
        "tsneArr": tsneArr,
        "DXList": DX_list
    }

    joblib.dump(dict2save,
                tSNE_save_path,
                compress=3)


def load_tSNE(tSNE_save_path):
    """
    _summary_

    Args:
        tSNE_save_path (_type_): _description_

    Returns:
        _type_: _description_
    """

    dict2save = joblib.load(tSNE_save_path)

    return dict2save['tSNE'], dict2save['tsneArr'], dict2save['DXList']


def args_parser():
    """
    _summary_

    Returns:
        _type_: _description_
    """
    
    parser = argparse.ArgumentParser(prog='Args')
    parser.add_argument('--splits_dir', type=str, default='/')
    parser.add_argument('--ProtAIDeDx_results_dir', type=str, default='/')
    parser.add_argument('--output_dir', type=str, default='/')
    parser.add_argument('--ref_dir', type=str, default='/')
    parser.add_argument('--probaThresholds_path', type=str, default='/')
    parser.add_argument('--targets', 
                        type=list,
                        default=['CU', 'AD', 'PD', 'FTD', 'ALS', 'StrokeTIA'])
    parser.add_argument('--nb_folds', type=int, default=10)

    parser.add_argument('--dim2proj', type=int, default=2)
    parser.add_argument('--perplexity', type=int, default=1000)

    tsne_args, _ = parser.parse_known_args()
    return tsne_args


def main(args):
    """
    _summary_

    Args:
        args (_type_): _description_
    """
    proba_df = gen_predicted_proba_df(
        args.ProtAIDeDx_results_dir,
        args.splits_dir,
        args.probaThresholds_path,
        args.nb_folds,
        args.targets
    )
    selected_ProbaArr, selected_DX_list = get_single_pos_subjects(
        proba_df, args.targets
    )

    # check whether input is replicated 
    tSNE_ref_pkl = load_pkl(
        os.path.join(args.ref_dir, 'Fig2_GNPC_ref.pkl')
    )
    assert np.allclose(tSNE_ref_pkl['input'],
                       selected_ProbaArr), "Input Replication Failure!"
    
    # since tSNE is a stochastic algorithm with multiple iterations
    # a very minor difference (e.g., 1e-10) could cause amplified difference
    # even our probabilities are very similar by np.allclose(),
    # there might be very tiny (<1e-6) differences
    # therefore, we used original input array to make sure results are same in paper
    fitted_tSNE, tsneArr = tSNE_fit(
        tSNE_ref_pkl['input'], 
        args.dim2proj,
        args.perplexity
    )

    tSNE_scatter_plot(
        tsneArr, selected_DX_list,
        os.path.join(args.output_dir, 'Fig2_tSNE.png')
    )

    save_tSNE(fitted_tSNE,
              tsneArr,
              selected_DX_list,
              os.path.join(args.output_dir, 'tSNE.pkl'))
    
    # check whether results are replicated 
    assert np.allclose(tSNE_ref_pkl['output'],
                       tsneArr), "Replication Failed!"
    
    print("Congrats! You have replicated Fig2: tSNE")


if __name__ == '__main__':
    main(args_parser())