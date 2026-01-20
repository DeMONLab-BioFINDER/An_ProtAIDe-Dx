#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os 
import argparse
import pandas as pd 


def append_columns(input_df, 
                   src_df,
                   cols_to_append,
                   keys):
    """
    Append specified columns from source DataFrame to input DataFrame .

    Args:
        input_df (pd.DataFrame): 
            Input DataFrame to which columns will be appended
        src_df (pd.DataFrame): 
            Source DataFrame from which columns will be taken
        cols_to_append (list): 
            List of column names to append from source DataFrame
        keys (list): 
            List of column names to use as keys for merging
    Returns:
        pd.DataFrame: Merged DataFrame with appended columns
    """
    merged_df = pd.merge(input_df, 
                         src_df[keys + cols_to_append], 
                         on=keys, 
                         how='left').reset_index(drop=True)
    return merged_df


def args_parser():
    """
    Parse command-line arguments for appending columns to a CSV file.
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(prog='Args')
    parser.add_argument('--input_csv_path', type=str, default='/')
    parser.add_argument('--src_csv_path', type=str, default='/')
    parser.add_argument('--output_csv_path', type=str, default='/')
    parser.add_argument('--cols_to_append', type=list, 
                        default=[
                                'Age_at_Visit',
                                'Sex',
                                'APOE',
                                'Years_of_Education',
                                'cognitive_status_baseline_variable',
                                'diagnosis_baseline_variable',
                                'DxGroup',
                                'Normal Control',
                                'AD',
                                'LBD',
                                'NonAD MCI',
                                'FTD Spectrum',
                                '4R Tauopathies',
                                'StrokeTIA',
                                'MMSE',
                                'UPDRS',
                                'Plasma_pTau217',
                                'Plasma_NFL',
                                'TauPET_MetaROI',
                                'MRI_CTADSign',
                                'MRI_WholeBrainCT',
                                'MRI_VentricleVol',
                                'MRI_WMH',
                                'CSF_Ab42/40',
                                'CSF_pTau217',
                                'CSF_SAA',
                                'CSF_GFAP',
                                'CSF_NFL',
                                'CSF_YKL40',
                                'CSF_sTREM2',
                                'CSF_SYT1',
                                'CSF_SNAP25',
                                'CSF_NPTX2',
                                'CSF_PDGFRB',
                                'CSF_S100'])
    parser.add_argument('--keys', type=list, 
                        default=['PersonGroup_ID', 
                                 'Visit', 'Contributor_Code'])
    
    args, _ = parser.parse_known_args()
    return args


def main(args):
    """
    Main function to append specified columns to an input CSV.

    Args:
        args (argparse.Namespace): Parsed command-line arguments
    """
    input_df = pd.read_csv(args.input_csv_path, low_memory=False)
    src_df = pd.read_csv(args.src_csv_path, low_memory=False)

    output_df = append_columns(input_df,
                               src_df,
                               args.cols_to_append,
                               args.keys)
    
    assert output_df.shape[1] == input_df.shape[1] + len(
        args.cols_to_append), \
        "Columns not appended correctly!"

    os.makedirs(os.path.dirname(args.output_csv_path), exist_ok=True)
    output_df.to_csv(args.output_csv_path, index=False)


if __name__ == "__main__":
    main(args_parser())