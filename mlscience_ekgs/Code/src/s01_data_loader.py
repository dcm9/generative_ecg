from pathlib import Path

import ast
import numpy as np
import pandas as pd
import tqdm
import wfdb

from mlscience_ekgs.settings import ptb_path, ptb_xl_path, mimic_ecg_path
from mlscience_ekgs.Code.src.s07_segment_ecg import segment_and_filter_ecg


CHANNELS = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6',]

def load_ptb_data(data_idx=0, starting_idx=0, length=10_000, channels=CHANNELS,
                  ecg_filepath=None):
    """Load 12-lead ECG data from the PTB database.
    
    Args:
        data_idx: index of the data file to load.
        starting_idx: index of the starting sample.
        length: number of samples to load.
        channels: list of channel names to load.
        ecg_filepath: path to the file to load. If None, the path is constructed
            from the data index.
        
    Returns:
        channel_data: 12-lead ECG data.
    """
    if ecg_filepath is None:
        with open(Path(ptb_path, "RECORDS")) as fp:
            data_list = fp.readlines()
        curr_data = data_list[data_idx].strip()
        ecg_filepath = Path(ptb_path, curr_data)
    channel_data, _ = wfdb.rdsamp(ecg_filepath, channel_names=channels)
    channel_data = channel_data.T
    channel_data = channel_data[:, starting_idx:starting_idx+length]
    
    return channel_data, channels


def _create_mimic_filepath(subject_id, study_id):
    subject_id, study_id = str(subject_id), str(study_id)
    patient_id_prefix = subject_id[:4]
    study_filepath = Path(
        mimic_ecg_path, "files", f"p{patient_id_prefix}",
        f"p{subject_id}", f"s{study_id}", study_id
    )
    
    return study_filepath
    

def load_mimic_data(subject_id=None, study_id=None, length=10_000, 
                    ecg_filepath=None):
    """Load 12-lead ECG data from the MIMIC database.

    Args:
        subject_id (str): the subject ID.
        study_id (str): the study ID.
        length (int): the number of samples to load.
        ecg_filepath (str): the path to the file to load. If None, the path is
            constructed from the subject ID and study ID.

    Returns:
        channel_data: 12-lead ECG data.
        channels: the channel names.
    """
    if ecg_filepath is None:
        ecg_filepath = _create_mimic_filepath(subject_id, study_id)
    channel_data, metadata = wfdb.rdsamp(ecg_filepath)
    channel_data = channel_data.T
    channel_data = channel_data[:, :length]
    channels = metadata["sig_name"]
    
    return channel_data, channels


def load_ptb_xl_dataset(sampling_rate=500, test_fold=10,
                        segmentation=False, target="age"):
    assert sampling_rate in [100, 500]
    if segmentation:
        X_name = "X_seg.npy" 
        y_name = f"y_{target}_seg.npy"
    else: 
        X_name = "X.npy"
        y_name = f"y_{target}.npy"
    
    curr_path = Path(ptb_xl_path, f"records{sampling_rate}")
    X_path = Path(curr_path, X_name)
    y_path = Path(curr_path, y_name)
    if X_path.exists() and y_path.exists():
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        df_annot = pd.read_csv(
            Path(ptb_xl_path, "ptbxl_database.csv"), 
            index_col="ecg_id"
        )
        df_annot.scp_codes = df_annot.scp_codes.apply(ast.literal_eval)
        if sampling_rate == 100:
            data = [wfdb.rdsamp(Path(ptb_xl_path, f)) 
                    for f in df_annot.filename_lr]
        else:
            data = [wfdb.rdsamp(Path(ptb_xl_path, f)) 
                    for f in df_annot.filename_hr]
        X = np.array([signal for signal, _ in data])
        y = df_annot[target].values
        X = X[np.where(df_annot.strat_fold != test_fold)]
        y = y[np.where(df_annot.strat_fold != test_fold)]
        X = X.transpose(0, 2, 1)
        if segmentation:
            beats, targets = [], []
            for i, signal in enumerate(tqdm.tqdm(X)):
                try:
                    curr_beats = segment_and_filter_ecg(signal, sampling_rate)
                    beats.extend(curr_beats)
                    targets.extend([y[i]] * len(curr_beats))
                except:
                    continue
            X = np.array(beats)
            y = np.array(targets)
        np.save(X_path, X)
        np.save(y_path, y)
    
    return X, y