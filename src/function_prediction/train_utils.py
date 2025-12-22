import math
import typing as t
from pathlib import Path

import pytorch_lightning as L
import torch
from lightning.pytorch.loggers import MLFlowLogger
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary

from src.function_prediction.uniprot_processing import (
    Ontology,
    one_hot_encode_go_labels,
)
import warnings
import numpy as np
import scipy.sparse as ssp
# from sklearn.metrics import average_precision_score as aupr
import math
import pandas as pd
from collections import OrderedDict,deque,Counter
import math
import re
import pickle as pkl
import os
import sys
import argparse

def encode_labels(
    ontology: Ontology, data_dict: t.Dict[str, t.Dict]
) -> t.Tuple[t.Dict[str, t.List[str]], t.Dict[str, torch.Tensor]]:
    """
    One-hot encode the labels for the given data dictionary.

    Parameters
    ----------
    ontology: Ontology
        Ontology object.
    data_dict: t.Dict[str, t.Dict]
        Dictionary containing the data dictionaries for each split (e.g., {"train": train_dict, "validation": valid_dict, "test": test_dict}).

    Returns
    -------
    uniprot_ids: t.Dict[str, t.List[str]]
        Dictionary containing the Uniprot ID for each data split ["train", "validation", "test"].
    encoded_labels: t.Dict[str, torch.Tensor]
        Dictionary containing the one-hot encoded labels for each data split ["train", "validation", "test"].
    """
    uniprot_ids = {}
    encoded_labels = {}
    for data_split in data_dict.keys():
        uniprot_id, one_hot_labels = one_hot_encode_go_labels(
            ontology.label_encoder,
            data_dict[data_split]["labels"],
            ontology.num_classes,
        )
        uniprot_ids[data_split] = uniprot_id
        encoded_labels[data_split] = one_hot_labels

    return uniprot_ids, encoded_labels


def log_model_summary(
    model: L.LightningModule, val_loader: DataLoader, mlf_logger: MLFlowLogger
) -> None:
    """
    Log the model summary using MLFlowLogger.

    Parameters
    ----------
    model : L.LightningModule
        The model to summarize.
    val_loader : DataLoader
        Validation DataLoader.
    mlf_logger : MLFlowLogger
        MLFlow logger object.
    """
    data_list = [val_loader.dataset[i] for i in range(2)]
    batch = Batch.from_data_list(data_list)
    summary_str = str(summary(model, batch))

    with open("model_summary.txt", "w") as f:
        f.write(summary_str)

    mlf_logger.experiment.log_artifact(mlf_logger.run_id, "model_summary.txt")
    Path("model_summary.txt").unlink()
    del data_list, batch


def benchmark(
    scores: torch.Tensor, targets: torch.Tensor, ontology: Ontology
) -> t.Dict[str, float]:
    """
    Calculate custom metrics including Fmax, Smin, AUPR, IC-AUPR, and DP-AUPR.

    Parameters
    ----------
    scores : torch.Tensor
        The predicted scores (probabilities) from the model.
    targets : torch.Tensor
        The ground truth labels.
    ontology : Ontology
        The ontology object used for IC and depth calculations.

    Returns
    -------
    metrics : t.Dict[str, float]
        Dictionary containing the calculated metrics.
    """
    targets = ssp.csr_matrix(targets.to("cpu").numpy())
    scores = scores.to("cpu").numpy()
    idx_goid = ontology.index_to_term
    go = ontology

    fmax_ = 0.0, 0.0, 0.0
    precisions = []
    recalls = []
    icprecisions = []
    icrecalls = []
    dpprecisions = []
    dprecalls = []
    goic_list = []
    godp_list = []
    
    for i in range(len(idx_goid)):
        goic_list.append(go.get_ic(idx_goid[i]))
        godp_list.append(go.get_icdepth(idx_goid[i]))
        
    goic_vector = np.array(goic_list).reshape(-1, 1)
    godp_vector = np.array(godp_list).reshape(-1, 1)
    
    for cut in (c / 100 for c in range(101)):
        cut_sc = ssp.csr_matrix((scores >= cut).astype(np.int32))
        correct = cut_sc.multiply(targets).sum(axis=1)
        correct_sc = cut_sc.multiply(targets)
        fp_sc = cut_sc - correct_sc
        fn_sc = targets - correct_sc
        
        correct_ic = ssp.csr_matrix(correct_sc.dot(goic_vector))
        cut_ic = ssp.csr_matrix(cut_sc.dot(goic_vector))
        targets_ic = ssp.csr_matrix(targets.dot(goic_vector))
        
        correct_dp = ssp.csr_matrix(correct_sc.dot(godp_vector))
        cut_dp = ssp.csr_matrix(cut_sc.dot(godp_vector))
        targets_dp = ssp.csr_matrix(targets.dot(godp_vector))
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p, r = correct / cut_sc.sum(axis=1), correct / targets.sum(axis=1)
            p, r = np.average(p[np.invert(np.isnan(p))]), np.average(r)
            
            mi = fp_sc.dot(goic_vector).sum(axis=0)
            ru = fn_sc.dot(goic_vector).sum(axis=0)
            mi /= len(targets.sum(axis=1))
            ru /= len(targets.sum(axis=1))
            
            icp, icr = correct_ic / cut_ic, correct_ic / targets_ic
            icp, icr = np.average(icp[np.invert(np.isnan(icp))]), np.average(icr)
            
            dpp, dpr = correct_dp / cut_dp, correct_dp / targets_dp
            dpp, dpr = np.average(dpp[np.invert(np.isnan(dpp))]), np.average(dpr)
            
        if np.isnan(p):
            precisions.append(0.0)
            recalls.append(r)
        else:
            precisions.append(p)
            recalls.append(r)
            
        if np.isnan(icp):
            icprecisions.append(0.0)
            icrecalls.append(icr)
        else:
            icprecisions.append(icp)
            icrecalls.append(icr)
            
        if np.isnan(dpp):
            dpprecisions.append(0.0)
            dprecalls.append(dpr)
        else:
            dpprecisions.append(dpp)
            dprecalls.append(dpr)
        
        try:
            fmax_ = max(fmax_, (2 * p * r / (p + r) if p + r > 0.0 else 0.0, math.sqrt(ru*ru + mi*mi), cut))
        except ZeroDivisionError:
            pass

    # Convert lists to numpy arrays after the loop
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    result_aupr = np.trapz(precisions, recalls)
    
    icprecisions = np.array(icprecisions)
    icrecalls = np.array(icrecalls)
    sorted_index = np.argsort(icrecalls)
    icrecalls = icrecalls[sorted_index]
    icprecisions = icprecisions[sorted_index]
    result_icaupr = np.trapz(icprecisions, icrecalls)
    
    dpprecisions = np.array(dpprecisions)
    dprecalls = np.array(dprecalls)
    sorted_index = np.argsort(dprecalls)
    dprecalls = dprecalls[sorted_index]
    dpprecisions = dpprecisions[sorted_index]
    result_dpaupr = np.trapz(dpprecisions, dprecalls)
        
    return {
        "Fmax": fmax_[0],
        "Smin": fmax_[1],
        "AUPR": result_aupr,
        "IC-AUPR": result_icaupr,
        "DP-AUPR": result_dpaupr,
        "threshold": fmax_[2],
    }

