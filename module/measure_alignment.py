import os
import argparse 

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

import module.metrics
from module.tasks import get_models
import module.utils
from pprint import pprint


import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def prepare_features(feats, q=0.95, exact=False):
    """
    Prepare features by removing outliers and normalizing
    Args:
        feats: a torch tensor of any share
        q: the quantile to remove outliers
    Returns:
        feats: a torch tensor of the same shape as the input
    """
    feats = module.metrics.remove_outliers(feats.float(), q=q, exact=exact)
    return feats.cuda()


def compute_score(x_feats, y_feats, metric="mutual_knn", topk=10, normalize=True):
    """
    Uses different layer combinations of x_feats and y_feats to find the best alignment
    Args:
        x_feats: a torch tensor of shape N x L x D
        y_feats: a torch tensor of shape N x L x D
    Returns:
        best_alignment_score: the best alignment score
        best_alignment: the indices of the best alignment
    """
    best_alignment_indices = None
    best_alignment_score = 0

    for i in range(-1, x_feats.shape[1]):
        x = x_feats.flatten(1, 2) if i == -1 else x_feats[:, i, :]

        for j in range(-1, y_feats.shape[1]):
            y = y_feats.flatten(1, 2) if j == -1 else y_feats[:, j, :]

            kwargs = {}
            if 'knn' in metric:
                kwargs['topk'] = topk
                    
            if normalize:
                x = F.normalize(x, p=2, dim=-1)
                y = F.normalize(y, p=2, dim=-1)
            
            score = module.metrics.AlignmentMetrics.measure(metric, x, y, **kwargs)

            if score > best_alignment_score:
                best_alignment_score = score
                best_alignment_indices = (i, j)
    
    return best_alignment_score, best_alignment_indices

    
def compute_alignment(x_feat_paths, y_feat_paths, metric, topk, precise=True):
    """
    Args:
        x_feat_paths: list of paths to x features
        y_feat_paths: list of paths to y features
        metric: the metric to use
        topk: the number of nearest neighbors to use (specific to knn metrics)
        precise: if true use exact quantiling. (helpful to set to false if running on cpu)
            this is more of a feature to speed up matmul if using float32 
            used in measure_alignment.py
    Returns:
        alignment_scores: a numpy array of shape len(x_feat_paths) x len(y_feat_paths)
        alignment_indices: a numpy array of shape len(x_feat_paths) x len(y_feat_paths) x 2
    """
    
    

    symmetric_metric = (x_feat_paths == y_feat_paths)
    if metric == "cycle_knn":
        symmetric_metric = False

    alignment_scores = np.zeros((len(x_feat_paths), len(y_feat_paths)))
    alignment_indices = np.zeros((len(x_feat_paths), len(y_feat_paths), 2))

    pbar = tqdm(total=len(y_feat_paths) * len(x_feat_paths))

    for i, x_fp in enumerate(x_feat_paths):
        x_feats = prepare_features(torch.load(x_fp, map_location="cuda:0")["feats"].float(), exact=precise)
            
        for j, y_fp in enumerate(y_feat_paths):
            if symmetric_metric:
                if i > j:
                    pbar.update(1)
                    continue           
                        
            y_feats = prepare_features(torch.load(y_fp, map_location="cuda:0")["feats"].float(), exact=precise)
            best_score, best_indices = compute_score(y_feats, x_feats, metric=metric, topk=topk)
            
            alignment_scores[i, j] = best_score
            alignment_indices[i, j] = best_indices
            
            if symmetric_metric:
                alignment_scores[j, i] = best_score
                alignment_indices[j, i] = best_indices[::-1]

            pbar.update(1)

            del y_feats
            torch.cuda.empty_cache()

    return alignment_scores, alignment_indices


