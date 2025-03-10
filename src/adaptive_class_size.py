import torch
import math
import gc
import numpy as np
import torch.nn.functional as F
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import mode

from . import utils
from . import class_size


def predict_adaptive_class_sizes(
    self_sim: torch.Tensor, 
    th_type: str = "double_max_step",
    vote_type: str = "none",
):

    ths = _compute_ths(self_sim, th_type)

    mask = self_sim >= ths
    count_map = mask * mask.sum(dim=1, keepdim=True)

    class_sizes = _vote(count_map, vote_type)
    
    return class_sizes

def _vote(count_map: torch.Tensor, vote_type) -> torch.Tensor:
    
    if vote_type == "none":
        return _vote_none(count_map)
    elif vote_type == "mode":
        return _vote_mode(count_map)
    elif vote_type == "mean":
        return _vote_mean(count_map)

def _vote_mean(count_map):
    class_sizes = count_map.sum(dim=0) / torch.count_nonzero(count_map, dim=0)
    return class_sizes.to(torch.float)

def _vote_none(count_map: torch.Tensor):
    class_sizes = count_map.float().mean(dim=1)
    return class_sizes.to(torch.float)

def _vote_mode(count_map: torch.Tensor):
    count_map = count_map.numpy()
    count_map_naned = np.where(count_map == 0, np.nan, count_map)
    class_sizes = mode(
        count_map_naned, axis=0, nan_policy="omit"
    ).mode
    return torch.from_numpy(class_sizes).to(torch.float)

def _compute_ths(self_sim: torch.Tensor, th_type) -> torch.Tensor:
    self_sim = self_sim.numpy()

    if th_type == "max_step":
        _compute_th = _compute_th_max_step
    elif th_type == "double_max_step":
        _compute_th = _compute_th_double_max_step
    elif th_type == "max_step_min_num_neighbors":
        _compute_th = _compute_th_max_step_min_num_neighbors
    elif th_type == "double_min_bin_count":
        _compute_th = _compute_th_double_min_bin_count
    elif th_type == "min_kde":
        _compute_th = _compute_th_min_kde
    elif th_type == "double_min_kde":
        _compute_th = _compute_th_double_min_kde
    elif th_type == "half_min":
        _compute_th = _compute_th_half_min
    elif th_type == "ruled_max_step":
        _compute_th = _compute_th_ruled_max_step
    elif th_type == "trim_min":
        _compute_th = _compute_th_trim_min
    elif th_type == "truncate_min":
        _compute_th = _compute_th_truncate_min
    

    n = len(self_sim)
    ths = [[]] * n

    def compute_th(i):
        _sim = self_sim[i]
        _th = _compute_th(_sim)
        return _th

    # for i in range(n):
    #     ths[i] = compute_th(i)

    ths = Parallel(n_jobs=-1)(delayed(compute_th)(i) for i in range(n))

    ths = torch.FloatTensor(ths)[:, None]
    return ths


def _compute_th_max_step(scores: np.ndarray):
    scores_sorted = _sort(scores, descending=False)
    diff = np.diff(scores_sorted)

    idx_max_step = diff.argmax()

    th = (scores_sorted[idx_max_step + 1] + scores_sorted[idx_max_step]) / 2

    return th

# TODO: keep it in case
# def _compute_th_double_max_step(scores: np.ndarray):
#     scores_sorted = _sort(scores, descending=False)
#     diff = np.diff(scores_sorted)

#     crit1 = diff

#     crit2 = np.empty_like(crit1)
#     for i in range(len(crit2) - 1):
#         crit2[i] = np.max(crit1[i + 1 :])

#     # crit2[-1] = diff[-1]
#     crit2[-1] = crit1.mean()

#     crit = crit1 / np.maximum(crit2, 1e-7)

#     idx_max_step = crit.argmax()

#     th = (scores_sorted[idx_max_step] + scores_sorted[idx_max_step + 1]) / 2

#     return th

def _compute_th_double_max_step(scores: np.ndarray):
    scores_sorted = _sort(scores, descending=False)
    diffs = np.diff(scores_sorted)

    idx_max_step = _double_criterion(diffs)

    th = (scores_sorted[idx_max_step] + scores_sorted[idx_max_step + 1]) / 2

    return th

def _double_criterion(crits: np.ndarray):

    next_crits = np.empty_like(crits)
    for i in range(len(next_crits) - 1):
        next_crits[i] = np.max(crits[i + 1 :])

    next_crits[-1] = crits[-1]

    final_crits = crits / np.maximum(next_crits, 1e-7)

    idx_max_crit = final_crits.argmax()

    return idx_max_crit

def _compute_th_double_min_bin_count(scores: np.ndarray):
    scores_sorted = _sort(scores, descending=False)
    bin_counts = scores_to_bin_counts(scores_sorted)[:-1]
    idx = _double_criterion(bin_counts.max() - bin_counts)

    th = (scores_sorted[idx] + scores_sorted[idx + 1]) / 2

    return th


def _compute_th_min_kde(scores: np.ndarray):
    scores_sorted = _sort(scores, descending=False)
    kde_log_density = scores_to_kde(scores_sorted)[:-1]

    idx = kde_log_density.argmin()

    th = (scores_sorted[idx] + scores_sorted[idx + 1]) / 2

    return th


def _compute_th_double_min_kde(scores: np.ndarray):
    scores_sorted = _sort(scores, descending=False)
    kde_log_density = scores_to_kde(scores_sorted)[:-1]

    idx = _double_criterion(kde_log_density.max() - kde_log_density)

    th = (scores_sorted[idx] + scores_sorted[idx + 1]) / 2

    return th

def _compute_th_half_min(scores: np.ndarray):
    m = scores.min()
    th = np.cos(np.arccos(m) / 2)

    return th

def _compute_th_trim_min(scores: np.ndarray):
    scores = _sort(scores, descending=False)
    m = scores.min()
    th_half = np.cos(np.arccos(m)/2)
    scores_half = scores[scores>th_half]

    p = 0.85
    scores_trim = scores_half[-int(len(scores_half) * p)]
    th = scores_trim
    return th

def _compute_th_truncate_min(scores: np.ndarray):
    scores = _sort(scores, descending=False)
    m = scores.min()
    th_half = np.cos(np.arccos(m)/2)
    scores_half = scores[scores>th_half]

    p = 0.5
    scores_trim = scores_half[-int(len(scores_half) * p)]
    th = scores_trim
    return th

def _compute_th_ruled_max_step(scores: np.ndarray):
    m = scores.min()
    th_quarter = np.cos(np.arccos(m) / 8)
    th_half = np.cos(np.arccos(m) / 2)

    th_double_max_step = _compute_th_double_max_step(scores)

    if th_double_max_step >= th_quarter:
        return th_half
    
    return th_double_max_step



def _compute_th_max_step_min_num_neighbors(scores: np.ndarray):
    scores_sorted = _sort(scores, descending=True)
    diff = -np.diff(scores_sorted)

    crit1 = diff
    crit2 = -np.log(np.arange(len(diff)) + 1) + math.log(len(diff))

    crit = crit1 * crit2

    idx_max_step = crit.argmax()

    th = (scores_sorted[idx_max_step] + scores_sorted[idx_max_step + 1]) / 2

    return th




def _sort(scores: np.ndarray, descending: bool = False, quantize=True) -> np.ndarray:

    dtype = scores.dtype

    if quantize:
        n = len(scores)
        scores = (n * scores).astype(np.int_)
    else:
        n = 1

    scores = np.sort(scores).astype(dtype) / n

    if descending:
        scores = scores[::-1]

    return scores


def adaptively_sample_few_shot(
    X: torch.Tensor,
    th_type: str = "double_max_step",
    vote_type: str = "none",
    return_class_sizes=False,
):
    self_sim = class_size.compute_self_sim(X)
    class_sizes = predict_adaptive_class_sizes(self_sim, th_type, vote_type).squeeze()
    if return_class_sizes:
        return class_sizes

    is_few_shot = class_size.predict_few_shot_class_samples(class_sizes)

    sample_indices = torch.where(is_few_shot == 1)[0]

    return X[sample_indices], sample_indices


def scores_to_bin_counts(scores):
    # Calculate IQR
    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)
    IQR = Q3 - Q1
    
    # Number of data points
    n = len(scores)
    
    # Freedman-Diaconis Rule for bin width
    bin_width = 2 * IQR / (n ** (1/3))
    
    # Calculate the number of bins
    data_range = np.max(scores) - np.min(scores)
    num_bins = int(np.round(data_range / bin_width))
    
    # Compute histogram bins
    hist, bin_edges = np.histogram(scores, bins=num_bins)
    
    bin_counts = np.zeros_like(scores)
    for i in range(num_bins):
        if i < num_bins -1 :
            in_bin = (scores >= bin_edges[i]) & (scores < bin_edges[i+1])
        else:
            in_bin = (scores >= bin_edges[i])
        bin_counts[in_bin] = hist[i]
    
    return bin_counts


from sklearn.neighbors import KernelDensity
def scores_to_kde(scores,):
    scores = np.array(scores).reshape(-1, 1)
    
    # Fit the KDE model
    kde = KernelDensity(kernel='tophat', bandwidth='silverman').fit(scores)
    
    # Evaluate the log density of the input scores
    log_density = kde.score_samples(scores)

    return log_density