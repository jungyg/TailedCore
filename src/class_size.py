import torch
import time
import gc
import numpy as np
import torch.nn.functional as F
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import mode

from . import utils




def compute_self_sim(X: torch.Tensor, normalize: bool = True) -> torch.Tensor:

    if normalize:
        X = F.normalize(X, dim=-1)

    if X.ndim == 2:
        Xt = X.t()
    elif X.ndim == 3:
        Xt = X.permute(0, 2, 1)
    else:
        raise NotImplementedError()

    self_sim = torch.matmul(X, Xt)

    return self_sim


def predict_class_sizes(self_sim: torch.Tensor, th, vote_type="mean") -> torch.Tensor:
    b = self_sim.shape[0]
    mask = self_sim >= th

    count_map = mask * mask.sum(dim=1, keepdim=True)

    if vote_type == "mean":
        class_sizes = count_map.sum(dim=0) / torch.count_nonzero(count_map, dim=0)
    elif vote_type == "mode":
        count_map = count_map.numpy()
        count_map_naned = np.where(count_map == 0, np.nan, count_map)
        class_sizes = mode(
            count_map_naned, axis=0, nan_policy="omit"
        ).mode  # FIXME: what happens when there is no majority?
    elif vote_type == "nearest":
        columnwise_means = (
            count_map.sum(dim=0, keepdim=True)
            / torch.count_nonzero(count_map, dim=0)[None, :]
        )
        idxes = torch.argmin((count_map - columnwise_means).abs(), dim=0)
        class_sizes = count_map[torch.arange(b), idxes]

    if isinstance(class_sizes, np.ndarray):
        class_sizes = torch.from_numpy(class_sizes)

    return class_sizes.to(torch.float)


def predict_num_samples_per_class(
    class_sizes: torch.FloatTensor, round_class_sizes=True
) -> torch.FloatTensor:
    # FIXME: topk average is required for robustness
    if round_class_sizes:
        class_sizes = torch.round(class_sizes).to(torch.long)
    _class_sizes_sorted = torch.sort(class_sizes, descending=False)[0]
    class_sizes_sorted = torch.maximum(
        _class_sizes_sorted, torch.ones_like(_class_sizes_sorted)
    )

    num_samples_per_class = []

    class_sizes_sorted = class_sizes_sorted.squeeze()

    while len(class_sizes_sorted) > 0:
        if len(class_sizes_sorted) < class_sizes_sorted[0]:
            num_samples_per_class[-1] += len(class_sizes_sorted)
            break

        _num_samples_in_current_class = class_sizes_sorted[0]
        _num_samples_in_current_class = (
            torch.round(
                class_sizes_sorted[:_num_samples_in_current_class].float().mean()
            )
            .long()
            .item()
        )
        _num_samples_in_current_class = min(
            _num_samples_in_current_class, len(class_sizes_sorted)
        )
        num_samples_per_class.append(_num_samples_in_current_class)
        class_sizes_sorted = class_sizes_sorted[_num_samples_in_current_class:]
    # return torch.FloatTensor(num_samples_per_class).sort(descending=True)[0]
    return torch.LongTensor(num_samples_per_class).sort(descending=True)[0]


def predict_few_shot_class_samples(class_sizes: torch.Tensor, percentile=.15) -> torch.Tensor:

    num_samples_per_class = predict_num_samples_per_class(class_sizes)

    max_K = predict_max_K(num_samples_per_class, percentile=percentile)

    few_shot_idxes = (class_sizes <= max_K).to(torch.long)
    return few_shot_idxes

def predict_max_K(num_samples_per_class: torch.Tensor, percentile=.15):
    
    max_K = _predict_max_K_elbow(num_samples_per_class)

    if percentile < 1:
        max_K_percentile = _predict_max_K_max_within_percnetile(num_samples_per_class, p=percentile)
        max_K = min(max_K, max_K_percentile)

    return max_K

def _predict_max_K_max_within_percnetile(num_samples_per_class: torch.Tensor, p=0.15):

    num_samples_per_class = num_samples_per_class.sort(descending=True)[0]

    n = int(num_samples_per_class.sum() * p)

    cum_den = num_samples_per_class.flip(dims=[0]).cumsum(dim=0)

    k = 0
    for nspc in cum_den:
        if nspc > n:
            break
        k += 1

    max_K_idx = len(cum_den) - k
    max_K_idx = min(max_K_idx, len(num_samples_per_class)-1)

    max_K = num_samples_per_class[max_K_idx]
    return max_K.item()

def _predict_max_K_elbow(num_samples_per_class):
    return elbow(num_samples_per_class)

def predict_few_shot_class_samples_by_scores(scores: torch.Tensor) -> torch.Tensor:
    elbow_score = elbow(scores)
    is_few_shot = (scores < elbow_score).long()
    return is_few_shot


def elbow(scores: torch.Tensor, sort=True, quantize=False):
    if quantize:
        n = float(len(scores))
        scores = (scores * n).long()
    else:
        n = 1.
    
    if sort:
        scores = scores.sort(descending=True)[0]
    
    _scores = torch.cat(
        [
            torch.arange(len(scores))[:, None],
            scores[:, None],
        ],
        dim=1,
    )

    ods = compute_orthogonal_distances(
        _scores, _scores[[0, -1], :]
    )

    elbow_idx = ods.argmax()

    return scores[elbow_idx].item() / n
    

def compute_self_sim_min(self_sim: torch.Tensor, mode="min") -> torch.Tensor:
    mins = torch.empty(len(self_sim))

    for i in range(len(self_sim)):
        mins[i] = compute_min(self_sim[i])

    if mode == "avg":
        minimum = mins.mean()
    elif mode == "min":
        minimum = mins.min()
    else:
        raise NotImplementedError()

    return minimum


def compute_min(scores: torch.Tensor) -> torch.Tensor:
    return _compute_min(scores)


def _compute_min(scores: torch.Tensor) -> torch.Tensor:
    return scores.min()


_WITHIN_FACTOR = 2


def compute_sym_th(self_sim: torch.Tensor, mode="min", within=False) -> float:

    _factor = 1
    if within:
        _factor = _WITHIN_FACTOR

    minimum = compute_self_sim_min(self_sim, mode=mode)
    th = torch.cos(torch.acos(minimum) / (2 * _factor)).item()

    return th


def _compute_th(self_sim: torch.Tensor, th_type="symmin", within=False) -> float:

    _factor = 1
    if within:
        _factor = _WITHIN_FACTOR

    if th_type == "symmin":
        th = compute_sym_th(self_sim, mode="min", within=within)
    elif th_type == "symavg":
        th = compute_sym_th(self_sim, mode="avg", within=within)
    elif th_type == "indep":
        th = np.cos((np.pi / 2) / (2 * _factor))
    else:
        raise NotImplementedError()

    return th


def compute_th(
    self_sim: torch.Tensor,
    num_bootstrapping=1,
    subsampling_ratio=1.0,
    th_type="symmin",
    within=False,
) -> float:

    if num_bootstrapping == 1:
        th = _compute_th(self_sim, th_type=th_type, within=within)
        return th

    assert num_bootstrapping > 1
    n = len(self_sim)
    b = round(subsampling_ratio * n)

    ths = []
    for _ in range(num_bootstrapping):
        _idxes = torch.randint(0, n, (b,))
        _self_sim = self_sim[_idxes, :]
        _self_sim = _self_sim[:, _idxes]
        _th = _compute_th(_self_sim, th_type=th_type, within=within)
        ths.append(_th)

    th = np.mean(ths)

    return th


def sample_few_shot(
    X: torch.Tensor,
    fea_map_shape: bool = None,
    th_type: str = "symmin",
    vote_type: str = "mean",
    num_bootstrapping: int = 1,
    subsampling_ratio: float = 1.0,
    return_class_sizes: bool = False,
):
    if fea_map_shape is not None:
        return _sample_patchwise_few_shot(
            X,
            fea_map_shape=fea_map_shape,
            th_type=th_type,
            vote_type=vote_type,
            num_bootstrapping=num_bootstrapping,
            subsampling_ratio=subsampling_ratio,
            return_class_sizes=return_class_sizes,
        )

    return _sample_few_shot(
        X,
        th_type=th_type,
        vote_type=vote_type,
        num_bootstrapping=num_bootstrapping,
        subsampling_ratio=subsampling_ratio,
        return_class_sizes=return_class_sizes,
    )


def _sample_few_shot(
    X: torch.Tensor,
    th_type="symmin",
    vote_type="mean",
    num_bootstrapping=1,
    subsampling_ratio=1.0,
    return_class_sizes=False,
) -> torch.Tensor:
    self_sim = compute_self_sim(X, normalize=True)
    th = compute_th(
        self_sim,
        num_bootstrapping=num_bootstrapping,
        subsampling_ratio=subsampling_ratio,
        th_type=th_type,
    )

    class_sizes = predict_class_sizes(self_sim, th=th, vote_type=vote_type)
    if return_class_sizes:
        return class_sizes

    is_few_shot = predict_few_shot_class_samples(class_sizes)

    sample_indices = torch.where(is_few_shot == 1)[0]

    return X[sample_indices], sample_indices


def _sample_patchwise_few_shot(
    features: torch.Tensor,
    fea_map_shape,
    th_type="symmin",
    vote_type="mean",
    num_bootstrapping=1,
    subsampling_ratio=1.0,
    return_class_sizes=False,
):
    """
    Args:
        features: (n*h*w, c). To revert it back to feature map, the below is required.
        (n*h*w, c) -> (n, h*w, c) -> (n, c, h*w) -> (n, c, h, w)
    """

    fea_dim = features.shape[1]
    h, w = fea_map_shape[0], fea_map_shape[1]
    num_pixels = h * w

    patchwise_features = features.reshape(-1, num_pixels, fea_dim).permute(
        1, 0, 2
    )  # (num_patches, n, c)
    b = patchwise_features.shape[1]

    patchwise_class_sizes = torch.empty((b, num_pixels), dtype=torch.float)
    patchwise_few_shot_idxes = torch.empty((b, num_pixels), dtype=torch.long)

    def _compute_patchwise_few_shot_idxes(p):
        _features = patchwise_features[p]
        _self_sim = compute_self_sim(_features, normalize=True)
        _th = compute_th(
            _self_sim,
            num_bootstrapping=num_bootstrapping,
            subsampling_ratio=subsampling_ratio,
            th_type=th_type,
            within=True,
        )

        _class_sizes = predict_class_sizes(_self_sim, th=_th, vote_type=vote_type)
        patchwise_class_sizes[:, p] = _class_sizes

        if not return_class_sizes:
            _few_shot_idxes = predict_few_shot_class_samples(_class_sizes)
            patchwise_few_shot_idxes[:, p] = _few_shot_idxes

    for p in tqdm(range(num_pixels), desc="Computing patchwise few-shot idxes"):
        _compute_patchwise_few_shot_idxes(p)

    # Parallel(n_jobs=4)(delayed(_compute_pixelwise_few_shot_idxes)(p) for p in range(num_pixels))
    if return_class_sizes:
        return patchwise_class_sizes.reshape((-1,))

    sample_indices = patchwise_few_shot_idxes.reshape((-1,))
    sample_features = features[sample_indices]

    return sample_features, sample_indices


def compute_orthogonal_distances(
    points: torch.Tensor, line_points: list
) -> torch.Tensor:

    points = points.numpy()

    # Extract coordinates of the line
    (x1, y1), (x2, y2) = line_points[0].numpy(), line_points[1].numpy()

    # Calculate the slope of the line (y = mx + c)
    dx, dy = x2 - x1, y2 - y1
    if dx == 0:  # Special case for vertical line
        return torch.FloatTensor([abs(x - x1) for x, y in points])

    m = dy / dx
    c = y1 - m * x1
    m_perp = -1 / m  # Slope of the perpendicular line

    distances = []
    for x, y in tqdm(points):
        c_perp = y - m_perp * x

        # Find intersection
        x_intersect = (c_perp - c) / (m - m_perp)
        y_intersect = m * x_intersect + c

        # Distance from point to line
        distance = np.sqrt((x_intersect - x) ** 2 + (y_intersect - y) ** 2)
        distances.append(distance)

    distances = torch.FloatTensor(distances)

    return distances


if __name__ == "__main__":
    # Test the function
    # num_samples, h, w, c = 1624, 28, 28, 1024  # Example dimensions
    # fea_maps = np.random.rand(num_samples, h, w, c)
    # self_sim = compute_pixelwise_self_sim(fea_maps)

    # print("Shape of fea_maps:", fea_maps.shape)
    # print("Shape of self_sim:", self_sim.shape)
    print("above requires revision")
