"""Lovász-Softmax loss for multiclass classification.

Adapted from https://github.com/bermanmaxim/lovasz-softmax
Original paper: "The Lovász-Softmax loss: A tractable surrogate for the
optimization of the intersection-over-union measure in neural networks"
by Berman, Triki, Blaschko (CVPR 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes="present"):
    """Multi-class Lovász-Softmax loss.

    Parameters
    ----------
    probas : Tensor (N, C)
        Class probabilities at each prediction (after softmax).
    labels : Tensor (N,)
        Ground truth class indices.
    classes : str or list
        "all", "present", or list of class indices.
    """
    if probas.numel() == 0:
        return probas * 0.0

    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            fg_class = 1.0 - probas[:, 0]
        else:
            fg_class = 1.0 - probas[:, c]
        errors = (fg - fg_class).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    if len(losses) == 0:
        return torch.tensor(0.0, device=probas.device, requires_grad=True)
    return torch.stack(losses).mean()


class LovaszLoss(nn.Module):
    """Lovász-Softmax loss for multiclass classification.

    Parameters
    ----------
    mode : str
        "multiclass" (only supported mode).
    classes : str
        "present" or "all".
    """

    def __init__(self, mode="multiclass", classes="present"):
        super().__init__()
        assert mode == "multiclass"
        self.classes = classes

    def forward(self, logits, labels):
        """
        Parameters
        ----------
        logits : Tensor (N, C)
            Raw class scores (before softmax).
        labels : Tensor (N,)
            Ground truth class indices.
        """
        probas = F.softmax(logits, dim=1)
        return lovasz_softmax_flat(probas, labels, classes=self.classes)
