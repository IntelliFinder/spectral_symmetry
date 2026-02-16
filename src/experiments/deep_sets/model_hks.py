"""Deep Sets model for HKS-based point cloud classification.

Simpler than ``DeepSetsClassifier`` since HKS features already
incorporate eigenvalue information -- no eigenvalue scaling needed.
"""

import torch.nn as nn


class HKSDeepSetsClassifier(nn.Module):
    """Permutation-invariant Deep Sets classifier using HKS features.

    Takes xyz coordinates (optional) concatenated with HKS features
    and classifies shapes via per-point encoding (phi), masked mean
    pooling, and a classification head (rho).

    Parameters
    ----------
    in_channels : int
        Number of spatial coordinate channels (typically 3 for xyz).
    n_times : int
        Number of HKS time-scale channels.
    n_classes : int
        Number of output classes.
    hidden_dim : int
        Hidden dimension for phi and rho networks.
    include_xyz : bool
        Whether xyz coordinates are included in the input features.
    """

    def __init__(
        self,
        in_channels=3,
        n_times=16,
        n_classes=40,
        hidden_dim=256,
        include_xyz=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_times = n_times
        self.include_xyz = include_xyz

        feat_dim = (in_channels if include_xyz else 0) + n_times

        # phi network (per-point encoder)
        self.phi = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # rho network (classification head after pooling)
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, features, mask=None):
        """Forward pass.

        Parameters
        ----------
        features : Tensor (B, N, feat_dim)
            Optionally xyz concatenated with HKS features.
        mask : BoolTensor (B, N) or None
            True = padded/invalid, False = valid point.

        Returns
        -------
        logits : Tensor (B, n_classes)
        """
        B, N, _ = features.shape

        # Per-point phi network (reshape for BatchNorm1d)
        x = features.reshape(B * N, -1)  # (B*N, feat_dim)
        h = self.phi(x)  # (B*N, hidden_dim)
        h = h.reshape(B, N, -1)  # (B, N, hidden_dim)

        # Masked mean pooling
        if mask is not None:
            valid = (~mask).unsqueeze(-1).float()  # (B, N, 1)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # (B, hidden_dim)
        else:
            pooled = h.mean(dim=1)  # (B, hidden_dim)

        # Classification head
        logits = self.rho(pooled)  # (B, n_classes)
        return logits
