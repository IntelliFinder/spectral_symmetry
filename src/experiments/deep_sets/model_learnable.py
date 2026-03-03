"""Learnable Spectral Weighting Deep Sets model.

Replaces the fixed exponential kernel in HKS/WES with a learned MLP that
maps the full eigenvalue spectrum to a (K, T) weight matrix.  Each output
channel j is a learned linear combination of eigenvectors, where the
combination depends on the shape's eigenvalue spectrum globally.

Key difference from HKS
------------------------
- HKS: ``w(k,t) = exp(-lambda_k * t)``  — eigenvalues are used independently
- Here: ``W = MLP(lambda_1,...,lambda_K)`` — all eigenvalues jointly determine
  the weighting, enabling inter-eigenvalue reasoning (gaps, ratios, clusters).

Forward signature
-----------------
``model(features, eigenvalues, mask)``

where

- ``features``   : (B, N, 3+K) -- xyz (optional) || raw/squared eigvecs
- ``eigenvalues`` : (B, K)    -- eigenvalue spectrum for MLP conditioning
- ``mask``       : (B, N)     -- True = padded

Output: ``logits`` (B, n_classes)
"""

import torch
import torch.nn as nn


class LearnableSpectralDeepSets(nn.Module):
    """Deep Sets classifier with a learnable spectral weighting MLP.

    Parameters
    ----------
    n_eigs : int
        Number of eigenpairs K.
    n_output_channels : int
        Number of output spectral channels T (analogous to HKS time scales).
    n_classes : int
        Number of output classes.
    hidden_dim : int
        Hidden dimension for phi and rho networks.
    mlp_hidden : int
        Hidden dimension for the spectral weighting MLP.
    include_xyz : bool
        Whether xyz coordinates are included as the first 3 feature channels.
    """

    def __init__(
        self,
        n_eigs=8,
        n_output_channels=32,
        n_classes=10,
        hidden_dim=512,
        mlp_hidden=128,
        include_xyz=True,
    ):
        super().__init__()
        self.n_eigs = n_eigs
        self.n_output_channels = n_output_channels
        self.include_xyz = include_xyz
        T = n_output_channels
        K = n_eigs

        # Spectral weighting MLP: eigenvalues -> (K, T) weight matrix
        # Input: K eigenvalues.  Output: K*T weights (reshaped to K x T).
        self.spectral_mlp = nn.Sequential(
            nn.Linear(K, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, K * T),
        )

        # phi network: per-point encoder
        feat_dim = (3 if include_xyz else 0) + T
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

        # rho network: classification head after pooling
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, features, eigenvalues, mask=None):
        """Forward pass.

        Parameters
        ----------
        features : Tensor (B, N, coord_dim + K)
            Per-point features: optional xyz followed by K eigvec entries
            (raw or squared depending on dataset configuration).
        eigenvalues : Tensor (B, K)
            Eigenvalue spectrum used to condition the weighting MLP.
        mask : BoolTensor (B, N) or None
            True = padded/invalid, False = valid point.

        Returns
        -------
        logits : Tensor (B, n_classes)
        """
        B, N, _ = features.shape
        K = self.n_eigs
        T = self.n_output_channels

        # -- Spectral weighting MLP --
        # weights: (B, K*T) -> (B, K, T)
        weights = self.spectral_mlp(eigenvalues).reshape(B, K, T)

        # -- Per-point spectral features --
        # eigvecs: (B, N, K)  -- eigvec part of features
        coord_dim = 3 if self.include_xyz else 0
        eigvecs = features[:, :, coord_dim:]  # (B, N, K)

        # spectral: (B, N, T) via batched matmul
        spectral = torch.einsum("bnk,bkt->bnt", eigvecs, weights)  # (B, N, T)

        # -- Concatenate xyz (optional) with spectral features --
        if self.include_xyz:
            xyz = features[:, :, :3]  # (B, N, 3)
            x = torch.cat([xyz, spectral], dim=-1)  # (B, N, 3+T)
        else:
            x = spectral  # (B, N, T)

        # -- phi: per-point encoder --
        x_flat = x.reshape(B * N, -1)
        h = self.phi(x_flat)  # (B*N, hidden_dim)
        h = h.reshape(B, N, -1)  # (B, N, hidden_dim)

        # -- Masked mean pooling --
        if mask is not None:
            valid = (~mask).unsqueeze(-1).float()  # (B, N, 1)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            pooled = h.mean(dim=1)

        # -- rho: classification head --
        logits = self.rho(pooled)
        return logits
