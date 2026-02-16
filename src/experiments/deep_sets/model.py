"""Deep Sets model for point cloud classification with eigenvalue scaling."""

import torch
import torch.nn as nn


class DeepSetsClassifier(nn.Module):
    """Permutation-invariant Deep Sets classifier for point clouds.

    Takes xyz coordinates concatenated with eigenvector features and classifies
    shapes via per-point encoding (phi), masked mean pooling, and a classification
    head (rho).

    Supports two modes of eigenvalue integration:
    - "fixed": eigenvectors scaled by 1/sqrt(eigenvalue)
    - "learnable": eigenvectors scaled by a learned function of eigenvalues

    When ``use_spectrum=True``, the eigenvalue spectrum is processed through a
    small MLP and concatenated to the pooled representation before rho, providing
    a global Shape-DNA-style descriptor that bypasses eigenvector sign ambiguity.

    Parameters
    ----------
    in_channels : int
        Number of spatial coordinate channels (typically 3 for xyz).
    n_eigs : int
        Number of eigenvector channels.
    n_classes : int
        Number of output classes.
    hidden_dim : int
        Hidden dimension for phi and rho networks.
    scaling_mode : str
        Either "fixed" or "learnable".
    use_spectrum : bool
        If True, process eigenvalue spectrum through MLP and concatenate
        to pooled features before the classification head.
    """

    def __init__(
        self,
        in_channels=3,
        n_eigs=8,
        n_classes=40,
        hidden_dim=256,
        scaling_mode="fixed",
        use_spectrum=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_eigs = n_eigs
        self.scaling_mode = scaling_mode
        self.use_spectrum = use_spectrum

        if scaling_mode not in ("fixed", "learnable"):
            raise ValueError(f"scaling_mode must be 'fixed' or 'learnable', got '{scaling_mode}'")

        if scaling_mode == "learnable":
            self.eigenvalue_fn = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Softplus(),
            )

        feat_dim = in_channels + n_eigs

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

        if use_spectrum:
            # MLP to process eigenvalue spectrum: (n_eigs,) -> (hidden_dim,)
            self.spectrum_mlp = nn.Sequential(
                nn.Linear(n_eigs, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            # rho input: hidden_dim (from pool) + hidden_dim (from spectrum)
            rho_in_dim = 2 * hidden_dim
        else:
            rho_in_dim = hidden_dim

        # rho network (classification head after pooling)
        self.rho = nn.Sequential(
            nn.Linear(rho_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, features, eigenvalues, mask=None):
        """Forward pass.

        Parameters
        ----------
        features : Tensor (B, N, in_channels + n_eigs)
            xyz coordinates concatenated with raw eigenvectors.
        eigenvalues : Tensor (B, n_eigs)
            Per-shape eigenvalues for scaling.
        mask : BoolTensor (B, N) or None
            True = padded/invalid, False = valid point.

        Returns
        -------
        logits : Tensor (B, n_classes)
        """
        B, N, _ = features.shape

        # 1. Split into xyz and eigenvectors
        xyz = features[:, :, : self.in_channels]  # (B, N, in_channels)
        eigvecs = features[:, :, self.in_channels :]  # (B, N, n_eigs)

        # 2. Scale eigenvectors by eigenvalues
        if self.scaling_mode == "fixed":
            # 1/sqrt(lambda) scaling (diffusion distance / heat kernel)
            scale = 1.0 / torch.sqrt(eigenvalues.unsqueeze(1).clamp(min=1e-8))  # (B, 1, n_eigs)
            scaled_eigvecs = eigvecs * scale  # (B, N, n_eigs)
        else:
            # Learnable scaling: f(lambda_j) per eigenvalue channel
            # eigenvalues: (B, n_eigs) -> (B, n_eigs, 1) for MLP input
            scale_factors = self.eigenvalue_fn(eigenvalues.unsqueeze(-1)).squeeze(-1)  # (B, n_eigs)
            scaled_eigvecs = eigvecs * scale_factors.unsqueeze(1)  # (B, N, n_eigs)

        # 3. Concatenate xyz with scaled eigenvectors
        x = torch.cat([xyz, scaled_eigvecs], dim=-1)  # (B, N, in_channels + n_eigs)

        # 4. Per-point phi network (reshape for BatchNorm1d)
        x = x.reshape(B * N, -1)  # (B*N, feat_dim)
        h = self.phi(x)  # (B*N, hidden_dim)
        h = h.reshape(B, N, -1)  # (B, N, hidden_dim)

        # 5. Masked mean pooling
        if mask is not None:
            valid = (~mask).unsqueeze(-1).float()  # (B, N, 1)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # (B, hidden_dim)
        else:
            pooled = h.mean(dim=1)  # (B, hidden_dim)

        # 5b. Concatenate eigenvalue spectrum descriptor if enabled
        if self.use_spectrum:
            spectrum_feat = self.spectrum_mlp(eigenvalues)  # (B, hidden_dim)
            pooled = torch.cat([pooled, spectrum_feat], dim=1)  # (B, 2*hidden_dim)

        # 6. Classification head
        logits = self.rho(pooled)  # (B, n_classes)
        return logits


class SpectrumClassifier(nn.Module):
    """Classifies shapes using only the eigenvalue spectrum (Shape-DNA style).

    No per-point processing -- just a global descriptor. The eigenvalue spectrum
    {lambda_1, ..., lambda_k} is a proven shape descriptor that completely
    bypasses eigenvector sign ambiguity.

    Parameters
    ----------
    n_eigs : int
        Number of eigenvalues in the spectrum.
    n_classes : int
        Number of output classes.
    hidden_dim : int
        Hidden dimension for the MLP.
    """

    def __init__(self, n_eigs=8, n_classes=40, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_eigs, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
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
        features : Tensor (B, N, C)
            Per-point features (ignored).
        eigenvalues : Tensor (B, n_eigs)
            Per-shape eigenvalue spectrum.
        mask : BoolTensor (B, N) or None
            Point mask (ignored).

        Returns
        -------
        logits : Tensor (B, n_classes)
        """
        # Ignore per-point features entirely, just use eigenvalues
        return self.net(eigenvalues)
