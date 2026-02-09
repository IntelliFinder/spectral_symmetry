"""Spectral Transformer classifier for point clouds."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralTransformerClassifier(nn.Module):
    """Transformer encoder that classifies point clouds using spectral positional encodings.

    Parameters
    ----------
    input_dim : int
        Dimension of input features (3 xyz + k eigenvector entries).
    d_model : int
        Hidden dimension of the transformer.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dim_feedforward : int
        Feedforward dimension in each encoder layer.
    dropout : float
        Dropout rate.
    n_classes : int
        Number of output classes.
    """

    def __init__(
        self,
        input_dim=19,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        n_classes=10,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, mask=None):
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, input_dim)
        mask : Tensor of shape (batch, seq_len), bool, True = padded position

        Returns
        -------
        logits : Tensor of shape (batch, n_classes)
        """
        h = self.input_proj(x)  # (B, S, d_model)
        h = self.transformer_encoder(h, src_key_padding_mask=mask)  # (B, S, d_model)

        # Masked mean pooling
        if mask is not None:
            # ~mask selects valid (non-padded) positions
            valid = (~mask).unsqueeze(-1).float()  # (B, S, 1)
            h = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)

        return self.classifier(h)  # (B, n_classes)


class DistanceAttentionLayer(nn.Module):
    """Transformer layer with multiplicative distance-weighted attention.

    Standard Q/K/V attention where logits are multiplied by the distance matrix
    before softmax, allowing k-NN distances to modulate attention weights.

    Parameters
    ----------
    d_model : int
    nhead : int
    dim_feedforward : int
    dropout : float
    """

    def __init__(self, d_model=128, nhead=8, dim_feedforward=256, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, dist_matrix, mask=None):
        """Forward pass.

        Parameters
        ----------
        x : (B, S, d_model)
        dist_matrix : (B, S, S) — k-NN distances, sparse non-negative
        mask : (B, S) bool, True = padded

        Returns
        -------
        (B, S, d_model)
        """
        B, S, _ = x.shape

        # Multi-head Q, K, V
        Q = self.q_proj(x).view(B, S, self.nhead, self.d_k).transpose(1, 2)  # (B, H, S, d_k)
        K = self.k_proj(x).view(B, S, self.nhead, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.nhead, self.d_k).transpose(1, 2)

        # Attention logits
        A = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, S, S)

        # Multiplicative distance weighting: broadcast (B, 1, S, S) over heads
        A = A * dist_matrix.unsqueeze(1)

        # Padding mask: set padded positions to -inf
        if mask is not None:
            # mask is (B, S), True = padded. Expand to (B, 1, 1, S) for key positions
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            A = A.masked_fill(mask_expanded, float("-inf"))

        attn_weights = F.softmax(A, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of V
        out = torch.matmul(attn_weights, V)  # (B, H, S, d_k)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.out_proj(out)

        # Residual + LayerNorm
        x = self.norm1(x + self.resid_dropout(out))

        # Feed-forward + Residual + LayerNorm
        x = self.norm2(x + self.ffn(x))

        return x


class DistanceTransformerClassifier(nn.Module):
    """Transformer classifier using k-NN distance matrices as edge features.

    Uses custom attention layers where distance matrices multiplicatively
    modulate attention logits before softmax.

    Parameters
    ----------
    input_dim : int
        Dimension of input features (3 for xyz).
    d_model : int
    nhead : int
    num_layers : int
    dim_feedforward : int
    dropout : float
    n_classes : int
    """

    def __init__(
        self,
        input_dim=3,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        n_classes=10,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            DistanceAttentionLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, dist_matrix, mask=None):
        """Forward pass.

        Parameters
        ----------
        x : (B, S, input_dim)
        dist_matrix : (B, S, S)
        mask : (B, S) bool, True = padded

        Returns
        -------
        logits : (B, n_classes)
        """
        h = self.input_proj(x)  # (B, S, d_model)

        for layer in self.layers:
            h = layer(h, dist_matrix, mask=mask)

        # Masked mean pooling
        if mask is not None:
            valid = (~mask).unsqueeze(-1).float()  # (B, S, 1)
            h = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)

        return self.classifier(h)  # (B, n_classes)


class SpectralDistanceAttentionLayer(nn.Module):
    """Transformer layer with multiplicative k-NN distance weighting and additive spectral bias.

    Attention logits are computed as:
        A = (QK^T / sqrt(d_k)) * dist_matrix + spectral_bias

    where ``spectral_bias`` is a per-head linear projection of ``n_spectral_channels``
    invariant spectral distance channels.

    Parameters
    ----------
    d_model : int
    nhead : int
    dim_feedforward : int
    n_spectral_channels : int
        Number of spectral distance channels (one per eigenvalue group, padded to n_eigs).
    dropout : float
    """

    def __init__(self, d_model=128, nhead=8, dim_feedforward=256,
                 n_spectral_channels=16, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Per-head projection of spectral channels -> scalar bias
        self.spectral_proj = nn.Linear(n_spectral_channels, nhead, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, dist_matrix, spectral_dists, mask=None):
        """Forward pass.

        Parameters
        ----------
        x : (B, S, d_model)
        dist_matrix : (B, S, S) -- k-NN distances
        spectral_dists : (B, S, S, C) -- spectral distance channels
        mask : (B, S) bool, True = padded

        Returns
        -------
        (B, S, d_model)
        """
        B, S, _ = x.shape

        Q = self.q_proj(x).view(B, S, self.nhead, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.nhead, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.nhead, self.d_k).transpose(1, 2)

        # Standard attention logits with multiplicative k-NN distance
        A = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        A = A * dist_matrix.unsqueeze(1)  # (B, H, S, S)

        # Additive spectral bias: project C channels -> H per-head biases
        spectral_bias = self.spectral_proj(spectral_dists)  # (B, S, S, H)
        spectral_bias = spectral_bias.permute(0, 3, 1, 2)  # (B, H, S, S)
        A = A + spectral_bias

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            A = A.masked_fill(mask_expanded, float("-inf"))

        attn_weights = F.softmax(A, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.out_proj(out)

        x = self.norm1(x + self.resid_dropout(out))
        x = self.norm2(x + self.ffn(x))
        return x


class SpectralDistanceTransformerClassifier(nn.Module):
    """Transformer classifier combining k-NN distances with spectral distance bias.

    Uses ``SpectralDistanceAttentionLayer`` which multiplicatively modulates
    attention by k-NN distances and adds a per-head spectral bias computed from
    invariant spectral distance channels.

    Parameters
    ----------
    input_dim : int
        Dimension of node features (3 for xyz).
    d_model : int
    nhead : int
    num_layers : int
    dim_feedforward : int
    n_spectral_channels : int
        Number of spectral distance channels (= n_eigs).
    dropout : float
    n_classes : int
    """

    def __init__(
        self,
        input_dim=3,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        n_spectral_channels=16,
        dropout=0.1,
        n_classes=10,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            SpectralDistanceAttentionLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                n_spectral_channels=n_spectral_channels,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, dist_matrix, spectral_dists, mask=None):
        """Forward pass.

        Parameters
        ----------
        x : (B, S, input_dim)
        dist_matrix : (B, S, S) -- k-NN distances
        spectral_dists : (B, S, S, C) -- spectral distance channels
        mask : (B, S) bool, True = padded

        Returns
        -------
        logits : (B, n_classes)
        """
        h = self.input_proj(x)

        for layer in self.layers:
            h = layer(h, dist_matrix, spectral_dists, mask=mask)

        # Masked mean pooling
        if mask is not None:
            valid = (~mask).unsqueeze(-1).float()
            h = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)

        return self.classifier(h)
