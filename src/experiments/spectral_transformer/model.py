"""Spectral Transformer classifier for point clouds."""

import torch.nn as nn


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
