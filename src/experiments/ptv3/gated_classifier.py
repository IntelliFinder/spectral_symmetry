"""PTv3 classifier with gated spectral feature fusion.

Injects a learned gate between PTv3's sparse convolution embedding and its
serialized attention encoder. The gate lets the network decide per-channel
whether to trust geometric or spectral features.

Architecture:
    h_geo  = PTv3.embedding(xyz+normals)     # (N, C)
    h_spec = Linear(eigvecs) + LN + GELU     # (N, C)
    gate   = sigmoid(FC([h_geo || h_spec]))   # (N, C)
    h      = gate * h_geo + (1-gate) * h_spec # (N, C)
    out    = PTv3.encoder(h) -> pool -> classify
"""

import torch
import torch.nn as nn

from .ptv3_model import Point, PointTransformerV3


class GatedSpectralPTv3Classifier(nn.Module):
    """PTv3 classifier with gated fusion of geometric and spectral features.

    Parameters
    ----------
    n_eigs : int
        Number of spectral eigenvector channels.
    in_channels : int
        Number of geometric input channels (e.g. 6 for xyz + normals).
    n_classes : int
        Number of output classes.
    grid_size : float
        Grid size for voxelization.
    drop_path : float
        Drop path rate for encoder blocks.
    **ptv3_kwargs
        Additional kwargs passed to PointTransformerV3.
    """

    def __init__(
        self,
        n_eigs=8,
        in_channels=6,
        n_classes=40,
        grid_size=0.01,
        drop_path=0.3,
        **ptv3_kwargs,
    ):
        super().__init__()
        self.grid_size = grid_size

        # Build the PTv3 backbone (geometric branch)
        ptv3_defaults = dict(
            in_channels=in_channels,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=drop_path,
            pre_norm=True,
            shuffle_orders=True,
            enable_rpe=False,
            enable_flash=False,
            upcast_attention=True,
            upcast_softmax=True,
            cls_mode=True,
            pdnorm_bn=False,
            pdnorm_ln=False,
        )
        ptv3_defaults.update(ptv3_kwargs)

        self.backbone = PointTransformerV3(**ptv3_defaults)

        # Embedding channel width (output of SubMConv3d stem)
        C = ptv3_defaults["enc_channels"][0]  # 32

        # Spectral projection: eigvecs -> same dim as geometric embedding
        self.spectral_proj = nn.Sequential(
            nn.Linear(n_eigs, C),
            nn.LayerNorm(C),
            nn.GELU(),
        )

        # Gate: concat [h_geo, h_spec] -> sigmoid -> per-channel weight
        self.gate_fc = nn.Sequential(
            nn.Linear(2 * C, C),
            nn.Sigmoid(),
        )

        # Classification head (same as PTv3PointceptClassifier)
        final_channels = ptv3_defaults["enc_channels"][-1]
        self.classifier = nn.Sequential(
            nn.Linear(final_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, data_dict):
        """Forward pass with gated spectral fusion.

        Parameters
        ----------
        data_dict : dict
            Must contain "coord", "feat", "eigvec", and "offset" (or "batch").

        Returns
        -------
        logits : Tensor of shape (B, n_classes)
        """
        if "grid_size" not in data_dict:
            data_dict["grid_size"] = self.grid_size

        # Extract eigvecs before Point() consumes the dict
        eigvec = data_dict.pop("eigvec")

        # Standard PTv3 preprocessing: Point -> serialization -> sparsify -> embedding
        point = Point(data_dict)
        point.serialization(
            order=self.backbone.order,
            shuffle_orders=self.backbone.shuffle_orders,
        )
        point.sparsify()
        point = self.backbone.embedding(point)

        # h_geo from sparse conv embedding
        h_geo = point.feat  # (N, C)

        # h_spec from linear projection of eigvecs
        h_spec = self.spectral_proj(eigvec)  # (N, C)

        # Gated fusion
        gate = self.gate_fc(torch.cat([h_geo, h_spec], dim=1))  # (N, C)
        fused = gate * h_geo + (1 - gate) * h_spec  # (N, C)

        # Update point features and sync sparse conv tensor
        point.feat = fused
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(fused)

        # Run through encoder
        point = self.backbone.enc(point)

        # Global mean pooling per sample
        feat = point.feat
        batch_idx = point.batch

        n_samples = batch_idx.max().item() + 1
        pooled = torch.zeros(n_samples, feat.size(1), device=feat.device, dtype=feat.dtype)
        counts = torch.zeros(n_samples, device=feat.device, dtype=feat.dtype)
        pooled.scatter_add_(0, batch_idx.unsqueeze(1).expand_as(feat), feat)
        counts.scatter_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=feat.dtype))
        pooled = pooled / counts.unsqueeze(1).clamp(min=1)

        return self.classifier(pooled)
