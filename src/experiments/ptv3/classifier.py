"""PTv3-based classifier for point cloud classification on ModelNet."""

import torch
import torch.nn as nn
from .ptv3_model import PointTransformerV3


class PTv3Classifier(nn.Module):
    """PointTransformerV3 encoder + global mean pooling + classification head.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels (e.g. 3 for xyz, 3+k for xyz+eigvecs).
    n_classes : int
        Number of output classes.
    grid_size : float
        Grid size for voxelization.
    **ptv3_kwargs
        Additional kwargs passed to PointTransformerV3.
    """

    def __init__(self, in_channels=3, n_classes=10, grid_size=0.02, **ptv3_kwargs):
        super().__init__()
        ptv3_defaults = dict(
            in_channels=in_channels,
            enc_depths=(2, 2, 2, 2, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(256, 256, 256, 256, 256),
            stride=(2, 2, 2, 2),
            cls_mode=True,
            enable_flash=False,
            pdnorm_bn=False,
            pdnorm_ln=False,
        )
        ptv3_defaults.update(ptv3_kwargs)

        self.backbone = PointTransformerV3(**ptv3_defaults)
        self.grid_size = grid_size

        final_channels = ptv3_defaults["enc_channels"][-1]
        self.classifier = nn.Sequential(
            nn.Linear(final_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, data_dict):
        """Forward pass.

        Parameters
        ----------
        data_dict : dict
            Must contain "coord", "feat", and "offset" (or "batch").

        Returns
        -------
        logits : Tensor of shape (B, n_classes)
        """
        if "grid_size" not in data_dict:
            data_dict["grid_size"] = self.grid_size

        point = self.backbone(data_dict)

        # Global mean pooling per sample
        feat = point.feat  # (N_total, C)
        batch_idx = point.batch  # (N_total,)

        n_samples = batch_idx.max().item() + 1
        pooled = torch.zeros(n_samples, feat.size(1), device=feat.device, dtype=feat.dtype)
        counts = torch.zeros(n_samples, device=feat.device, dtype=feat.dtype)
        pooled.scatter_add_(0, batch_idx.unsqueeze(1).expand_as(feat), feat)
        counts.scatter_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=feat.dtype))
        pooled = pooled / counts.unsqueeze(1).clamp(min=1)

        return self.classifier(pooled)


class PTv3PointceptClassifier(nn.Module):
    """PTv3 classifier matching the Pointcept ModelNet40 config.

    Key differences from PTv3Classifier:
    - in_channels=6 (xyz + normals)
    - enc_depths=(2, 2, 2, 6, 2) — deeper stage 4
    - enc_patch_size=1024 at all stages
    - 4 serialization orders: z, z-trans, hilbert, hilbert-trans
    - 3-layer classifier head: 512->256->128->n_classes
    - grid_size=0.01
    """

    def __init__(self, in_channels=6, n_classes=40, grid_size=0.01,
                 drop_path=0.3, **ptv3_kwargs):
        super().__init__()
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
        self.grid_size = grid_size

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
        if "grid_size" not in data_dict:
            data_dict["grid_size"] = self.grid_size

        point = self.backbone(data_dict)

        feat = point.feat
        batch_idx = point.batch

        n_samples = batch_idx.max().item() + 1
        pooled = torch.zeros(n_samples, feat.size(1), device=feat.device, dtype=feat.dtype)
        counts = torch.zeros(n_samples, device=feat.device, dtype=feat.dtype)
        pooled.scatter_add_(0, batch_idx.unsqueeze(1).expand_as(feat), feat)
        counts.scatter_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=feat.dtype))
        pooled = pooled / counts.unsqueeze(1).clamp(min=1)

        return self.classifier(pooled)
