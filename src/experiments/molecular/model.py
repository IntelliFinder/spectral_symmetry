"""GIN + Laplacian Positional Encoding for molecular property prediction.

Architecture:
  1. Atom features (9-dim OGB one-hot) projected via linear layer
  2. LapPE (k eigenvector values per node) projected via linear layer
  3. Sum of projections → GIN backbone → global mean pool → MLP → logits

Supports configurable hidden_dim, num_layers, dropout, and JumpingKnowledge.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool


class GINLapPE(nn.Module):
    """GIN with Laplacian Positional Encoding for graph-level prediction.

    Parameters
    ----------
    atom_dim : int
        Dimension of atom input features (default 9 for OGB).
    pe_dim : int
        Number of eigenvector channels (LapPE dimension).
    hidden_dim : int
        Hidden dimension for all layers.
    num_layers : int
        Number of GIN convolution layers.
    num_tasks : int
        Number of output tasks (12 for moltox21, 128 for molpcba).
    dropout : float
        Dropout rate between GIN layers.
    jumping_knowledge : bool
        If True, concatenate all layer outputs before readout.
    """

    def __init__(
        self,
        atom_dim=9,
        pe_dim=8,
        hidden_dim=256,
        num_layers=5,
        num_tasks=12,
        dropout=0.5,
        jumping_knowledge=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge

        # Input projections
        self.atom_encoder = nn.Linear(atom_dim, hidden_dim)
        self.pe_encoder = nn.Linear(pe_dim, hidden_dim)

        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Classification head
        if jumping_knowledge:
            jk_dim = hidden_dim * num_layers
        else:
            jk_dim = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, x_atom, x_pe, edge_index, batch):
        """Forward pass.

        Parameters
        ----------
        x_atom : Tensor (N_total, atom_dim)
            Atom features for all nodes in the batch.
        x_pe : Tensor (N_total, pe_dim)
            Laplacian PE values for all nodes in the batch.
        edge_index : LongTensor (2, E_total)
            Edge indices for the batch.
        batch : LongTensor (N_total,)
            Batch assignment vector.

        Returns
        -------
        logits : Tensor (B, num_tasks)
        """
        # Encode and sum
        h = self.atom_encoder(x_atom) + self.pe_encoder(x_pe)

        # GIN layers
        layer_outputs = []
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = torch.relu(h)
            h = nn.functional.dropout(h, p=self.dropout, training=self.training)
            layer_outputs.append(h)

        # Readout
        if self.jumping_knowledge:
            h = torch.cat(layer_outputs, dim=-1)
        # else: h is already the last layer output

        h = global_mean_pool(h, batch)
        logits = self.classifier(h)
        return logits
