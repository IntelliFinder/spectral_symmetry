"""Tests for the GIN+LapPE molecular model."""

import torch

from src.experiments.molecular.model import GINLapPE


class TestGINLapPEForward:
    """Forward pass shape and gradient tests."""

    def _make_model(self, **kwargs):
        defaults = dict(
            atom_dim=9,
            pe_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_tasks=12,
            dropout=0.0,
            jumping_knowledge=False,
        )
        defaults.update(kwargs)
        return GINLapPE(**defaults)

    def _make_batch(self, n_nodes=20, n_edges=40, atom_dim=9, pe_dim=8, batch_size=2):
        """Create a fake batched graph."""
        x_atom = torch.randn(n_nodes, atom_dim)
        x_pe = torch.randn(n_nodes, pe_dim)
        # Random edges within range
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        # Assign nodes to graphs roughly equally
        batch = torch.zeros(n_nodes, dtype=torch.long)
        batch[n_nodes // 2 :] = 1
        return x_atom, x_pe, edge_index, batch

    def test_output_shape(self):
        """Output shape is (batch_size, num_tasks)."""
        model = self._make_model(num_tasks=12)
        x_atom, x_pe, edge_index, batch = self._make_batch()
        logits = model(x_atom, x_pe, edge_index, batch)
        assert logits.shape == (2, 12)

    def test_jk_variant_shape(self):
        """JumpingKnowledge variant outputs same shape."""
        model = self._make_model(num_tasks=12, jumping_knowledge=True)
        x_atom, x_pe, edge_index, batch = self._make_batch()
        logits = model(x_atom, x_pe, edge_index, batch)
        assert logits.shape == (2, 12)

    def test_gradient_flow(self):
        """Gradients flow through atom_encoder, pe_encoder, and convs."""
        model = self._make_model()
        x_atom, x_pe, edge_index, batch = self._make_batch()
        logits = model(x_atom, x_pe, edge_index, batch)
        loss = logits.sum()
        loss.backward()

        # Check that key parameters have gradients
        assert model.atom_encoder.weight.grad is not None
        assert model.pe_encoder.weight.grad is not None
        assert model.convs[0].nn[0].weight.grad is not None

    def test_single_graph(self):
        """Works with a single graph (batch_size=1) in eval mode."""
        model = self._make_model(num_tasks=5)
        model.eval()  # BatchNorm requires >1 sample in training mode
        x_atom = torch.randn(10, 9)
        x_pe = torch.randn(10, 8)
        edge_index = torch.randint(0, 10, (2, 20))
        batch = torch.zeros(10, dtype=torch.long)
        with torch.no_grad():
            logits = model(x_atom, x_pe, edge_index, batch)
        assert logits.shape == (1, 5)


class TestGINLapPEPermutationInvariance:
    """Permuting nodes within a graph should give the same graph-level output."""

    def test_node_permutation_invariance(self):
        """Graph-level readout is invariant to node ordering."""
        model = GINLapPE(
            atom_dim=9,
            pe_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_tasks=12,
            dropout=0.0,
        )
        model.eval()

        n = 15
        x_atom = torch.randn(n, 9)
        x_pe = torch.randn(n, 8)

        # Build a simple graph: chain 0-1-2-...-n
        src = list(range(n - 1))
        dst = list(range(1, n))
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        batch = torch.zeros(n, dtype=torch.long)

        with torch.no_grad():
            out1 = model(x_atom, x_pe, edge_index, batch)

        # Permute nodes
        perm = torch.randperm(n)
        inv_perm = torch.argsort(perm)

        x_atom_p = x_atom[perm]
        x_pe_p = x_pe[perm]
        # Remap edges
        edge_index_p = inv_perm[edge_index]
        batch_p = batch[perm]

        with torch.no_grad():
            out2 = model(x_atom_p, x_pe_p, edge_index_p, batch_p)

        torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)
