"""Training loop for PTv3 classifier on ModelNet."""

import torch
import torch.nn as nn


def train_one_epoch_ptv3(model, loader, criterion, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    for data_dict, labels in loader:
        # Move tensors to device
        data_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data_dict.items()
        }
        labels = labels.to(device)

        logits = model(data_dict)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate_ptv3(model, loader, device):
    """Evaluate and return accuracy."""
    model.eval()
    correct = 0
    total = 0
    for data_dict, labels in loader:
        data_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data_dict.items()
        }
        labels = labels.to(device)

        logits = model(data_dict)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)
