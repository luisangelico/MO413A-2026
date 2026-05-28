"""GATv2 classifier for TCGA melanoma graphs."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm, global_mean_pool


class GATv2Classifier(torch.nn.Module):
    def __init__(self, hidden_channels=64, num_classes=3, heads=4, dropout=0.4, in_channels=1):
        super().__init__()
        self.config = {
            "hidden_channels": hidden_channels,
            "num_classes": num_classes,
            "heads": heads,
            "dropout": dropout,
            "in_channels": in_channels,
        }
        self.dropout = dropout
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads)
        self.norm1 = LayerNorm(hidden_channels * heads)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1)
        self.norm2 = LayerNorm(hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, return_attn=False):
        x, (edge_index_attn, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x, _ = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.norm2(x)
        x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x)

        if return_attn:
            return x, edge_index_attn, alpha
        return x


def save_model(model, path):
    path = Path(path)
    torch.save(model.state_dict(), path)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(model.config, f, indent=2)


def load_model(path, device=None):
    path = Path(path)
    config_path = path.with_suffix(".json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"hidden_channels": 64, "num_classes": 3, "heads": 4, "dropout": 0.4, "in_channels": 1}
    if device is None:
        from src.config import get_device
        device = get_device()
    model = GATv2Classifier(**config).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, config
