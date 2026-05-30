"""GATv2 classifier for TCGA melanoma graphs."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm, global_mean_pool


class GATv2Classifier(torch.nn.Module):
    def __init__(self, hidden_channels=128, num_classes=3, heads=4, dropout=0.4, in_channels=1):
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
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads)
        self.norm2 = LayerNorm(hidden_channels * heads)
        self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1)
        self.norm3 = LayerNorm(hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, return_attn=False, return_embedding=False,
                return_all_attn=False):
        x, (ei1, a1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x, (ei2, a2) = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.norm2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x, (ei3, a3) = self.conv3(x, edge_index, return_attention_weights=True)
        x = self.norm3(x)
        x = F.elu(x)

        embedding = global_mean_pool(x, batch)
        logits = self.lin(embedding)

        out = (logits,)
        if return_attn:
            # back-compat: layer-1 (edge_index, alpha) tuple
            out = out + (ei1, a1)
        if return_all_attn:
            # list of (edge_index, alpha) per layer; alpha shape: [E, heads]
            out = out + ([(ei1, a1), (ei2, a2), (ei3, a3)],)
        if return_embedding:
            out = out + (embedding,)
        return out[0] if len(out) == 1 else out


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
