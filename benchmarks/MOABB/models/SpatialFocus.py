import torch
import torch.nn as nn

from torch_geometric.data import Data

class SpatialFocus(nn.Module):
    def __init__(self, projection_dim, position_dim=3, tau=1.0, sigma=0.0):
        super().__init__()
        self.projection_dim = projection_dim
        self.position_dim = position_dim
        self.tau = tau
        self.sigma = sigma

        self.similarity_module = nn.Sequential(
            nn.Linear(position_dim, projection_dim),
            nn.ELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x: Data):
        if self.training and self.sigma > 0:
            x.pos = x.pos + torch.randn_like(x.pos) * self.sigma
        weights = self.similarity_module(x.pos)
        weights = self.softmax(weights / self.tau)
        x = torch.einsum("...cf, cd -> ...df", x, weights)
        return x
