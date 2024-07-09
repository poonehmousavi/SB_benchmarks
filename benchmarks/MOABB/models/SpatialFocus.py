import torch
import torch.nn as nn


class SpatialFocus(nn.Module):
    def __init__(
        self,
        n_focal_points,
        focus_dims=3,
        similarity_func="cosine",
        similarity_transform=nn.Softmax(0),
    ):
        super().__init__()
        self.n_focal_points = n_focal_points
        self.focus_dims = focus_dims
        self.similarity_func = similarity_func
        self.similarity_transform = similarity_transform

        self.focal_points = nn.Embedding(
            num_embeddings=n_focal_points, embedding_dim=self.focus_dims
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        focal_points = self.focal_points.weight

        if self.similarity_func == "cosine":
            similarity = nn.functional.cosine_similarity(
                positions.unsqueeze(1), focal_points.unsqueeze(0), dim=-1
            )
        else:
            similarity = self.similarity_func(positions, focal_points)

        if self.similarity_transform:
            weights = self.similarity_transform(similarity)
        else:
            weights = similarity

        x = torch.einsum("...cf, cd -> ...df", x, weights)
        return x
