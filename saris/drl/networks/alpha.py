import torch.nn as nn
import torch


class Alpha(nn.Module):
    """
    Temperature parameter for entropy.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.alpha = nn.Linear(in_features=1, out_features=1, bias=False)
        with torch.no_grad():
            weights = torch.tensor([temperature])
            weights = torch.reshape(weights, self.alpha.weight.shape)
            self.alpha.weight.copy_(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.alpha(x)
        x = torch.mean(x)
        x = torch.clip(x, 0.0001, 0.15)
        return x
