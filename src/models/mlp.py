import numpy as np
import torch
from torch import ge, nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MLP(pl.LightningModule):
    def __init__(self, d, hidden_size, axes, output_size=1, lr=1e-4, hidden_layers=5):
        super().__init__()
        layers = np.array(
            [nn.Linear(hidden_size, hidden_size) for i in range(hidden_layers)]
        )
        layers = np.insert(layers, np.arange(1, hidden_layers + 1), Swish())
        self.output = nn.Sequential(
            nn.Linear(d, hidden_size),
            nn.LayerNorm(hidden_size),
            Swish(),
            *layers,
            nn.Linear(hidden_size, output_size),
            nn.Softplus()
        )
        self.axes = axes
        self.lr = lr

    def compute_normalization(self, axis):
        n_dim = len(axis)
        grid = (
            torch.from_numpy(np.array(np.meshgrid(*axis)).reshape(n_dim, -1).T)
            .float()
            
        )
        value = self.unnormalized_pass(grid)
        shape = []
        for i in range(n_dim):
            shape.append(len(axis[i]))
        norm = value.T.reshape(shape)
        for i in range(n_dim):
            norm = torch.trapz(norm, x=axis[i], axis=0)
        return norm

    def unnormalized_pass(self, x):
        return self.output(x)

    def forward(self, x, axis):
        return self.output(x) / self.compute_normalization(axis)

    def log_prob(self, x):
        return self(x, self.axes).log()

    def step(self, batch, batch_idx):
        (x,) = batch
        output = self(x, self.axes)
        geometric_term = torch.sum(torch.log(torch.mean(output, dim=0)), dim=0)
        loss = -geometric_term
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# class MLP(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size, hidden_layers=5):
#         super().__init__()
#         layers = np.array(
#             [nn.Linear(hidden_size, hidden_size) for i in range(hidden_layers)]
#         )
#         layers = np.insert(layers, np.arange(1, hidden_layers + 1), Swish())
#         self.output = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.LayerNorm(hidden_size),
#             Swish(),
#             *layers,
#             nn.Linear(hidden_size, output_size),
#             nn.Softplus()
#         )

#     def compute_normalization(self, axis):
#         n_dim = len(axis)
#         grid = (
#             torch.from_numpy(np.array(np.meshgrid(*axis)).reshape(n_dim, -1).T)
#             .float()
#             .cuda()
#         )
#         value = self.unnormalized_pass(grid)
#         shape = []
#         for i in range(n_dim):
#             shape.append(len(axis[i]))
#         norm = value.T.reshape(shape)
#         for i in range(n_dim):
#             norm = torch.trapz(norm, x=torch.from_numpy(axis[i]).float().cuda(), axis=0)
#         return norm

#     def unnormalized_pass(self, x):
#         return self.output(x)

#     def forward(self, x, axis):
#         return self.output(x) / self.compute_normalization(axis)
