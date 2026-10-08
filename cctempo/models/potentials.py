"""Potential (landscape) networks."""
import torch
from torch import nn


def _act(name):
    return {"softplus": nn.Softplus, "relu": nn.LeakyReLU, "tanh": nn.Tanh, "silu": nn.SiLU}[name]


class PotentialMLP(nn.Module):
    """PRESCIENT-style scalar potential U(x): Linear -> act (x layers) -> Linear(no bias, zero init)."""

    def __init__(self, in_dim, hidden=500, layers=1, act="softplus"):
        super().__init__()
        mods, d = [], in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), _act(act)()]
            d = hidden
        self.body = nn.Sequential(*mods)
        self.out = nn.Linear(d, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x):
        return self.out(self.body(x)).squeeze(-1)


class FiLMPotential(nn.Module):
    """Signal-conditioned landscape U_c(x; s).

        h = act(W1 x + b1)
        U_c = w^T [ h * (1 + tanh(G s)) + B s ] + s^T A x

    G/B modulate the shape of the landscape (feature-wise linear modulation) and A is a
    signal-dependent linear tilt -- the same role the morphogen parameters (a, b) play in the
    heteroclinic-flip landscape of Cislo et al. (PNAS 2025). With s = 0 the term reduces to an
    ordinary potential, so 'CC-off' controls keep identical capacity.
    """

    def __init__(self, x_dim, s_dim, hidden=256, act="softplus"):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(x_dim, hidden), _act(act)())
        self.gamma = nn.Linear(s_dim, hidden, bias=False)
        self.beta = nn.Linear(s_dim, hidden, bias=False)
        self.out = nn.Linear(hidden, 1, bias=False)
        self.A = nn.Parameter(torch.zeros(s_dim, x_dim))
        nn.init.zeros_(self.out.weight)
        nn.init.normal_(self.gamma.weight, std=0.01)
        nn.init.normal_(self.beta.weight, std=0.01)

    def forward(self, x, s):
        h = self.h(x)
        mod = h * (1.0 + torch.tanh(self.gamma(s))) + self.beta(s)
        return self.out(mod).squeeze(-1) + ((s @ self.A) * x).sum(-1)
