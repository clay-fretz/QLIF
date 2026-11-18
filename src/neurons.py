"""Neuron models for the QLIF / QSNN project (Phase 1).

- make_lif: classical LIF (snnTorch Leaky) with the arctan surrogate.
- QLIF: quantum LIF neuron from Brand & Petruccione (arXiv:2407.16398), Eq. 5-9.

QLIF follows the authors' interface: per step it takes a rotation angle theta,
a delay tau and a binary "spike" bit x (1 = apply RX(theta), 0 = idle for tau).
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

EPS = 1e-6  # keeps sqrt/arcsin away from their infinite-gradient points (0 and 1)


def make_lif(beta=0.9, threshold=1.0):
    """Classical LIF neuron: U[t] = beta*U[t-1] + I[t] - S[t-1]*threshold."""
    return snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.atan())


class QLIF(nn.Module):
    """Quantum leaky integrate-and-fire neuron, simulated in closed form.

    leak="paper"    : no-spike step uses Eq. 8-9 as printed (what the authors' code does).
                      Note: tau=0 empties the neuron and large tau keeps it full.
    leak="physical" : no-spike step is true T1 relaxation, alpha * exp(-tau / T1).

    grad="surrogate": arctan surrogate centred on the threshold, with gradients
                      flowing through the spikes and through time (BPTT).
    grad="authors"  : reproduces the backward pass of the authors' alpha.py:
                      spike gradient dropped, no gradient through time.
    """

    def __init__(self, threshold=0.75, T1=60.0, leak="paper", grad="surrogate"):
        super().__init__()
        assert leak in ("paper", "physical") and grad in ("surrogate", "authors")
        self.threshold, self.T1, self.leak, self.grad = threshold, T1, leak, grad
        self.spike_fn = surrogate.atan()

    def forward(self, theta, tau, x, mem):
        """One time step. Returns (spike, new excited-state population)."""
        if self.grad == "authors":
            return _AuthorsQLIF.apply(theta, tau, x.float(), mem, self.threshold, self.T1, self.leak)
        new = qlif_update(theta, tau, x.float(), mem, self.threshold, self.T1, self.leak)
        spk = self.spike_fn(new - self.threshold)
        return spk, new


def qlif_update(theta, tau, x, mem, threshold, T1, leak):
    """Excited-state population after one step (Eq. 5-9), including reset.

    A neuron that was at or above threshold last step is reset to |0> (alpha = 0),
    exactly as in the authors' code.
    """
    fired = mem >= threshold
    m = mem.clamp(EPS, 1 - EPS)
    phi = torch.where(fired, torch.zeros_like(m), 2 * torch.asin(torch.sqrt(m)))  # memory rotation, Eq. 5
    spiked = torch.sin((theta + phi) / 2) ** 2                                     # Eq. 6
    if leak == "paper":
        gamma = -2 * torch.asin(torch.sqrt((m * torch.exp(-tau / T1)).clamp(EPS, 1 - EPS)))  # Eq. 8
        idle = torch.sin((gamma + phi) / 2) ** 2
    else:
        idle = m * torch.exp(-tau / T1)
    new = x * spiked + (1 - x) * idle                                              # Eq. 9
    return torch.where(fired, torch.zeros_like(new), new)


class _AuthorsQLIF(torch.autograd.Function):
    """Same forward as qlif_update; backward copies the logic of the authors' alpha.py.

    Their backward ignores the incoming spike gradient, so a layer only learns
    through the membrane output it returns. Earlier layers, which pass on only
    spikes, receive no gradient.
    """

    @staticmethod
    def forward(ctx, theta, tau, x, mem, threshold, T1, leak):
        new = qlif_update(theta, tau, x, mem, threshold, T1, leak)
        ctx.save_for_backward(theta, tau, x, new)
        ctx.T1 = T1
        return (new > threshold).float(), new

    @staticmethod
    def backward(ctx, grad_spike, grad_mem_out):
        theta, tau, x, mem = ctx.saved_tensors
        T1 = ctx.T1
        g = grad_mem_out / (1 + (torch.pi * mem) ** 2)  # arctan surrogate on alpha, not centred on threshold
        m = mem.clamp(EPS, 1 - EPS)
        me = (m * torch.exp(-tau / T1)).clamp(EPS, 1 - EPS)
        dtheta = x * torch.sin(x * (theta + 2 * torch.asin(torch.sqrt(m)))) / 2
        dtau = ((x - 1) * torch.sqrt(me) * torch.sin(2 * (x - 1) * (torch.asin(torch.sqrt(m)) - torch.asin(torch.sqrt(me))))
                / (2 * T1 * torch.sqrt(1 - me)))
        # grad_spike is deliberately unused, and no gradient goes to the previous mem: as in the authors' code.
        return dtheta * g, dtau * g, None, None, None, None, None
