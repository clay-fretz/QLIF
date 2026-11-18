"""Phase 1 self-checks for neurons.py. Run: .venv/bin/python src/test_neurons.py

1. QLIF fires and resets; LIF fires.
2. Finding 1 from Phase 0: with the authors' backward pass, the hidden layer of a
   2-layer QSNN gets zero gradient; with the surrogate version it does not.
"""
import torch
import torch.nn as nn
from neurons import QLIF, make_lif

torch.manual_seed(0)

# 1. Basic behaviour: repeated spikes push alpha over threshold, then it resets.
q = QLIF(threshold=0.75)
mem = torch.zeros(1)
spikes = []
for _ in range(6):
    s, mem = q(torch.tensor([0.8]), torch.tensor([0.0]), torch.tensor([1]), mem)
    spikes.append(int(s.item()))
assert 1 in spikes, f"QLIF never fired: {spikes}"
first = spikes.index(1)
assert spikes[first + 1] == 0, "QLIF did not reset after firing"
print("QLIF spikes over 6 input steps:", spikes)

lif = make_lif(beta=0.9, threshold=1.0)
m = lif.init_leaky()
lif_spk = []
for _ in range(6):
    s, m = lif(torch.tensor([0.5]), m)
    lif_spk.append(int(s.item()))
assert 1 in lif_spk
print("LIF  spikes over 6 input steps:", lif_spk)


# 2. Gradient flow, same layout as the authors' fully connected network.
def hidden_grad(grad_mode, steps=25, B=8):
    torch.manual_seed(0)
    fcq1, fct1 = nn.Linear(784, 100), nn.Linear(784, 100)
    fcq2, fct2 = nn.Linear(100, 10), nn.Linear(100, 10)
    n1, n2 = QLIF(grad=grad_mode), QLIF(grad=grad_mode)
    x, y = torch.rand(B, 784), torch.randint(0, 10, (B,))
    m1, m2, loss = torch.zeros(B, 100), torch.zeros(B, 10), 0
    for _ in range(steps):
        s1, m1 = n1(fcq1(x).abs() * torch.pi, fct1(x).abs() * 200, (fcq1(x) > 0), m1)
        s2, m2 = n2(fcq2(s1).abs() * torch.pi, fct2(s1).abs() * 200, (fcq2(s1) > 0), m2)
        loss = loss + nn.functional.cross_entropy(m2, y)
    loss.backward()
    return fcq1.weight.grad.abs().sum().item(), fcq2.weight.grad.abs().sum().item()


h_a, o_a = hidden_grad("authors")
h_s, o_s = hidden_grad("surrogate")
print(f"authors'  backward: hidden-layer grad = {h_a:.3e}, output-layer grad = {o_a:.3e}")
print(f"surrogate backward: hidden-layer grad = {h_s:.3e}, output-layer grad = {o_s:.3e}")
assert h_a == 0.0 and o_a > 0, "expected the authors' backward to train only the output layer"
assert h_s > 0, "surrogate version should train the hidden layer"
print("CHECKPOINT 1 (neurons): all checks passed.")
