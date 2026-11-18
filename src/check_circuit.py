"""Phase 1.3: check the closed-form QLIF update against a PennyLane circuit simulation.

The circuit: RX(phi) restores the previous population; then either RX(theta) (spike)
or an amplitude-damping channel with gamma = 1 - exp(-tau/T1) (idle, true T1 decay).
Run: .venv/bin/python src/check_circuit.py
"""
import itertools
import numpy as np
import pennylane as qml
import torch
from neurons import qlif_update

dev = qml.device("default.mixed", wires=1)
T1 = 60.0


@qml.qnode(dev)
def circuit(alpha, theta, tau, spike):
    qml.RX(2 * np.arcsin(np.sqrt(alpha)), wires=0)  # memory
    if spike:
        qml.RX(theta, wires=0)
    else:
        qml.AmplitudeDamping(1 - np.exp(-tau / T1), wires=0)
    return qml.probs(wires=0)


def formula(alpha, theta, tau, spike, leak):
    t = lambda v: torch.tensor([float(v)], dtype=torch.float64)
    # threshold 2.0 so the reset never triggers; we only compare the dynamics
    return qlif_update(t(theta), t(tau), t(spike), t(alpha), 2.0, T1, leak).item()


worst = {"physical": 0.0, "paper": 0.0}
print(f"{'alpha':>6} {'theta':>6} {'tau':>6} {'spike':>5} | {'circuit':>8} {'physical':>8} {'paper':>8}")
for alpha, theta, tau, spike in itertools.product([0.1, 0.5, 0.7], [0.3, 1.2], [0.0, 6.0, 60.0, 600.0], [1, 0]):
    if spike and tau != 0.0:
        continue  # tau is unused on a spike step
    c = circuit(alpha, theta, tau, spike)[1]
    row = {k: formula(alpha, theta, tau, spike, k) for k in worst}
    for k in worst:
        worst[k] = max(worst[k], abs(row[k] - c))
    print(f"{alpha:6.2f} {theta:6.2f} {tau:6.1f} {spike:5d} | {c:8.4f} {row['physical']:8.4f} {row['paper']:8.4f}")

print(f"\nmax |formula - circuit|: physical leak = {worst['physical']:.2e}, paper leak (Eq. 8-9) = {worst['paper']:.2e}")
assert worst["physical"] < 1e-5, "physical-leak formula disagrees with the circuit"
print("CHECKPOINT 1 (circuit): physical-leak formula matches the circuit.")
