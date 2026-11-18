# QLIF: Quantum Leaky Integrate-and-Fire Neurons

Course final project. We recreate parts of the paper *"A Quantum Leaky Integrate-and-Fire Spiking Neuron and Network"* by Brand and Petruccione ([arXiv:2407.16398](https://arxiv.org/abs/2407.16398)), then try one change of our own.

**The plan**
1. Compare a normal LIF neuron with the quantum QLIF neuron (the paper's Fig. 4).
2. Train small spiking networks on MNIST and compare accuracy (the paper's Table I): SNN vs QSNN, and SCNN vs QSCNN.
3. Our own experiment: give the QLIF neuron an adaptive threshold and see how it changes accuracy.

## What is done so far

**Neurons (`src/neurons.py`)**
- `make_lif()`: a normal LIF neuron (from snnTorch).
- `QLIF`: the quantum neuron. It is one qubit:
  - an input spike rotates the qubit, which raises its "membrane" (excited-state population);
  - no input lets the qubit relax back towards 0, which is the "leak";
  - when the membrane passes the threshold, the neuron spikes and resets to 0.

  It has two switches:
  - `leak="paper"` follows the paper's equations; `leak="physical"` uses real qubit decay.
  - `grad="surrogate"` lets every layer learn; `grad="authors"` copies how the authors' code trains.

**Checks**
- `src/check_circuit.py` compares our formula with a real quantum circuit simulation (PennyLane). The "physical" version matches it exactly. The paper's version does not: with no delay, the paper's equation empties the neuron when it should stay unchanged.
- `src/test_neurons.py` checks that the neurons fire and reset. It also shows that, trained the way the authors' code trains, only the last layer of the network learns; the earlier layers get no gradient.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd src
python test_neurons.py
python check_circuit.py
```

## Next

Recreate Fig. 4 (LIF vs QLIF), then train the networks.
