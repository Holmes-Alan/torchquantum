# 👋 Welcome

#### Forked from mit-han-lab/torchquantum:main

## Installation
```bash
git clone [https://github.com/mit-han-lab/torchquantum.git](https://github.com/Holmes-Alan/torchquantum.git)
cd torchquantum
pip install --editable .
pip install qiskit_ibm_runtime==0.20.0
pip install -U qiskit-aer
```

## Basic Usage

```python
import torchquantum as tq
import torchquantum.functional as tqf

qdev = tq.QuantumDevice(n_wires=2, bsz=5, device="cpu", record_op=True) # use device='cuda' for GPU

# use qdev.op
qdev.h(wires=0)
qdev.cnot(wires=[0, 1])

# use tqf
tqf.h(qdev, wires=1)
tqf.x(qdev, wires=1)

# use tq.Operator
op = tq.RX(has_params=True, trainable=True, init_params=0.5)
op(qdev, wires=0)

# print the current state (dynamic computation graph supported)
print(qdev)

# obtain the qasm string
from torchquantum.plugin import op_history2qasm
print(op_history2qasm(qdev.n_wires, qdev.op_history))

# measure the state on z basis
print(tq.measure(qdev, n_shots=1024))

# obtain the expval on a observable by stochastic sampling (doable on simulator and real quantum hardware)
from torchquantum.measurement import expval_joint_sampling
expval_sampling = expval_joint_sampling(qdev, 'ZX', n_shots=1024)
print(expval_sampling)

# obtain the expval on a observable by analytical computation (only doable on classical simulator)
from torchquantum.measurement import expval_joint_analytical
expval = expval_joint_analytical(qdev, 'ZX')
print(expval)

# obtain gradients of expval w.r.t. trainable parameters
expval[0].backward()
print(op.params.grad)


# Apply gates to qdev with tq.QuantumModule
ops = [
    {'name': 'hadamard', 'wires': 0},
    {'name': 'cnot', 'wires': [0, 1]},
    {'name': 'rx', 'wires': 0, 'params': 0.5, 'trainable': True},
    {'name': 'u3', 'wires': 0, 'params': [0.1, 0.2, 0.3], 'trainable': True},
    {'name': 'h', 'wires': 1, 'inverse': True}
]

qmodule = tq.QuantumModule.from_op_history(ops)
qmodule(qdev)
```


<!--
## Basic Usage 2

```
python QuantumNAS_main.py
```
 
## Citation
```
@inproceedings{hanruiwang2022quantumnas,
    title     = {Quantumnas: Noise-adaptive search for robust quantum circuits},
    author    = {Wang, Hanrui and Ding, Yongshan and Gu, Jiaqi and Li, Zirui and Lin, Yujun and Pan, David Z and Chong, Frederic T and Han, Song},
    booktitle = {The 28th IEEE International Symposium on High-Performance Computer Architecture (HPCA-28)},
    year      = {2022}
}
```
