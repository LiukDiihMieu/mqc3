# MQC-mini

**Simulation-only SDK for Measurement-based Quantum Computation with Continuous-variable Cluster states.**

MQC-mini is a software development kit for optical quantum computers.
It helps you design CV measurement‑based programs and run them on a local simulator.

> **MQC-mini** is a simulation-only fork of MQC3: the cloud connection and
> real-hardware execution paths have been removed. Programs run locally via the
> PyTorch Gaussian backend.

The PyTorch backend is a forward-only Gaussian circuit simulator. It evolves
single-peak Gaussian states through Gaussian gates and homodyne measurements
using dense PyTorch linear algebra.

> Looking for end‑user docs? See the user guide at [docs/source/index.md](docs/source/index.md).

## Requirements

- **Supported Platforms**
  - Linux (Ubuntu 24.04 LTS recommended; WSL2 on Windows 11 is also supported)
  - Windows 11
  - macOS (Apple silicon, ARM64)
- **Python versions:** 3.10, 3.11, 3.12, 3.13

## Installation

Install the library (PyTorch, the simulator backend, is a core dependency):

```sh
python -m pip install .
```

By default, only a minimal set of features is installed (e.g., visualization may be unavailable).
To enable all optional features:

```sh
python -m pip install "<path/to/sdk>[all]"
```

To install tools for tests and docs:

```sh
python -m pip install "<path/to/sdk>[dev]"
```

### CPU-only PyTorch

By default `pip` installs the standard PyTorch build (which may pull large CUDA
wheels). For a CPU-only setup, install the PyTorch CPU wheel first, then
MQC-mini:

```sh
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install "<path/to/sdk>"
```

Select the floating-point precision through `SimulatorClient`:

```python
from mqc3.client import SimulatorClient

client = SimulatorClient(dtype="float64", seed=1234)
```

The PyTorch backend defaults to `float64` for numerical stability. `float32`
is available for experiments with less strongly squeezed states. It currently
runs on CPU and supports `CircuitRepr` programs with single-peak Gaussian
initial states, Gaussian operations, homodyne measurements, and feedforward.

## Quickstart

After installation, try a minimal program:

```python
from math import pi
from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops import intrinsic
from mqc3.circuit.state import BosonicState
from mqc3.client import SimulatorClient

# Create a circuit representation of a program
c = CircuitRepr("sample_circuit")
c.Q(0) | intrinsic.PhaseRotation(phi=pi / 2)  # Apply a phase rotation of pi/2 to qumode 0
c.Q(0) | intrinsic.Measurement(theta=0.0)     # Measure qumode 0 (homodyne)
c.set_initial_state(0, BosonicState.vacuum())

client = SimulatorClient(n_shots=10, seed=1234)
result = client.run(c)
print(result.circuit_result)
```

See the user docs for details: [docs/source/index.md](docs/source/index.md).

## Tests

**Some test suites are long‑running. Use `pytest-xdist` to parallelize.**

```sh
# basic tests
pytest

# tests requiring simulator access
pytest --simulator

# long‑running tests (parallel)
pytest -n auto --longrun

# everything
pytest -n auto --longrun --simulator
```

## Project layout

```text
├── docs/
├── src/
│   └── mqc3/
│       ├── circuit/
│       ├── client/
│       ├── execute/
│       ├── feedforward/
│       ├── graph/
│       ├── machinery/
│       └── pb/
└── tests/
```

- `docs/` : User & developer documentation sources.
- `src/mqc3/` : Main source tree.
  - `circuit/` : Circuit representation.
  - `client/` :  Local simulator client and result types.
  - `execute/` : Unified wrapper for running representations with a client.
  - `feedforward/` : Mechanisms to update operation parameters conditioned on measurement outcomes.
  - `graph/` : Graph representation.
  - `machinery/` : Machinery representation.
  - `pb/` : Protocol Buffers (auto-generated).
- `tests/` : Unit and integration tests.

## License

MIT License. See [LICENSE](LICENSE).
