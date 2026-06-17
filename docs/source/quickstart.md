---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  name: mqc3
  language: python
  display_name: mqc3
---

# Quickstart

This section introduces MQC-mini through a local circuit-simulation example.

MQC-mini retains three representations of optical quantum programs: **circuit representation**, **graph representation**, and **machinery representation**. The current local simulators execute circuit representations. Graph and machinery representations remain available for constructing, transforming, and visualizing measurement-based programs; support for simulating them is planned for a later stage.

| Representation | Brief description |
| --- | --- |
| Circuit representation | A continuous-variable quantum circuit consisting of modes and operations. This is the representation accepted by the current local simulators. See [Constructing Circuit Representation](circuit_repr.ipynb). |
| Graph representation | A two-dimensional measurement-based representation. See [Constructing Graph Representation](graph_repr.ipynb). |
| Machinery representation | A time-ordered representation derived from a graph. See [Constructing Machinery Representation](machinery_repr.ipynb) and [Theory](theory.md). |

## Example circuit

The image below shows a two-mode circuit with Gaussian operations and homodyne measurements. See [Gates](gates.md) for operation definitions.

![circuit](_images/circuit_repr_sample_circuit.svg)

+++

## Creating a circuit

Let's create a single-mode circuit and run it on the local PyTorch Gaussian simulator. This circuit performs a phase rotation and homodyne measurement on mode `0`.

![quickstart_example](_images/quickstart_example.svg)

First, import the necessary packages.

```{code-cell}
from math import pi

import matplotlib.pyplot as plt
import numpy as np

from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops import intrinsic
from mqc3.circuit.state import BosonicState
from mqc3.client import SimulatorClient
from mqc3.execute import execute
```

You can create a circuit using `CircuitRepr`.
A mode can be created using `Q`.
Operations are created using `PhaseRotation` and `Measurement`.

```{code-cell}
circuit = CircuitRepr("example")
circuit.Q(0) | intrinsic.PhaseRotation(pi / 4)
circuit.Q(0) | intrinsic.Measurement(pi / 4)
circuit.set_initial_state(0, BosonicState.squeezed(r=1.0, phi=pi / 2))

circuit
```

## Creating a client

Create a local `SimulatorClient` and select a simulator backend.

The `"torch"` backend is a forward-only Gaussian simulator implemented with PyTorch. It supports single-peak Gaussian states, Gaussian operations, homodyne measurements, and feedforward. Use `dtype="float64"` for the numerical-stability baseline; `float32` is also available.

```{code-cell}
client = SimulatorClient(n_shots=1000, backend="torch", dtype="float64", seed=1234)
```

## Simulating the circuit

Run the circuit using `execute`. The call is synchronous and returns after local simulation completes. You can also call `client.run(circuit)` directly.

```{code-cell}
result = execute(circuit, client)
```

## Checking the simulation result

The result is returned as an instance of `ExecutionResult`.

```{code-cell}
from pprint import pprint

pprint(result)
```

You can get the execution result of the 0th shot as shown below.

```{code-cell}
pprint(result[0])
```

## Checking the rotation result

As an example of a slightly more complex code, the following example demonstrates how to validate the previously mentioned phase rotation.
This time, the mode after the rotation is measured at various angles ($\theta=0,\frac{\pi}{8},\frac{\pi}{4},\dots,\pi$), and the variance of the measured values is calculated for each angle.

![quickstart_multi_param](_images/quickstart_multi_param.svg)

```{code-cell}
def construct_circuit(theta: float) -> CircuitRepr:
    circuit = CircuitRepr(f"theta={theta}")
    circuit.Q(0) | intrinsic.PhaseRotation(pi / 4)
    circuit.Q(0) | intrinsic.Measurement(theta)
    circuit.set_initial_state(0, BosonicState.squeezed(r=1.0, phi=pi / 2))

    return circuit


# Define measurement angles: 0, pi/8, pi/4, ..., pi.
angle_list = np.linspace(0, pi, 9)

var_list = []

for theta in angle_list:
    circuit = construct_circuit(theta)
    result = execute(circuit, client)

    # Collect the measured values for mode 0.
    mode0_values = [smv[0].value for smv in result]

    # Calculate and store the variance.
    var_list.append(np.var(mode0_values))

print(var_list)
```

### Plotting the variance

The mode is initially in an $x$-squeezed state (squeezing angle=$\frac{\pi}{2}$).
After the $\frac{\pi}{4}$ rotation, the state acquires a squeezing angle of $\frac{3\pi}{4}$.
Therefore, when measured in the direction of $\frac{3\pi}{4}$, the variance is minimized, whereas in the direction of $\frac{\pi}{4}$, the variance is maximized.

```{code-cell}
angle_labels = [
    r"$0$",
    r"$\frac{\pi}{8}$",
    r"$\frac{\pi}{4}$",
    r"$\frac{3\pi}{8}$",
    r"$\frac{\pi}{2}$",
    r"$\frac{5\pi}{8}$",
    r"$\frac{3\pi}{4}$",
    r"$\frac{7\pi}{8}$",
    r"$\pi$",
]

plt.figure(figsize=(10, 6))
plt.plot(angle_list, var_list, "ro-", label="Mode 0")

plt.xlabel(r"Measurement angle")
plt.ylabel("Variance")
plt.xticks(angle_list, angle_labels)
plt.legend()
plt.show()
```
