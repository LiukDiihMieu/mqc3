---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  name: python3
  language: python
  display_name: python3
---

# Simulating Circuit

This section explains how to run a circuit locally with MQC-mini and visualize Gaussian states as Wigner functions. The current simulator accepts `CircuitRepr`; graph and machinery simulation is planned for a later stage.

(sec:simulation-configuring-circuit-representation)=

## Configuring circuit representation

First, we create the teleportation circuit below.

![teleportation](_images/circuit_repr_teleportation.svg)

```{code-cell}
# pyright: reportUnusedExpression=false

from math import pi

import numpy as np

from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops.intrinsic import Displacement, Measurement
from mqc3.circuit.ops.std import BeamSplitter
from mqc3.circuit.state import BosonicState, GaussianState
from mqc3.feedforward import feedforward


# Define feedforward functions.
@feedforward
def displace_x(x):
    from math import sqrt  # noqa:PLC0415

    return sqrt(2) * x


@feedforward
def displace_p(p):
    from math import sqrt  # noqa:PLC0415

    return -sqrt(2) * p


# Construct the teleportation circuit.
circuit = CircuitRepr("teleportation")
circuit.Q(1, 2) | BeamSplitter(theta=pi / 4, phi=pi)
circuit.Q(0, 1) | BeamSplitter(theta=pi / 4, phi=pi)
# Measure modes 0 and 1.
m0 = circuit.Q(0) | Measurement(theta=pi / 2)
m1 = circuit.Q(1) | Measurement(theta=0)
# Apply displacement with feedforward.
circuit.Q(2) | Displacement(displace_x(m0), displace_p(m1))
# Set initial states.
coeff = np.array([1.0 + 0.0j])
mode0_x = 3.0
mode0_p = -1.5
mean = np.array([mode0_x + 0.0j, mode0_p + 0.0j])
# Set the initial state of the teleported mode.
circuit.set_initial_state(0, BosonicState(coeff, [GaussianState(mean, GaussianState.vacuum().cov)]))
# Set the initial states of modes 1 and 2.
circuit.set_initial_state(1, BosonicState(coeff, [GaussianState.squeezed(r=1.5, phi=pi / 2)]))  # p-squeezed.
circuit.set_initial_state(2, BosonicState(coeff, [GaussianState.squeezed(r=1.5, phi=0)]))  # x-squeezed.

circuit
```

### Visualizing the initial state modes

A mode in a `BosonicState` object can be visualized as a wigner function, using `visualize_state` module.
This module provides the following functions.

* `meshed_wigner`: Calculate the Wigner function of a specified mode, which is discretized according to the specified mesh for quadrature variables x and p.
* `plot_wigner`: Plot the discretized Wigner function of a specified mode calculated by `meshed_wigner` internally.

In this section, the usage of `plot_wigner` function is demonstrated.
Now let's visualize the Wigner function of each mode in the circuit.

```{code-cell}
from numpy import linspace

from mqc3.circuit.visualize_state import make_wigner_figure

input_mode = circuit.get_initial_state(0)
assert isinstance(input_mode, BosonicState)
make_wigner_figure(state=input_mode, xvec=linspace(-10, 10, 401), pvec=linspace(-10, 10, 401))
```

```{code-cell}
p_squeezed_mode = circuit.get_initial_state(1)
assert isinstance(p_squeezed_mode, BosonicState)
make_wigner_figure(state=p_squeezed_mode, xvec=linspace(-10, 10, 401), pvec=linspace(-10, 10, 401))
```

```{code-cell}
x_squeezed_mode = circuit.get_initial_state(2)
assert isinstance(x_squeezed_mode, BosonicState)
make_wigner_figure(state=x_squeezed_mode, xvec=linspace(-10, 10, 401), pvec=linspace(-10, 10, 401))
```

## Configuring client

Create a local `SimulatorClient`. It runs on the forward-only PyTorch Gaussian
engine, which supports single-peak Gaussian initial states, Gaussian operations,
homodyne measurements, and feedforward.

The PyTorch backend defaults to `dtype="float64"` for numerical stability. Use `dtype="float32"` for lower-precision experiments. A seed makes homodyne samples reproducible.

```{code-cell}
from mqc3.client import SimulatorClient, SimulatorClientResult

client = SimulatorClient(dtype="float64", seed=1234)
```

The simulator runs synchronously on the local machine. Configure the number of independent circuit executions with `n_shots` and choose which final states to retain with `state_save_policy`.

```{code-cell}
# Run the circuit one hundred times and retain every final state.
client.n_shots = 100
client.state_save_policy = "all"
```

## Simulating circuit

You can simulate the circuit with `execute`.
This function takes a circuit and a client as arguments and returns the execution result.

```{code-cell}
from mqc3.execute import execute

result = execute(circuit, client)
```

```{code-cell}
type(result.client_result)
```

`SimulatorClientResult` contains local timing details, the circuit measurement result, and any states retained by `state_save_policy`. Its `execution_result` property returns the same `CircuitResult` exposed by `circuit_result`.

Because the current simulator accepts only circuit representations, `graph_result` and `machinery_result` are always `None`.

+++

When you simulate the circuit representation, you can get the measurement result of the circuit representation for any shot by using index access.

```{code-cell}
from pprint import pprint

pprint(result[0])  # Get the 0-th shot result
```

```{code-cell}
pprint(result[1])  # Get the 1-st shot result
```

The `index` parameter of each `CircuitOperationMeasuredValue` object represents the mode index.

You can access the quantum states after circuit simulation via `states`.

In this example, the executed circuit is a teleportation circuit that transfers the state of mode 0 to mode 2.  
We can verify whether the `mean` of mode 2 in the final state matches the initial `mean` of mode 0.

```{code-cell}
error_threshold = 1.0

assert isinstance(result.client_result, SimulatorClientResult)
for state in result.client_result.states:
    mean = state.get_gaussian_state(0).mean
    cov = state.get_gaussian_state(0).cov
    teleported_x = mean[2].real
    teleported_p = mean[5].real

    assert abs(teleported_x - mode0_x) < error_threshold
    assert abs(teleported_p - mode0_p) < error_threshold

print("Teleportation successful!")
```

### Visualizing the output mode

To see the effect of the teleportation protocol, we visualize the Wigner function of the output mode in the first shot.

```{code-cell}
client_result = result.client_result
assert isinstance(client_result, SimulatorClientResult)
final_state = client_result.states[0]
output_mode = final_state.extract_mode(2)
make_wigner_figure(state=output_mode, xvec=linspace(-10, 10, 401), pvec=linspace(-10, 10, 401))
```

Since the initial modes 1 and 2 have finite squeezing levels, output mode 2 differs slightly and randomly from input mode 0.
