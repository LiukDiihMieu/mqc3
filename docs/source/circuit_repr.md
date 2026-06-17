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

# Constructing Circuit Representation

In this section, we explain how to construct quantum circuits with the "Circuit Representation".

## Creating an empty circuit

To construct a quantum circuit in the circuit representation, you first need to create an instance of the `CircuitRepr` class.
You can create an instance `c` with the name "empty" as shown below:

```{code-cell}
from mqc3.circuit import CircuitRepr

c = CircuitRepr("empty")
```

The assigned name can be retrieved using the `name` property:

```{code-cell}
print(c.name)
```

## Adding modes

When a circuit instance is created, it contains no modes or gates.
You can add modes using `Q`.
A newly added mode is initialized in the $x$-squeezed state, which serves as an input to the machinery of MQC3, as explained in {ref}`sec:theory`.

```{code-cell}
c = CircuitRepr("create_mode")
print("#modes =", c.n_modes)
c.Q(0)  # Create a mode 0.
print("#modes =", c.n_modes)
```

Since modes are indexed starting from `0`, creating a mode with id=`N` implicitly creates modes with IDs ranging from `0` to `N-1`.

```{code-cell}
c = CircuitRepr("create_modes")
print("#modes =", c.n_modes)
c.Q(9)  # Create a mode with id=9 implicitly creates modes with IDs 0 to 8 as well.
print("#modes =", c.n_modes)
```

## Adding gates

The available gates are defined in the `mqc3.circuit.ops.intrinsic` and `mqc3.circuit.ops.std` modules.

* The `intrinsic` module defines gates that can be executed efficiently on the machinery of MQC3.
* The `std` module provides commonly used gates in general quantum computing. These gates are implemented using intrinsic gates and are expanded into an array of intrinsic gates when the circuit is executed.

You can add gates to a circuit by specifying the modes to which they apply.
For instance, you can apply the `PhaseRotation` with $\phi = \pi$ to mode `0` of `c` as shown below:

![single mode gate](_images/circuit_repr_single_mode_gate.svg)

```{code-cell}
# pyright: reportUnusedExpression=false

from math import pi

from mqc3.circuit.ops.intrinsic import PhaseRotation

c = CircuitRepr("single_mode_gate")
c.Q(0) | PhaseRotation(phi=pi)

c
```

You can apply multiple gates to the same mode by chaining.
For example,

* `c.Q(0) | PhaseRotation(phi=pi / 2)` and `c.Q(0) | Displacement(1, 1)` apply two gates separately.
* `c.Q(0) | PhaseRotation(phi=pi / 2) | Displacement(1, 1)` chains them together in a single statement.

![chain of single mode gates](_images/circuit_repr_single_mode_gate_chain.svg)

```{code-cell}
# pyright: reportUnusedExpression=false

from mqc3.circuit.ops.intrinsic import Displacement

c = CircuitRepr("chain_of_single_mode_gates")
c.Q(0) | PhaseRotation(phi=pi / 2) | Displacement(1, 1)

c
```

For two-mode gates, you need to specify both modes (in this case, mode `0` and mode `1`):

![two-mode gate](_images/circuit_repr_two_mode_gate.svg)

```{code-cell}
# pyright: reportUnusedExpression=false

from mqc3.circuit.ops.intrinsic import ControlledZ

c = CircuitRepr("two_mode_gate")
c.Q(0, 1) | ControlledZ(g=1)

c
```

For each mode, gates are applied in the order they are written.

```{caution}
In circuit representations, it is assumed that no operation is applied to a mode once it has been measured. The behavior is undefined if any operation is applied after `mqc3.circuit.ops.intrinsic.Measurement` operation.
```

The following example demonstrates a circuit with multiple gates:

![sample circuit](_images/circuit_repr_sample_circuit.svg)

```{code-cell}
# pyright: reportUnusedExpression=false

from math import pi

from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops.intrinsic import ControlledZ, Displacement, Measurement, PhaseRotation

c = CircuitRepr("multiple_gates")
c.Q(0) | PhaseRotation(phi=pi / 4) | Displacement(1, -1)
c.Q(0, 1) | ControlledZ(g=1)
c.Q(0) | Measurement(theta=0)
c.Q(1) | Measurement(theta=pi / 2)

c
```

You can verify the gates have been added correctly by printing the circuit information:

```{code-cell}
print("name        =", c.name)
print("#modes      =", c.n_modes)
print("#operations =", c.n_operations)
print(c)
```

(sec:circuit-adding-feedforward)=

## Adding feedforward

In MQC3, the parameters of certain operations can depend on the measurement results of other modes.
This mechanism is referred to as **feedforward** in MQC3.

To use feedforward, follow these steps:

1. Define a feedforward function `f` (`FeedForwardFunction`)
    * Create a Python function that takes a float as input and returns a float.
    * Decorate the function with `feedforward`.
    * **Note:** The function must be **self-contained** and able to run independently.
        * If it depends on external modules such as `math` or `numpy`, import them **inside** the function.
2. Obtain the measurement result variable `m` (`MeasuredVariable`)
3. Use the feedforwarded parameter `f(m)` as a parameter of an operation.

For example, the teleportation circuit shown below can be implemented as follows:

![teleportation](_images/circuit_repr_teleportation.svg)

```{code-cell}
# pyright: reportUnusedExpression=false

from math import pi

from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops.intrinsic import Displacement, Measurement
from mqc3.circuit.ops.std import BeamSplitter
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
c = CircuitRepr("teleportation")
c.Q(1, 2) | BeamSplitter(theta=pi / 4, phi=pi)
c.Q(0, 1) | BeamSplitter(theta=pi / 4, phi=pi)
# Measure modes 0 and 1.
m0 = c.Q(0) | Measurement(theta=pi / 2)
m1 = c.Q(1) | Measurement(theta=0)
# Apply displacement with feedforward.
c.Q(2) | Displacement(displace_x(m0), displace_p(m1))

c
```

The created circuit can be visualized using the `make_figure` function.
For more details, see [Visualizing Circuit Representation](viz_circuit_repr.ipynb).

```{code-cell}
from mqc3.circuit.visualize import make_figure

make_figure(c)
```

## Initializing modes

The local `SimulatorClient` requires every circuit mode to have an explicitly defined initial state. Set one with `set_initial_state`.

The current local simulators accept a `BosonicState` containing exactly one Gaussian peak. The PyTorch backend supports real quadrature means and Gaussian circuit evolution. For an example, see {ref}`sec:simulation-configuring-circuit-representation`.
