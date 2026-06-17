---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: mqc3
  language: python
  name: mqc3
---

# Constructing Graph Representation

In this section, we will explain how to construct quantum circuits with the "Graph Representation".
We recommend reading [Theory](theory.md) before reading this document.

+++

## Creating a graph representation

Constructing a quantum circuit in graph representation starts with creating an instance of the `GraphRepr`.
Graph representation is created by specifying the following three parameters.

* `n_local_macronodes` : the number of macronodes per step
* `n_steps` : the number of steps

For example, graph representation with `n_local_macronodes=3` and `n_steps=4` can be created as follows.
Every macronode is initialized with an operation `Wiring` which is configured to pass through each mode straightforward.

```{code-cell} ipython3
from mqc3.graph import GraphRepr

g = GraphRepr(3, 4)
```

You can visualize graph representation using `make_figure`.
See [Visualizing Graph Representation](viz_graph_repr.md) for detailed usage of `make_figure`.

```{code-cell} ipython3
from mqc3.graph.visualize import make_figure

make_figure(g)
```

All macronodes are assigned coordinates and indices, as shown in the image below.
The top-left macronode is located at coordinates `(0, 0)`, while the bottom-right macronode is located at `(n_local_macronodes - 1, n_steps - 1)`.
To get a macronode's coordinate, call `get_coord` with its index `i`.
To find the index of a macronode at a given coordinate `(h, w)`, use `get_index`.
The edge from the bottom-most macronode to the top-most macronode is not a special edge.
Let `N` be the number of local macronodes and `M` be the number of steps.
Macronode `k` has an edge to the macronode `k + 1` and `k + N`.
When this directed graph is written in two dimensions, the edge from macronode `i * N - 1` to macronode `i * N` (`i = 0, 1, ..., M - 2`) is an edge from the bottom to the top.

<img src="_images/graph_repr_coord.png" width=500px>

```{code-cell} ipython3
print(g.get_coord(0))
print(g.get_coord(11))
print(g.get_index(0, 0))
print(g.get_index(2, 3))
```

## Placing operations

You can place operations using the `place_operation`.
Graph operations are defined in `mqc3.graph.ops`.
See [Gates](gates.md) for the documentation.
In the graph representation, displacement is not a standalone operation, but it can be associated with any operation and applied to the output modes.
The displacement of a mode teleporting to the macronode `k` from the macronode `k - 1` can be specified by `dislacement_k_minus_1` argument in the operation's constructor.
Similary, the displacement of a mode teleporting to the macronode `k` from the macronode `k - n_local_macrondoes` can be specified by `dislacement_k_minus_n` argument in the operation's constructor.

When applying an operation on a mode, you first need to initialize the mode with the `Initialization`.

```{code-cell} ipython3
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.ops import Initialization

g = GraphRepr(1, 2)
# initialized_modes=(BLANK_MODE, 0) means that the mode teleporting to the macronode `k + 1` is `BLANK_MODE`,
# and the mode teleporting to the macronode `k + N` is mode `0`.
# `BLANK_MODE` means that this mode is not the target of any operations.
g.place_operation(Initialization(macronode=(0, 0), theta=0, initialized_modes=(BLANK_MODE, 0)))
make_figure(g)
```

The circuit below can be implemented as follows:

![graph_repr_example](_images/graph_repr_example.svg)

```{code-cell} ipython3
from numpy import pi

from mqc3.graph.ops import Initialization, Measurement, PhaseRotation

g1 = GraphRepr(1, 3)

# Displacement is specified by `displacement_k_minus_n`.
g1.place_operation(Initialization(macronode=(0, 0), theta=0, initialized_modes=(BLANK_MODE, 0)))
g1.place_operation(PhaseRotation(macronode=(0, 1), phi=pi / 4, swap=False, displacement_k_minus_n=(-1, 0)))
g1.place_operation(Measurement(macronode=(0, 2), theta=pi / 2, displacement_k_minus_n=(0, 1)))

make_figure(g1, scale=4.0)
```

A graph representation allows an equivalent circuit to be implemented in multiple ways.
For example, the circuit from earlier can be implemented as follows.

```{code-cell} ipython3
from numpy import pi

from mqc3.graph import Wiring
from mqc3.graph.ops import Initialization, Measurement, PhaseRotation

g2 = GraphRepr(2, 3)

g2.place_operation(Initialization(macronode=(0, 0), theta=0, initialized_modes=(BLANK_MODE, 0)))
g2.place_operation(PhaseRotation(macronode=(0, 1), phi=pi / 4, swap=True, displacement_k_minus_n=(-1, 0)))
g2.place_operation(Wiring(macronode=(1, 1), swap=True, displacement_k_minus_1=(0, 1)))
g2.place_operation(Measurement(macronode=(1, 2), theta=pi / 2))

make_figure(g2, scale=4.0)
```

The equivalence of the two graph representations `g1` and `g2`  can be confirmed by the fact that the lists of operations applied to the mode are identical.

```{code-cell} ipython3
print(g1.calc_mode_operations(0))
print(g2.calc_mode_operations(0))
```

### Adding Feedforward

In the graph representation, feedforward allows the parameters of certain operations or displacements to be updated based on the measurement result of another mode.
To begin, we create the graph representation of the following two-mode system:

```{code-cell} ipython3
from numpy import pi

from mqc3.graph.ops import ControlledZ, Measurement, PhaseRotation

g = GraphRepr(3, 5)

g.place_operation(Initialization((1, 0), 0.0, (BLANK_MODE, 0)))
g.place_operation(PhaseRotation((1, 1), pi / 2, swap=False, displacement_k_minus_n=(1, -1)))
g.place_operation(Initialization((0, 2), 0.0, (1, BLANK_MODE)))
g.place_operation(ControlledZ((1, 2), 1, swap=True))
g.place_operation(Measurement((2, 2), 0))
g.place_operation(Measurement((1, 4), pi / 2))

# Check the operations applied to each mode
print(g.calc_mode_operations(0))
print(g.calc_mode_operations(1))

make_figure(g)
```

Suppose that a `PhaseRotation` and a displacement are applied to mode 1, depending on the measurement result of mode 0.
The following code adds feedforward to the graph representation accordingly:

```{code-cell} ipython3
from mqc3.feedforward import feedforward


@feedforward
def f1(x: float) -> float:
    return x * x


@feedforward
def f2(x: float) -> float:
    return x * x * x


x = g.get_mode_measured_value(0)
g.place_operation(PhaseRotation((1, 3), f1(x), swap=False, displacement_k_minus_n=(f1(x), f2(x))))

make_figure(g)
```

## Converting circuit representation to graph representation

![sample circuit](_images/circuit_repr_sample_circuit.svg)

Convert the above circuit representation to a graph representation.
This can be achieved as follows:

* Place operations that initialize the modes.
* Place each operation, excluding the displacements, on the macronodes
* Displace the modes that are teleporting

```{code-cell} ipython3
from numpy import pi

from mqc3.graph.ops import ControlledZ, Measurement, PhaseRotation

g = GraphRepr(3, 4)

g.place_operation(Initialization((1, 0), 0.0, (BLANK_MODE, 0)))
g.place_operation(PhaseRotation((1, 1), pi / 2, swap=False, displacement_k_minus_n=(1, -1)))
g.place_operation(Initialization((0, 2), 0.0, (1, BLANK_MODE)))
g.place_operation(ControlledZ((1, 2), 1, swap=True))
g.place_operation(Measurement((1, 3), pi / 2))
g.place_operation(Measurement((2, 2), 0))

# Check the operations applied to each mode
print(g.calc_mode_operations(0))
print(g.calc_mode_operations(1))
```

```{code-cell} ipython3
make_figure(g, show_displacement=True)
```

MQC3 provides a class for converting circuit representation to graph representation in the `mqc3.graph.embed`.
In the `mqc3.graph.embed`, the following two conversion classes are defined.

* `GreedyConverter` : Convert circuits using greedy algorithm. This algorithm runs very quickly but tends to use a large number of macronodes.
* `BeamSearchConverter` : Convert circuits using beam search algorithm. This algorithm runs quickly and uses fewer macronodes.

By using these classes, you can implement the conversion as follows:

```{code-cell} ipython3
from numpy import pi

from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops.intrinsic import ControlledZ, Displacement, Measurement, PhaseRotation

c = CircuitRepr("multiple_gates")
c.Q(0) | PhaseRotation(phi=pi / 4) | Displacement(1, -1)
c.Q(0, 1) | ControlledZ(g=1)
c.Q(0) | Measurement(theta=0)
c.Q(1) | Measurement(theta=pi / 2)

c
```

```{code-cell} ipython3
from mqc3.graph.convert import BeamSearchConverter, BeamSearchConvertSettings

g = BeamSearchConverter(BeamSearchConvertSettings(n_local_macronodes=3, beam_width=10)).convert(c)

# Check the operations applied to each mode
print(g.calc_mode_operations(0))
print(g.calc_mode_operations(1))
```

```{code-cell} ipython3
make_figure(g)
```

## Initial states

Graph representations describe initialization through graph operations and squeezed resource states. The current local simulator accepts only circuit representations; graph simulation is planned for a later stage. Graphs can still be constructed, transformed, and visualized.
