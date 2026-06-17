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

# Constructing Machinery Representation

In this section, we will explain how to construct quantum circuits with the "Machinery Representation".

Machinery representation is the lowest-level representation of quantum circuits, which is constructed by specifying the following parameters.

| Parameter | Setter     | Getter | Description |
| --------- | ---------- | ------ | ----------- |
| `n_local_macronodes` | `__init__` | `n_local_macronodes` | The number of macronodes per step. |
| `n_steps` | `__init__` | `n_steps` | The number of steps. |
| `readout_macronode_indices` | `readout_macronode_indices` | `readout_macronode_indices` | Set of macronode indices to get measurement results. |
| `homodyne_angles` | `set_homodyne_angle` | `get_homodyne_angle` | List of four measurement angles in each macronode. The setter and getter handle one element at a time. |
| `displacements_k_minus_1` | `set_displacement_k_minus_1` | `get_displacement_k_minus_1` | Displacement applied to the mode teleporting to the macronode `k` from the macronode `k - 1`. |
| `displacements_k_minus_n` | `set_displacement_k_minus_n` | `get_displacement_k_minus_n` | Displacement applied to the mode teleporting to the macronode `k` from the macronode `k - n_local_macronodes`. |
| `ff_coeff_matrix_k_plus_1` | `ff_coeff_matrix_k_plus_1` | `ff_coeff_matrix_k_plus_1` | Feedforward coefficient matrix from the macronode `k` to the macronode `k + 1`. |
| `ff_coeff_matrix_k_plus_n` | `ff_coeff_matrix_k_plus_n` | `ff_coeff_matrix_k_plus_n` | Feedforward coefficient matrix from the macronode `k` to the macronode `k + n_local_macronodes`. |

By default, the feedforward coefficient matrix is automatically calculated from the homodyne angles.

+++

## Creating a machinery representation

```{code-cell}
from mqc3.machinery import MachineryRepr
from mqc3.machinery.visualize import make_figure

m = MachineryRepr(
    n_local_macronodes=3,
    n_steps=4,
)

make_figure(m, readout_edge_color="red")
```

### Setting macronodes to readout

```{code-cell}
m.readout_macronode_indices = {6, 8, 10}

make_figure(m, readout_edge_color="red")
```

### Setting homodyne angles

```{code-cell}
from math import pi

from mqc3.machinery.macronode_angle import MacronodeAngle

m.set_homodyne_angle(4, MacronodeAngle(0, pi / 2, 0, pi / 2))
m.set_homodyne_angle(5, MacronodeAngle(pi / 2, 0, pi / 2, 0))
m.set_homodyne_angle(7, MacronodeAngle(0, pi / 2, 0, pi / 2))
m.set_homodyne_angle(10, MacronodeAngle(pi / 2, pi / 2, pi / 2, pi / 2))

make_figure(m, readout_edge_color="red")
```

### Adding displacements

```{code-cell}
m.set_displacement_k_minus_1(4, (1.0, -1.0))

make_figure(m, readout_edge_color="red")
```

### Settings generation method of feedforward coefficient matrix

You can specify how the feedforward coefficient matrix is generated, as shown in the code below.
The available methods are defined in the enum class `FFCoeffMatrixGenerationMethods`.

Currently, you can choose from the following options:

- `ZERO_FILLED`: Initializes the matrix with all zeros.
- `FROM_MACRONODE_ANGLES`: Computes the matrix based on macronode measurement angles.

```{code-cell}
from mqc3.machinery.program import FFCoeffMatrixGenerationMethods

m.ff_coeff_matrix_k_plus_n.generation_methods[1] = FFCoeffMatrixGenerationMethods.ZERO_FILLED
m.ff_coeff_matrix_k_plus_n.generation_methods[4] = FFCoeffMatrixGenerationMethods.FROM_MACRONODE_ANGLE
```

### Adding feedforward

Measured variables used for feedforward are created using the `MeasuredVariable` class.  
This class takes two arguments:

- The first argument is the **macronode index**.
- The second argument is the **micronode index** within the macronode.

The micronode index corresponds to the following labels:

- `0`: micronode **a**
- `1`: micronode **b**
- `2`: micronode **c**
- `3`: micronode **d**

```{code-cell}
from mqc3.feedforward import feedforward
from mqc3.machinery.macronode_angle import MacronodeAngle, MeasuredVariable


@feedforward
def f1(x: float) -> float:
    return x * x


@feedforward
def f2(x: float) -> float:
    return x * x * x


x = MeasuredVariable(2, 1)  # macronode 2, micronode b
y = MeasuredVariable(3, 2)  # macronode 3, micronode c
m.set_displacement_k_minus_n(4, (f1(x), f2(y)))
m.set_homodyne_angle(4, MacronodeAngle(f1(x), f2(y), 0, pi / 2))
m.set_homodyne_angle(8, MacronodeAngle(pi / 2, 0, f1(x), f2(y)))

make_figure(m, readout_edge_color="red")
```

## Converting graph representation to machinery representation

You can construct a machinery representation from a graph representation using the `from_graph_repr` method.

```{code-cell}
from math import pi

from mqc3.feedforward import feedforward
from mqc3.graph import GraphRepr, Wiring, ops
from mqc3.graph.constant import BLANK_MODE
from mqc3.machinery import MachineryRepr


@feedforward
def add_pi(x: float) -> float:
    from math import pi  # noqa:PLC0415

    return x + pi


@feedforward
def square(x: float) -> float:
    return x * x


@feedforward
def cube(x: float) -> float:
    return x * x * x


graph = GraphRepr(5, 4)
graph.place_operation(ops.Initialization((0, 1), 0.0, (0, BLANK_MODE)))
graph.place_operation(ops.Initialization((1, 0), 0.0, (BLANK_MODE, 1)))
graph.place_operation(ops.Initialization((2, 0), 0.0, (BLANK_MODE, 2)))
graph.place_operation(ops.BeamSplitter((1, 1), 0.0, 0.0, displacement_k_minus_n=(1.0, -1.0), swap=True))
graph.place_operation(ops.TwoModeShear((2, 1), 0.0, 0.0, displacement_k_minus_1=(-1.0, 1.0), swap=True))
graph.place_operation(ops.PhaseRotation((3, 1), pi / 2, swap=True))
graph.place_operation(ops.Measurement((1, 2), 0.0))  # Measure mode 0
m0 = graph.get_mode_measured_value(0)
graph.place_operation(Wiring((2, 2), swap=True))
graph.place_operation(ops.ControlledZ((3, 2), 0.0, swap=False))
graph.place_operation(ops.Measurement((4, 2), 0.0))  # Measure mode 1
m1 = graph.get_mode_measured_value(1)
graph.place_operation(
    ops.ShearPInvariant((3, 3), square(m0), displacement_k_minus_n=(add_pi(m0), cube(m1)), swap=True)
)
graph.place_operation(ops.Measurement((4, 3), 0.0))  # Measure mode 2

m = MachineryRepr.from_graph_repr(graph)

print("#local macronodes         =", m.n_local_macronodes)
print("#steps                    =", m.n_steps)
print("homodyne angles           =", [m.get_homodyne_angle(i) for i in range(12)])
print("readout_macronode_indices =", m.readout_macronode_indices)
print("displacement k-1->k       =", m.displacements_k_minus_1)
print("displacement k-N->k       =", m.displacements_k_minus_n)
print("ff coeff matrix k->k+1    = ", m.ff_coeff_matrix_k_plus_1)
print("ff coeff matrix k->k+N    = ", m.ff_coeff_matrix_k_plus_n)

make_figure(m, readout_edge_color="red")
```

## Initial states

Machinery representations describe time-ordered operations and squeezed resource states. The current local simulator accepts only circuit representations; machinery simulation is planned for a later stage. Machinery representations can still be constructed and visualized.
