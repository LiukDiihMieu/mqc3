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

# Visualizing Machinery Representation

+++

For analyzing the machinery representation, we can visualize the graph using `mqc3.machinery.visualize` module. 
This module provides functions 

* `make_figure`: visualize the machinery representation
* `savefig`: save the figure of the visualized machinery representation.

+++

## Prepare a sample machinery representation

```{code-cell}
from mqc3.graph import GraphRepr
from mqc3.graph.constant import BLANK_MODE
from mqc3.machinery import MachineryRepr

graph = GraphRepr(5, 5)
```

```{code-cell}
from math import pi

from mqc3.feedforward import feedforward
from mqc3.graph import Wiring, ops


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
```

```{code-cell}
machinery = MachineryRepr.from_graph_repr(graph)
```

`make_figure` function can be used to visualize the machinery representation.

The function takes the machinery representation and visualize options as input and returns a `matplotlib.figure.Figure` object which can be displayed using `matplotlib.pyplot.show()` function.

```{code-cell}
from mqc3.machinery.visualize import make_figure

make_figure(machinery)
```

`savefig` function can be used to save the machinery representation as an image file.

```{code-cell}
from mqc3.machinery.visualize import savefig

savefig(machinery, "mqc3_example.png")
```

## Options

+++

You can customize the appearance of the graph by specifying the following options.

+++

### `title`

The default value is `""`.

The title of the machinery representation.
If no title is specified, the graph will be displayed without a title.

### `scale`

The default value is `2.0`.

The scale of the figure size.
If the image is low resolution, try increasing the `scale`.

### `fontsize`

The fontsize of each string. The default value is `7.0`.

### `macronode_radius`

The default value is `1.0`.

### `micronode_radius`

The default value is `0.2`.

### `measurement_color`

The default value is `black`.

The color of the measurement.

### `operation_color`

The default value is `white`.

The color of the operation.

### `readout_edge_color`

The default value is `red`.

Thr edge color of macronodes to get measurement results.

### `show_feedforward`

The default value is `True`.

The `show_feedforward` option controls whether feedforward operations are displayed.

### `show_feedforward_param_label`

The default value is `True`.

The `show_feedforward_param_label` option controls whether parameter values for feedforward operations are displayed in the circuit representation.
When set to `True`, these values appear next to their corresponding operations in the visualized circuit.

### `feedforward_param_fontsize`

The default value is `7.0`.

The `feedforward_param_fontsize` option specifies the font size used for displaying parameter values of feedforward operations in the circuit representation.
Adjusting this value can improve readability in the visualized circuit.

### `feedforward_arrow_style`

The default value is `->`.

The `feedforward_arrow_style` option customizes the style of arrows used to indicate feedforward operations.

### `feedforward_line_style`

The default value is `-`.

The `feedforward_line_style` option specifies the line style for feedforward connections.

### `feedforward_arrow_color`

The default value is `lightgreen`.

The `feedforward_arrow_color` option specifies the color of arrows indicating feedforward operations.
Colors can be defined in multiple formats, such as `red`, `blue`, or `green`.

+++

### Combining options

Multiple options can be combined.
For example, set

* the color of measurement macronodes to `gray`,
* the color of operation macronodes to `blue`,
* the edge color of macronodes to get measurement results to `yellow`.

```{code-cell}
make_figure(
    machinery,
    measurement_color="gray",
    operation_color="blue",
    readout_edge_color="yellow",
);
```

Options are also available for the `savefig` function.

```{code-cell}
savefig(
    machinery,
    "mqc3_viz_machinery_repr.png",
    measurement_color="gray",
    operation_color="blue",
    readout_edge_color="yellow",
)
```

## Visualizing a larger local window

Machinery representations can be constructed and visualized directly.

```{code-cell}
make_figure(machinery)
```
