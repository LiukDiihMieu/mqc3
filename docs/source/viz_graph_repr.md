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

# Visualizing Graph Representation

+++

For analyzing the graph representation, we can visualize the graph using `mqc3.graph.visualize` module. 
This module provides functions 

* `make_figure`: visualize the graph representation
* `savefig`: save the figure of the visualized graph representation.

+++

## Prepare a sample graph

```{code-cell}
from mqc3.graph import GraphRepr
from mqc3.graph.constant import BLANK_MODE

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

`make_figure` function can be used to visualize the graph representation.

The function takes the graph representation and visualize options as input and returns a `matplotlib.figure.Figure` object which can be displayed using `matplotlib.pyplot.show()` function.

```{code-cell}
from mqc3.graph.visualize import make_figure

make_figure(graph)
```

`savefig` function can be used to save the graph representation as an image file.

```{code-cell}
from mqc3.graph.visualize import savefig

savefig(graph, "mqc3_example.png")
```

## Options

+++

You can customize the appearance of the graph by specifying the following options.

+++

### `title`

The default value is `""`.

The title of the graph representation.
If no title is specified, the graph will be displayed without a title.

```{code-cell}
make_figure(graph, title="example")
```

### `scale`

The default value is `2.0`.

The scale of the figure size.
If the image is low resolution, try increasing the `scale`.

```{code-cell}
make_figure(graph, scale=3.0)
```

### `fontsize`

The default value is `10.0`.

```{code-cell}
make_figure(graph, fontsize=15.0)
```

### `macronode_radius`

The default value is `1.0`.

```{code-cell}
make_figure(graph, macronode_radius=2.0)
```

### `macronode_edge_color`

The default value is `lightgray`.

```{code-cell}
make_figure(graph, macronode_edge_color="brown")
```

### `micronode_radius`

The default value is `0.2`.

```{code-cell}
make_figure(graph, micronode_radius=0.28)
```

### `micronode_edge_color`

The default value is `lightgray`.

```{code-cell}
make_figure(graph, micronode_edge_color="brown")
```

### `highlight_macronode_edge_color`



A dictionary mapping macronode edges to their highlight colors.
Keys are tuples of two integers representing the macronode coordinate.
Values are strings specifying the color (e.g., "red", "blue").

```{code-cell}
make_figure(graph, highlight_macronode_edge_color={(0, 0): "red", (1, 1): "blue"})
```

### `show_op_params`

The default value is `False`.

```{code-cell}
make_figure(graph, show_op_params=True)
```

### `operation_colors`

### `operation_description`

### `operation_params_format`

### `operation_params_symbol_format`

+++

### `mode_none_color`

The default value is `lightgray`.

```{code-cell}
make_figure(graph, mode_none_color="black")
```

### `mode_none_linestyle`

The default value is `:`.

```{code-cell}
make_figure(graph, mode_none_linestyle="-")
```

### `mode_colors`

The colors of modes. The argument should be a list of colors.

```{code-cell}
make_figure(graph, mode_colors=["red", "green", "blue"])
```

### `show_displacement`

The default value is `True`.

```{code-cell}
make_figure(graph, show_displacement=False)
```

### `displacement_text_format`

+++

### `ignore_wiring`

The default value is `False`. By setting `ignore_wiring=True`, you can simplify a graph visualization by hiding columns and rows that contain only wiring connections.

```{code-cell}
make_figure(graph, ignore_wiring=True)
```

### `show_mode_index`

The default value is `False`.

Controls whether mode indices are displayed.

```{code-cell}
make_figure(graph, show_mode_index=True, ignore_wiring=True)
```

### `show_feedforward`

The default value is `True`.

The `show_feedforward` option controls whether feedforward operations are displayed.

### `show_feedforward_param_label`

The default value is `True`.

The `show_feedforward_param_label` option controls whether parameter values for feedforward operations are displayed in the circuit representation.
When set to `True`, these values appear next to their corresponding operations in the visualized circuit.

### `feedforward_param_fontsize`

The default value is `8.0`.

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

* the color of the circumference of macronodes to `black`, 
* the radius of micronodes to `0.1`, and
* the color of modes to `['red', 'blue', 'green']`.

```{code-cell}
make_figure(
    graph,
    macronode_edge_color="black",
    micronode_radius=0.1,
    mode_colors=["red", "blue", "green"],
    ignore_wiring=True,
);
```

Options are also available for the `savefig` function.

```{code-cell}
savefig(
    graph,
    "mqc3_example_large.png",
    macronode_edge_color="black",
    micronode_radius=0.1,
    mode_colors=["red", "blue", "green"],
    ignore_wiring=True,
)
```
