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

# Visualizing Circuit Representation

+++

The `mqc3.circuit.visualize` module enables you to analyze circuit representations by generating graphical visualizations. It provides the following functions:

* `make_figure`: Generates a figure of the circuit representation.
* `savefig`: Save the generated figure to a file.

+++

## Prepare a sample circuit

```{code-cell}
from mqc3.circuit import CircuitRepr

c = CircuitRepr("sample_circuit")
```

```{code-cell}
from math import pi

from mqc3.circuit.ops import intrinsic, std
from mqc3.feedforward import feedforward

c.Q(0) | intrinsic.PhaseRotation(pi)
c.Q(1) | intrinsic.ShearPInvariant(eta=0.1)
c.Q(0, 2) | std.BeamSplitter(theta=1, phi=0.1)
c.Q(3) | intrinsic.ShearPInvariant(eta=0.1)


@feedforward
def f(x: float) -> float:
    return x + 1


x = c.Q(0) | intrinsic.Measurement(theta=0.0)
y = c.Q(1) | intrinsic.Measurement(theta=pi)
c.Q(2, 3) | intrinsic.TwoModeShear(a=f(x), b=f(y))

c.Q(2) | intrinsic.Measurement(theta=0.0)
c.Q(3) | intrinsic.Measurement(theta=pi)
```

`make_figure` function can be used to visualize the circuit representation.

The function takes the circuit representation and visualize options as input and returns a `matplotlib.figure.Figure` object which can be displayed using `matplotlib.pyplot.show()` function.

```{code-cell}
from mqc3.circuit.visualize import make_figure

make_figure(c)
```

`savefig` function can be used to save the circuit representation as an image file.

```{code-cell}
from mqc3.circuit.visualize import savefig

savefig(c, "mqc3_circuit_example.png")
```

## Options

+++

You can customize the appearance of the visualized circuit figure by specifying the following options.

+++

### `show_parameters`

The default value is `True`.

The `show_parameters` option controls whether the parameters of each operation are displayed.
When set to `True`, parameter values are shown next to their corresponding operations in the visualized circuit.

### `parameters_fontsize`
The default value is `8.0`.


The `parameters_fontsize` option specifies the font size used for displaying parameter values.
Adjusting this value can improve readability in the visualized circuit.

```{code-cell}
make_figure(c, show_parameters=True, parameters_fontsize=15.0)
```

### `show_feedforward_param_label`

The default value is `True`.

The `show_feedforward_param_label` option controls whether parameter values for feedforward operations are displayed in the circuit representation.
When set to `True`, these values appear next to their corresponding operations in the visualized circuit.

### `feedforward_param_fontsize`

The default value is `8.0`.

The `feedforward_param_fontsize` option specifies the font size used for displaying parameter values of feedforward operations in the circuit representation.
Adjusting this value can improve readability in the visualized circuit.

```{code-cell}
make_figure(c, show_feedforward_param_label=True, feedforward_param_fontsize=15.0)
```

### `feedforward_arrow_style`

The default value is `->`.

The `feedforward_arrow_style` option customizes the style of arrows used to indicate feedforward operations.

### `feedforward_arrow_color`

The default value is `lightgreen`.

The `feedforward_arrow_color` option specifies the color of arrows indicating feedforward operations.
Colors can be defined in multiple formats, such as `red`, `blue`, or `green`.

```{code-cell}
make_figure(c, feedforward_arrow_style="->", feedforward_arrow_color="red")
```

### `qumode_labels`

The default value is $q_{i}$, where `i` is the mode index.

The `qumode_labels` option specifies custom labels for the quantum modes.
You can provide a list of strings to use as labels.

### `qumode_hline_spacing`

The default value is `DEFAULT_QUMODE_HLINE_SPACING`.

The `qumode_hline_spacing` option sets the spacing between horizontal lines separating quantum modes.

```{code-cell}
make_figure(c, qumode_labels=["q0", "q1", "q2", "q3"], qumode_hline_spacing=2.0)
```

### `intrinsic_operation_name_table`

The default value is `DEFAULT_INTRINSIC_OP_NAME_TABLE`.

The `intrinsic_operation_name_table` option customizes the names of intrinsic operations.
You can provide a dictionary mapping operation names to their desired display names.

### `user_operation_name_table`

The default value is `DEFAULT_USER_OP_NAME_TABLE`.

The `user_operation_name_table` option customizes the names of user-defined operations.
You can provide a dictionary mapping operation names to their desired display names.

### `intrinsic_operation_color_table`

The default value is `DEFAULT_INTRINSIC_OP_COLOR_TABLE`.

The `intrinsic_operation_color_table` option customizes the colors of intrinsic operations in the circuit representation.
You can provide a dictionary mapping operation names to their desired colors.

### `opbox_width`

The default value is `DEFAULT_OPBOX_WIDTH`.

The `opbox_width` option sets the width of operation boxes.

### `opbox_vertical_margin`

The default value is `DEFAULT_OPBOX_VERTICAL_MARGIN`.

The `opbox_vertical_margin` option sets the vertical margin between operation boxes.

+++

### `show_op_legend`

The default value is `False`.

The `show_op_legend` option controls whether a legend for intrinsic operations is displayed.
When set to `True`, the legend appears at the bottom of the visualized circuit.

```{code-cell}
from mqc3.circuit.visualize import (
    DEFAULT_INTRINSIC_OP_COLOR_TABLE,
    DEFAULT_INTRINSIC_OP_NAME_TABLE,
    DEFAULT_USER_OP_NAME_TABLE,
)

intrinsic_names = DEFAULT_INTRINSIC_OP_NAME_TABLE.copy()
intrinsic_names["PhaseRotation"] = r"$Rot$"
user_names = DEFAULT_USER_OP_NAME_TABLE.copy()
user_names["BeamSplitter"] = r"$B$"
intrinsic_colors = DEFAULT_INTRINSIC_OP_COLOR_TABLE.copy()
intrinsic_colors["PhaseRotation"] = "lightblue"

make_figure(
    c,
    intrinsic_operation_name_table=intrinsic_names,
    user_operation_name_table=user_names,
    intrinsic_operation_color_table=intrinsic_colors,
    opbox_vertical_margin=0.5,
    opbox_width=1.2,
    show_op_legend=True,
);
```
