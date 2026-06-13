(mqc3-index)=

# MQC-mini Documentation

MQC-mini is a simulation-only Python library for measurement-based optical
quantum computing. It provides tools for constructing continuous-variable
quantum programs and running circuit representations on local simulators.

The forward-only PyTorch Gaussian backend supports single-peak Gaussian states,
Gaussian operations, homodyne measurements, and feedforward. Strawberry Fields
is retained as a legacy reference backend.

## GET STARTED

* [Installation](installation.md)
* [Quickstart](quickstart.ipynb)

```{toctree}
:caption: GET STARTED
:hidden:
installation.md
quickstart.ipynb
```

## BASIC

* [Constructing Circuit Representation](circuit_repr.ipynb)
* [Constructing Graph Representation](graph_repr.ipynb)
* [Constructing Machinery Representation](machinery_repr.ipynb)
* [Feedforward](feedforward.ipynb)
* [Simulating Circuit](simulation.ipynb)

```{toctree}
:caption: BASIC
:hidden:
circuit_repr.ipynb
graph_repr.ipynb
machinery_repr.ipynb
feedforward.ipynb
simulation.ipynb
```

## ADVANCED

* [Visualizing Circuit Representation](viz_circuit_repr.ipynb)
* [Visualizing Graph Representation](viz_graph_repr.ipynb)
* [Visualizing Machinery Representation](viz_machinery_repr.ipynb)

```{toctree}
:caption: ADVANCED
:hidden:
viz_circuit_repr.ipynb
viz_graph_repr.ipynb
viz_machinery_repr.ipynb
```

## THEORY

* [Gates](gates.md)
* [Theory](theory.md)
* [Derivation](derivation.ipynb)

```{toctree}
:caption: THEORY
:hidden:
gates.md
theory.md
derivation.ipynb
```

## REFERENCE

* [API Reference](reference/modules.rst)

```{toctree}
:caption: REFERENCE
:hidden:
:maxdepth: 2
reference/modules.rst
```
