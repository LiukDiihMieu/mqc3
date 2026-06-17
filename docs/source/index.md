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
* [Quickstart](quickstart.md)

## BASIC

* [Constructing Circuit Representation](circuit_repr.md)
* [Constructing Graph Representation](graph_repr.md)
* [Constructing Machinery Representation](machinery_repr.md)
* [Feedforward](feedforward.md)
* [Simulating Circuit](simulation.md)

## ADVANCED

* [Visualizing Circuit Representation](viz_circuit_repr.md)
* [Visualizing Graph Representation](viz_graph_repr.md)
* [Visualizing Machinery Representation](viz_machinery_repr.md)

## THEORY

* [Gates](gates.md)
* [Theory](theory.md)
* [Derivation](derivation.md)

<!-- REFERENCE (API) is intentionally not built in the Jupyter Book 2 setup.
     The Sphinx autodoc pipeline (conf.py, Makefile, docs/README.md) is kept so
     the API Reference can be re-added later. Navigation lives in myst.yml. -->
