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

# Feedforward

In this section, we provide a detailed explanation of *feedforward*.

In MQC3, **feedforward** refers to the process of updating the parameters of certain operations in a quantum circuit based on measurement outcomes obtained during the execution of that circuit.

Feedforward consists of two main components: **variables** and **functions**.

## Variables

A *variable* represents a measurement result obtained during the execution of a quantum program.  
In this SDK, variables are represented by the class `Variable` or a subclass thereof.

The way variables are obtained depends on the specific representation of the quantum circuit.  
For details on how to access measurement results and update parameters, refer to the documentation for each representation:  
([Circuit representation](circuit_repr.md), [Graph representation](graph_repr.md), [Machinery representation](machinery_repr.md)).

## Functions

Feedforward functions are defined using the class `FeedForwardFunction`.
You can create a feedforward function by decorating a regular Python function with `feedforward`.

To remain serializable and consistently evaluable, feedforward functions must satisfy the following constraints:

* The function must take a `float` as input and return a `float`.
  * It must contain exactly one `return` statement.
* It must be self-contained.
  * The function must not rely on any variables or functions defined outside its scope.
* Only the `math` module may be imported.
  * The only allowed import format is `from math import foo`.
* Only simple mathematical operations are permitted, including:
  * Variable assignments (`=`)
  * Certain operators:
    * Unary: `+`, `-`, `~`
    * Binary: `+`, `-`, `*`, `/`, `//`, `%`, `<<`, `>>`, `&`, `^`, `|`, `**`, `@`
  * Function calls from the `math` module
  * A limited set of built-in functions:
    * `abs`, `bool`, `complex`, `divmod`, `float`, `int`, `pow`, `round`

To verify whether a function satisfies these constraints, use `verify_feedforward` or `verify`. The local simulator evaluates feedforward when execution reaches the dependent operation, using measurements obtained earlier in the same shot.
