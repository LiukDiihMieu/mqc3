"""Beam-splitter network circuit template.

Implements the triangular beam-splitter network and the Bloch-Messiah-based
multimode Gaussian network of `arXiv:2506.11236 <https://arxiv.org/abs/2506.11236>`_
(Fig. 7 and Fig. 9, respectively). Gate parameters are placeholders; only the
circuit structure is meaningful so far.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqc3.circuit.ops.intrinsic import (
    BeamSplitter,
    PhaseRotation,
    Squeezing,
)

if TYPE_CHECKING:
    from mqc3.circuit import CircuitRepr


def bs_triangle(
    circuit: CircuitRepr,
    n_modes: int,
) -> CircuitRepr:
    """Append one triangular beam-splitter network (Fig. 7).

    Args:
        circuit (CircuitRepr): Circuit to append the network to.
        n_modes (int): Number of modes the network acts on.

    Returns:
        CircuitRepr: The same `circuit` instance with the network appended.
    """
    for i in range(n_modes - 1, -1, -1):
        for j in range(n_modes - 1, i, -1):
            circuit.Q(i, j) | BeamSplitter(
                sqrt_r=0.707,
                theta_rel=0.0,
            )
        circuit.Q(i) | PhaseRotation(phi=0.0)
    return circuit


def squeezing_layer(
    circuit: CircuitRepr,
    n_modes: int,
) -> CircuitRepr:
    """Append one single-mode squeezing gate to each mode.

    Args:
        circuit (CircuitRepr): Circuit to append the squeezing gates to.
        n_modes (int): Number of modes to squeeze.

    Returns:
        CircuitRepr: The same `circuit` instance with the squeezing gates appended.
    """
    for i in range(n_modes):
        circuit.Q(i) | Squeezing(theta=1.0)
    return circuit


def gaussian_network(
    circuit: CircuitRepr,
    n_modes: int,
) -> CircuitRepr:
    """Append a general multimode Gaussian network (Fig. 9).

    Implements the Bloch-Messiah decomposition of an arbitrary multimode Gaussian
    unitary as two triangular beam-splitter networks (`bs_triangle`) sandwiching a
    single-mode squeezing layer (`squeezing_layer`).

    Args:
        circuit (CircuitRepr): Circuit to append the network to.
        n_modes (int): Number of modes the network acts on.

    Returns:
        CircuitRepr: The same `circuit` instance with the network appended.
    """
    bs_triangle(circuit, n_modes)
    squeezing_layer(circuit, n_modes)
    bs_triangle(circuit, n_modes)
    return circuit
