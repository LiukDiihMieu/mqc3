"""CV-QAOA circuit template.

Generalizes the two-mode QAOA circuit of `arXiv:2606.10432 <https://arxiv.org/abs/2606.10432>`_
(Fig. 2(a)) to an arbitrary number of modes with linear-chain ControlledZ couplings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqc3.circuit.ops.intrinsic import ControlledZ, Displacement, ShearPInvariant, ShearXInvariant

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqc3.circuit import CircuitRepr


def _broadcast(value: float | Sequence[float], n: int, name: str) -> list[float]:
    """Broadcast a scalar to length `n`, or validate the length of a sequence.

    Args:
        value (float | Sequence[float]): Scalar to broadcast or sequence to validate.
        n (int): Required length.
        name (str): Parameter name used in the error message.

    Returns:
        list[float]: `value` as a list of length `n`.

    Raises:
        ValueError: If `value` is a sequence whose length is not `n`.
    """
    if isinstance(value, (int, float)):
        return [float(value)] * n
    values = [float(v) for v in value]
    if len(values) != n:
        msg = f"{name} must be a scalar or a sequence of length {n}, got length {len(values)}"
        raise ValueError(msg)
    return values


def _broadcast_pairs(
    value: tuple[float, float] | Sequence[tuple[float, float]], n: int, name: str
) -> list[tuple[float, float]]:
    """Broadcast a single `(x, p)` pair to length `n`, or validate a sequence of pairs.

    Args:
        value (tuple[float, float] | Sequence[tuple[float, float]]): Pair to broadcast
            or sequence of pairs to validate.
        n (int): Required length.
        name (str): Parameter name used in the error message.

    Returns:
        list[tuple[float, float]]: `value` as a list of pairs of length `n`.

    Raises:
        ValueError: If `value` is a sequence of pairs whose length is not `n`.
    """
    seq = list(value)
    if seq and isinstance(seq[0], (int, float)):
        x, p = seq
        return [(float(x), float(p))] * n
    pairs = [(float(x), float(p)) for x, p in seq]
    if len(pairs) != n:
        msg = f"{name} must be an (x, p) pair or a sequence of {n} pairs, got length {len(pairs)}"
        raise ValueError(msg)
    return pairs


def qaoa(  # noqa: PLR0913
    circuit: CircuitRepr,
    n_modes: int,
    n_layers: int,
    *,
    gammas: float | Sequence[float] = 1.0,
    betas: float | Sequence[float] = 1.5,
    quadratic: float | Sequence[float] = 1.0,
    coupling: float | Sequence[float] = 1.0,
    linear: tuple[float, float] | Sequence[tuple[float, float]] = (0.5, 0.0),
) -> CircuitRepr:
    """Append CV-QAOA layers with linear-chain couplings to a circuit.

    Each layer applies the cost part — ShearXInvariant on every mode, ControlledZ on every
    chain edge (i, i+1), and a Displacement on every mode — followed by the ShearPInvariant
    mixer on every mode. The layers act on modes 0 to `n_modes - 1`; no measurement is
    appended.

    The parameters separate the cost problem from the layer schedule, as in standard QAOA:
    `quadratic`, `coupling` and `linear` are the per-mode/per-edge coefficients of the cost
    Hamiltonian (constant across layers), while `gammas` and `betas` are the per-layer
    schedule. The gate parameters in layer `l` are:

    - ShearXInvariant on mode `i`: `gammas[l] * quadratic[i]`
    - ControlledZ on edge `(i, i + 1)`: `gammas[l] * coupling[i]`
    - Displacement on mode `i`: `gammas[l] * linear[i]` (both `x` and `p` components)
    - ShearPInvariant on mode `i`: `betas[l]`

    Every parameter also accepts a scalar (a single `(x, p)` pair for `linear`), which is
    broadcast to all layers, modes or edges.

    Args:
        circuit (CircuitRepr): Circuit to append the layers to.
        n_modes (int): Number of modes the layers act on.
        n_layers (int): Number of QAOA layers to append.
        gammas (float | Sequence[float]): Per-layer schedule of the cost part.
        betas (float | Sequence[float]): Per-layer schedule of the mixer part.
        quadratic (float | Sequence[float]): Per-mode diagonal coefficients of the cost.
        coupling (float | Sequence[float]): Per-edge coupling coefficients of the cost.
        linear (tuple[float, float] | Sequence[tuple[float, float]]): Per-mode `(x, p)`
            linear coefficients of the cost.

    Returns:
        CircuitRepr: The same `circuit` instance with the layers appended.

    Raises:
        ValueError: If `n_modes < 1`, `n_layers < 0`, or a sequence parameter has the
            wrong length.

    Examples:
        >>> from mqc3.circuit import CircuitRepr
        >>> from mqc3.templates import qaoa
        >>> circuit = qaoa(CircuitRepr("qaoa"), n_modes=2, n_layers=1)
        >>> circuit.n_operations
        7
        >>> circuit = qaoa(CircuitRepr("qaoa"), n_modes=2, n_layers=2, gammas=[0.3, 0.7], betas=[0.9, 0.4])
        >>> circuit.n_operations
        14
    """
    if n_modes < 1:
        msg = f"n_modes must be >= 1, got {n_modes}"
        raise ValueError(msg)
    if n_layers < 0:
        msg = f"n_layers must be >= 0, got {n_layers}"
        raise ValueError(msg)

    gamma_list = _broadcast(gammas, n_layers, "gammas")
    beta_list = _broadcast(betas, n_layers, "betas")
    quadratic_list = _broadcast(quadratic, n_modes, "quadratic")
    coupling_list = _broadcast(coupling, n_modes - 1, "coupling")
    linear_list = _broadcast_pairs(linear, n_modes, "linear")

    for layer in range(n_layers):
        gamma, beta = gamma_list[layer], beta_list[layer]
        for i in range(n_modes):
            circuit.Q(i) | ShearXInvariant(gamma * quadratic_list[i])
        for i in range(n_modes - 1):
            circuit.Q(i, i + 1) | ControlledZ(g=gamma * coupling_list[i])
        for i in range(n_modes):
            circuit.Q(i) | Displacement(x=gamma * linear_list[i][0], p=gamma * linear_list[i][1])
        for i in range(n_modes):
            circuit.Q(i) | ShearPInvariant(beta)
    return circuit
