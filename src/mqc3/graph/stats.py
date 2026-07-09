"""Summary statistics of a graph representation.

This module is meant for comparing the outcome of embedding many circuits at once
(e.g. average swap count over 100 embedded circuits) without inspecting each graph visually.
"""

from __future__ import annotations

from dataclasses import dataclass

from mqc3.feedforward import FeedForward
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.ops import (
    ArbitraryFirst,
    ArbitrarySecond,
    BeamSplitter,
    ControlledZ,
    Manual,
    Measurement,
    ModeMeasuredVariable,
    PhaseRotation,
    ShearPInvariant,
    ShearXInvariant,
    Squeezing,
    Squeezing45,
    TwoModeShear,
    Wiring,
)
from mqc3.graph.program import GraphRepr, to_pos_measured_variable

_SINGLE_MODE_OP_TYPES = (
    PhaseRotation,
    ShearXInvariant,
    ShearPInvariant,
    Squeezing,
    Squeezing45,
    ArbitraryFirst,
    ArbitrarySecond,
)
_TWO_MODE_OP_TYPES = (ControlledZ, BeamSplitter, TwoModeShear, Manual)


@dataclass
class GraphStatistics:
    """Summary statistics of a `GraphRepr`."""

    n_local_macronodes: int
    "The number of local macronodes."

    n_columns: int
    "The number of columns (steps)."

    n_total_macronodes: int
    "The total number of macronodes (`n_local_macronodes * n_columns`)."

    n_active_macronodes: int
    "The number of macronodes carrying at least one non-blank input or output mode."

    n_through: int
    "The number of `Wiring` macronodes that are neither a swap nor a gate."

    n_swap: int
    "The number of operations with `swap=True`, whether or not the macronode carries a mode."

    single_mode_op_indices: dict[str, list[int]]
    "Macronode indices of single-mode gates, keyed by operation class name."

    two_mode_op_indices: dict[str, list[int]]
    "Macronode indices of two-mode gates, keyed by operation class name."

    measurement_indices: list[int]
    "Macronode indices of `Measurement` operations."

    feedforward_distances: list[int]
    "Macronode index gaps between each feedforward's source measurement and its consuming operation."

    n_macronodes_per_mode: dict[int, int]
    "For each mode, the number of macronodes the mode enters or leaves (including its initialization macronode)."

    n_operations: int
    "The number of non-`Wiring` operations, which equals the node count of the embedded dependency DAG."

    avg_macronodes_per_operation: float
    "`n_active_macronodes / n_operations` (0.0 if there is no operation): routing overhead per operation."


def _resolve_feedforward_distances(graph: GraphRepr) -> list[int]:
    distances = []
    for op in graph.operations.values():
        to_index = graph.get_index(*op.macronode)
        params = [*op.parameters, *op.displacement_k_minus_1, *op.displacement_k_minus_n]
        for p in params:
            if not isinstance(p, FeedForward):
                continue
            var = p.variable
            if isinstance(var, ModeMeasuredVariable):
                var = to_pos_measured_variable(var, graph)
            from_index = graph.get_index(var.h, var.w)
            distances.append(to_index - from_index)
    return distances


def _count_active_and_mode_usage(graph: GraphRepr) -> tuple[int, dict[int, int]]:
    n_active_macronodes = 0
    n_macronodes_per_mode: dict[int, int] = {}
    for io_modes in graph.io_modes_dict().values():
        modes_here = {mode for mode in io_modes if mode != BLANK_MODE}
        if modes_here:
            n_active_macronodes += 1
        for mode in modes_here:
            n_macronodes_per_mode[mode] = n_macronodes_per_mode.get(mode, 0) + 1
    return n_active_macronodes, dict(sorted(n_macronodes_per_mode.items()))


def compute_graph_statistics(graph: GraphRepr) -> GraphStatistics:
    """Compute summary statistics of a graph representation.

    Args:
        graph (GraphRepr): The graph representation to summarize.

    Returns:
        GraphStatistics: Summary statistics of `graph`.

    Examples:
        >>> from mqc3.graph import GraphRepr, ops
        >>> from mqc3.graph.stats import compute_graph_statistics
        >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
        >>> graph.place_operation(ops.PhaseRotation((1, 2), phi=1, swap=True))
        >>> compute_graph_statistics(graph).n_swap
        1
    """
    n_through = 0
    n_swap = 0
    single_mode_op_indices: dict[str, list[int]] = {}
    two_mode_op_indices: dict[str, list[int]] = {}
    measurement_indices: list[int] = []

    n_operations = 0
    for (h, w), op in graph.operations.items():
        index = graph.get_index(h, w)

        if isinstance(op, Wiring):
            if not op.swap:
                n_through += 1
        else:
            n_operations += 1
        if op.swap:
            n_swap += 1

        if isinstance(op, _SINGLE_MODE_OP_TYPES):
            single_mode_op_indices.setdefault(type(op).__name__, []).append(index)
        elif isinstance(op, _TWO_MODE_OP_TYPES):
            two_mode_op_indices.setdefault(type(op).__name__, []).append(index)
        elif isinstance(op, Measurement):
            measurement_indices.append(index)

    for indices in (*single_mode_op_indices.values(), *two_mode_op_indices.values()):
        indices.sort()
    measurement_indices.sort()

    n_active_macronodes, n_macronodes_per_mode = _count_active_and_mode_usage(graph)

    return GraphStatistics(
        n_local_macronodes=graph.n_local_macronodes,
        n_columns=graph.n_steps,
        n_total_macronodes=graph.n_total_macronodes,
        n_active_macronodes=n_active_macronodes,
        n_through=n_through,
        n_swap=n_swap,
        single_mode_op_indices=single_mode_op_indices,
        two_mode_op_indices=two_mode_op_indices,
        measurement_indices=measurement_indices,
        feedforward_distances=_resolve_feedforward_distances(graph),
        n_macronodes_per_mode=n_macronodes_per_mode,
        n_operations=n_operations,
        avg_macronodes_per_operation=n_active_macronodes / n_operations if n_operations else 0.0,
    )
