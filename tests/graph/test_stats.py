"""Test graph statistics."""

from itertools import starmap
from math import pi

import pytest

from mqc3.feedforward import feedforward
from mqc3.graph import GraphRepr, Wiring
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.ops import (
    Initialization,
    Measurement,
    PhaseRotation,
)
from mqc3.graph.stats import compute_graph_statistics

from .common import make_sample_graph


def test_empty_graph():
    stats = compute_graph_statistics(GraphRepr(2, 3))
    assert stats.n_local_macronodes == 2
    assert stats.n_columns == 3
    assert stats.n_total_macronodes == 6
    assert stats.n_active_macronodes == 0
    assert stats.n_through_active == 0
    assert stats.n_through_blank == 6
    assert stats.n_swap_wiring == 0
    assert stats.n_swap_operation == 0
    assert stats.single_mode_op_indices == {}
    assert stats.two_mode_op_indices == {}
    assert stats.measurement_indices == []
    assert stats.feedforward_distances == []
    assert stats.n_macronodes_per_mode == {}
    assert stats.n_operations == 0
    assert stats.avg_macronodes_per_operation == pytest.approx(0.0)


def test_sample_graph():
    stats = compute_graph_statistics(make_sample_graph())
    g = make_sample_graph()

    assert stats.n_local_macronodes == 5
    assert stats.n_columns == 6
    assert stats.n_total_macronodes == 30

    # Pure through wirings: (3, 1), (1, 2), (1, 3), (2, 3), (0, 4), (0, 5), (2, 5), (3, 5), (4, 5).
    # Of these, (0, 5), (2, 5), (3, 5), (4, 5) are blank; the other 5 carry a mode.
    assert stats.n_through_active == 5
    assert stats.n_through_blank == 4

    # Swap wirings (2, 1), (0, 2), (0, 3), (4, 3); swap gate: PhaseRotation (0, 1).
    assert stats.n_swap_wiring == 4
    assert stats.n_swap_operation == 1

    assert stats.single_mode_op_indices == {
        "PhaseRotation": [g.get_index(0, 1), g.get_index(4, 2)],
        "ShearXInvariant": [g.get_index(3, 2)],
    }
    assert stats.two_mode_op_indices == {
        "ControlledZ": [g.get_index(1, 1), g.get_index(1, 4)],
        "BeamSplitter": [g.get_index(4, 1), g.get_index(3, 3)],
    }
    assert stats.measurement_indices == sorted(starmap(g.get_index, [(2, 2), (2, 4), (3, 4), (4, 4), (1, 5)]))

    # Only the four all-blank Wirings (0, 5), (2, 5), (3, 5), (4, 5) carry no mode.
    assert stats.n_active_macronodes == 26
    # Macronodes each mode enters or leaves, traced through the swaps, e.g. mode 0:
    # (0, 0) init -> (0, 1) swap down -> (1, 1) -> (2, 1) swap right -> (2, 2) measured.
    assert stats.n_macronodes_per_mode == {0: 5, 1: 6, 2: 11, 3: 5, 4: 7}

    # 5 Initialization + 2 PhaseRotation + 1 ShearXInvariant + 2 ControlledZ + 2 BeamSplitter + 5 Measurement.
    assert stats.n_operations == 17
    assert stats.avg_macronodes_per_operation == pytest.approx(26 / 17)

    # n_through_active, n_through_blank, n_swap_wiring, and n_operations partition n_total_macronodes.
    assert (
        stats.n_through_active + stats.n_through_blank + stats.n_swap_wiring + stats.n_operations
        == stats.n_total_macronodes
    )


def test_two_mode_op_indices_sorted():
    stats = compute_graph_statistics(make_sample_graph())
    for indices in stats.two_mode_op_indices.values():
        assert indices == sorted(indices)
    for indices in stats.single_mode_op_indices.values():
        assert indices == sorted(indices)


def test_active_macronodes_and_mode_usage():
    # Mode 0 is initialized at (0, 0), passes a gate at (0, 1), and is measured at (0, 2).
    g = GraphRepr(1, 4)
    g.place_operation(Initialization((0, 0), 0.0, (BLANK_MODE, 0)))
    g.place_operation(PhaseRotation((0, 1), 0.0, swap=False))
    g.place_operation(Measurement((0, 2), 0.0))
    stats = compute_graph_statistics(g)

    # (0, 3) carries no mode; the other three macronodes are active.
    assert stats.n_active_macronodes == 3
    assert stats.n_macronodes_per_mode == {0: 3}
    # Every active macronode hosts an operation, so the embedding is maximally dense.
    assert stats.n_operations == 3
    assert stats.avg_macronodes_per_operation == pytest.approx(1.0)


def test_swap_wiring_counts():
    g = GraphRepr(3, 3)
    g.place_operation(Wiring((2, 0), swap=True))
    stats = compute_graph_statistics(g)
    assert stats.n_swap_wiring == 1
    assert stats.n_swap_operation == 0
    assert stats.n_through_active == 0
    assert stats.n_through_blank == 8
    # No mode is ever initialized, so nothing is active.
    assert stats.n_active_macronodes == 0
    assert stats.n_macronodes_per_mode == {}


def test_feedforward_distances():
    # Mode 0 is measured at (0, 1); its result feeds forward into the
    # PhaseRotation applied to mode 1 at (1, 2).
    g = GraphRepr(2, 4)
    g.place_operation(Initialization((0, 0), 0.0, (BLANK_MODE, 0)))
    g.place_operation(Initialization((1, 0), 0.0, (BLANK_MODE, 1)))
    g.place_operation(Measurement((0, 1), pi / 2))

    # Mode 0 arrives at the measurement from the left micronode (bd=1).
    v = g.get_measured_value(0, 1, 1)

    @feedforward
    def f(x: float) -> float:
        return x + 1

    g.place_operation(
        PhaseRotation(
            (1, 2),
            f(v),
            swap=False,
            displacement_k_minus_1=(f(v), 0),
        ),
    )

    stats = compute_graph_statistics(g)
    from_index = g.get_index(0, 1)
    to_index = g.get_index(1, 2)
    # One from the parameter, one from the displacement.
    assert stats.feedforward_distances == [to_index - from_index] * 2


def test_feedforward_distances_mode_variable():
    # Embedders emit `ModeMeasuredVariable` feedforwards, which must be resolved
    # to the macronode where the mode is actually measured.
    g = GraphRepr(2, 4)
    g.place_operation(Initialization((0, 0), 0.0, (BLANK_MODE, 0)))
    g.place_operation(Initialization((1, 0), 0.0, (BLANK_MODE, 1)))
    g.place_operation(Measurement((0, 1), 0.0))

    v = g.get_mode_measured_value(0)

    @feedforward
    def f(x: float) -> float:
        return x + 1

    g.place_operation(PhaseRotation((1, 2), f(v), swap=False))

    stats = compute_graph_statistics(g)
    # Mode 0 is measured at (0, 1) = index 2; the consumer sits at (1, 2) = index 5.
    assert stats.feedforward_distances == [g.get_index(1, 2) - g.get_index(0, 1)]
    assert stats.feedforward_distances == [3]
