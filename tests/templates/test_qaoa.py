"""Test the CV-QAOA circuit template."""

import pytest

from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops.intrinsic import ControlledZ, Displacement, Measurement, PhaseRotation, ShearXInvariant
from mqc3.graph.convert import (
    BeamSearchConverter,
    BeamSearchConvertSettings,
    GreedyConverter,
    GreedyConvertSettings,
)
from mqc3.templates import qaoa


def count_ops_by_type(circuit: CircuitRepr) -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in range(circuit.n_operations):
        name = type(circuit.get_operation(i)).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


def test_gate_counts():
    n_modes, n_layers = 4, 3
    circuit = qaoa(CircuitRepr("qaoa"), n_modes, n_layers)
    counts = count_ops_by_type(circuit)
    assert counts["ShearXInvariant"] == n_layers * n_modes
    assert counts["ControlledZ"] == n_layers * (n_modes - 1)
    assert counts["Displacement"] == n_layers * n_modes
    assert counts["ShearPInvariant"] == n_layers * n_modes
    assert circuit.n_operations == n_layers * (4 * n_modes - 1)
    assert circuit.n_modes == n_modes


def test_chain_couplings():
    n_modes = 5
    circuit = qaoa(CircuitRepr("qaoa"), n_modes, 1)
    cz_modes = [
        circuit.get_operation(i).opnd().get_ids()
        for i in range(circuit.n_operations)
        if isinstance(circuit.get_operation(i), ControlledZ)
    ]
    assert cz_modes == [[i, i + 1] for i in range(n_modes - 1)]


def test_scalar_parameters_broadcast():
    circuit = qaoa(
        CircuitRepr("qaoa"),
        2,
        2,
        gammas=2.0,
        betas=0.2,
        quadratic=0.1,
        coupling=0.3,
        linear=(0.4, 0.5),
    )
    for i in range(circuit.n_operations):
        op = circuit.get_operation(i)
        if isinstance(op, ShearXInvariant):
            assert op.parameters() == pytest.approx([2.0 * 0.1])
        elif isinstance(op, ControlledZ):
            assert op.parameters() == pytest.approx([2.0 * 0.3])
        elif isinstance(op, Displacement):
            assert op.parameters() == pytest.approx([2.0 * 0.4, 2.0 * 0.5])
        else:
            assert op.parameters() == pytest.approx([0.2])


def test_schedule_times_problem_coefficients():
    gammas = [0.5, 2.0]
    betas = [0.9, 0.4]
    quadratic = [1.0, 3.0]
    coupling = [0.7]
    linear = [(0.1, 0.2), (0.3, 0.4)]
    circuit = qaoa(
        CircuitRepr("qaoa"),
        2,
        2,
        gammas=gammas,
        betas=betas,
        quadratic=quadratic,
        coupling=coupling,
        linear=linear,
    )
    # Per-layer op order: ShearX(0), ShearX(1), CZ, Disp(0), Disp(1), ShearP(0), ShearP(1).
    n_ops_per_layer = 7
    for layer, (gamma, beta) in enumerate(zip(gammas, betas)):
        ops = [circuit.get_operation(layer * n_ops_per_layer + k) for k in range(n_ops_per_layer)]
        assert ops[0].parameters() == pytest.approx([gamma * quadratic[0]])
        assert ops[1].parameters() == pytest.approx([gamma * quadratic[1]])
        assert ops[2].parameters() == pytest.approx([gamma * coupling[0]])
        assert ops[3].parameters() == pytest.approx([gamma * linear[0][0], gamma * linear[0][1]])
        assert ops[4].parameters() == pytest.approx([gamma * linear[1][0], gamma * linear[1][1]])
        assert ops[5].parameters() == pytest.approx([beta])
        assert ops[6].parameters() == pytest.approx([beta])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gammas": [1.0]},
        {"betas": [1.0, 2.0, 3.0]},
        {"quadratic": [1.0]},
        {"coupling": [1.0]},
        {"linear": [(0.0, 0.0)]},
    ],
)
def test_wrong_parameter_length_raises(kwargs):
    with pytest.raises(ValueError, match="length"):
        qaoa(CircuitRepr("qaoa"), 3, 2, **kwargs)


def test_appends_to_existing_circuit():
    circuit = CircuitRepr("qaoa")
    circuit.Q(0) | PhaseRotation(phi=1.0)
    n_before = circuit.n_operations

    result = qaoa(circuit, 2, 1)

    assert result is circuit
    assert circuit.n_operations == n_before + 7
    assert isinstance(circuit.get_operation(0), PhaseRotation)


def test_single_mode_has_no_couplings():
    circuit = qaoa(CircuitRepr("qaoa"), 1, 2)
    assert "ControlledZ" not in count_ops_by_type(circuit)
    assert circuit.n_operations == 2 * 3


def test_zero_layers_appends_nothing():
    circuit = qaoa(CircuitRepr("qaoa"), 3, 0)
    assert circuit.n_operations == 0


@pytest.mark.parametrize(("n_modes", "n_layers"), [(0, 1), (-1, 1), (2, -1)])
def test_invalid_arguments(n_modes, n_layers):
    with pytest.raises(ValueError, match="must be >="):
        qaoa(CircuitRepr("qaoa"), n_modes, n_layers)


@pytest.mark.parametrize(
    "converter",
    [
        GreedyConverter(GreedyConvertSettings(n_local_macronodes=8)),
        BeamSearchConverter(BeamSearchConvertSettings(n_local_macronodes=8, beam_width=4)),
    ],
)
def test_embeddable(converter):
    circuit = qaoa(CircuitRepr("qaoa"), 3, 2)
    for i in range(3):
        circuit.Q(i) | Measurement(theta=0)
    graph = converter.convert(circuit)
    assert graph.n_local_macronodes == 8
