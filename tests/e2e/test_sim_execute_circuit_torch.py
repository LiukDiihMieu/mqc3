"""Tests for the PyTorch Gaussian circuit simulator."""

# pyright: reportUnusedExpression=false

from importlib.util import find_spec
from math import pi

import numpy as np
import pytest

pytest.importorskip("torch")

from mqc3.circuit import BosonicState, CircuitRepr, GaussianState
from mqc3.circuit.ops import intrinsic, std
from mqc3.circuit.ops._base import Operation
from mqc3.client import SimulatorBackend, SimulatorClient, SimulatorClientResult
from mqc3.execute import execute

from .common import teleportation_circuit

requires_sf = pytest.mark.skipif(find_spec("strawberryfields") is None, reason="requires strawberryfields")


def _run_state(circuit: CircuitRepr, backend: SimulatorBackend) -> GaussianState:
    result = execute(circuit, SimulatorClient(n_shots=1, state_save_policy="all", backend=backend, seed=1234))
    assert isinstance(result.client_result, SimulatorClientResult)
    return result.client_result.states[0].get_gaussian_state(0)


def _two_mode_circuit(operation: Operation) -> CircuitRepr:
    circuit = CircuitRepr("two_mode")
    circuit.Q(0) | intrinsic.Displacement(1.0, -0.5)
    circuit.Q(1) | intrinsic.Displacement(-0.25, 2.0)
    circuit.Q(0, 1) | operation
    circuit.set_initial_state(0, BosonicState.squeezed(r=0.7, phi=0.2))
    circuit.set_initial_state(1, BosonicState.squeezed(r=-0.4, phi=1.1))
    return circuit


@pytest.mark.parametrize(
    "operation",
    [
        intrinsic.Displacement(1.2, -0.3),
        intrinsic.PhaseRotation(0.7),
        intrinsic.ShearXInvariant(-0.4),
        intrinsic.ShearPInvariant(0.8),
        intrinsic.Squeezing(0.9),
        intrinsic.Squeezing45(-0.6),
        intrinsic.Arbitrary(0.2, -0.7, 0.3),
        std.Squeezing(-0.5),
    ],
)
@requires_sf
def test_single_mode_operations_match_sf(operation: Operation):
    circuit = CircuitRepr("single_mode")
    circuit.Q(0) | intrinsic.Displacement(0.3, -1.2)
    circuit.Q(0) | operation
    circuit.set_initial_state(0, BosonicState.squeezed(r=0.8, phi=-0.4))

    sf_state = _run_state(circuit, "sf")
    torch_state = _run_state(circuit, "torch")
    assert np.allclose(torch_state.mean, sf_state.mean)
    assert np.allclose(torch_state.cov, sf_state.cov)


@pytest.mark.parametrize(
    "operation",
    [
        intrinsic.ControlledZ(-0.8),
        intrinsic.BeamSplitter(0.3, 0.7),
        intrinsic.TwoModeShear(0.2, -0.5),
        std.BeamSplitter(0.4, -0.2),
    ],
)
@requires_sf
def test_two_mode_operations_match_sf(operation: Operation):
    circuit = _two_mode_circuit(operation)
    sf_state = _run_state(circuit, "sf")
    torch_state = _run_state(circuit, "torch")
    assert np.allclose(torch_state.mean, sf_state.mean)
    assert np.allclose(torch_state.cov, sf_state.cov)


def test_measurement_sampling_and_seed():
    circuit = CircuitRepr("measurement")
    circuit.Q(0) | intrinsic.Displacement(1.5, -0.5)
    circuit.Q(0) | intrinsic.Measurement(theta=pi / 2)
    circuit.set_initial_state(0, BosonicState.squeezed(r=0.4, phi=0.0))

    first = SimulatorClient(n_shots=4000, backend="torch", seed=1234).run(circuit)
    second = SimulatorClient(n_shots=4000, backend="torch", seed=1234).run(circuit)
    first_samples = np.array([shot.get_value(0) for shot in first.circuit_result])
    second_samples = np.array([shot.get_value(0) for shot in second.circuit_result])

    assert np.array_equal(first_samples, second_samples)
    assert np.mean(first_samples) == pytest.approx(1.5, abs=0.03)
    assert np.var(first_samples) == pytest.approx(0.5 * np.exp(-0.8), rel=0.08)


@requires_sf
def test_measurement_conditioned_covariance_matches_sf():
    circuit = CircuitRepr("conditioned_measurement")
    circuit.Q(0, 1) | intrinsic.ControlledZ(0.7)
    circuit.Q(0) | intrinsic.Measurement(theta=0.3)
    circuit.set_initial_state(0, BosonicState.squeezed(r=0.5, phi=0.2))
    circuit.set_initial_state(1, BosonicState.squeezed(r=-0.4, phi=0.7))

    sf_state = _run_state(circuit, "sf")
    torch_state = _run_state(circuit, "torch")
    assert np.allclose(torch_state.cov, sf_state.cov)


def test_feedforward_teleportation():
    disp_x, disp_p = 15.0, -2.78
    result = SimulatorClient(n_shots=5, state_save_policy="all", backend="torch", seed=1234).run(
        teleportation_circuit((disp_x, disp_p), sq=1.56),
    )

    for state in result.states:
        gaussian = state.get_gaussian_state(0)
        assert gaussian.mean[2] == pytest.approx(disp_x, rel=1e-2)
        assert gaussian.mean[5] == pytest.approx(disp_p, rel=1e-2)
        assert gaussian.cov[2, 2] == pytest.approx(10**1.56 / 2, rel=1e-5)
        assert gaussian.cov[5, 5] == pytest.approx(10 ** (-1.56) / 2, rel=1e-5)


def test_dtype_is_configurable():
    circuit = CircuitRepr("dtype")
    circuit.Q(0) | intrinsic.PhaseRotation(0.2)
    circuit.set_initial_state(0, BosonicState.vacuum())

    state = _run_state(circuit, "torch")
    float32_result = SimulatorClient(n_shots=1, state_save_policy="all", backend="torch", dtype="float32").run(circuit)
    float32_state = float32_result.states[0].get_gaussian_state(0)
    assert np.allclose(float32_state.mean, state.mean)
    assert np.allclose(float32_state.cov, state.cov)


def test_invalid_backend_and_dtype():
    with pytest.raises(ValueError, match="Unsupported simulator backend"):
        SimulatorClient(backend="invalid")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="Unsupported torch dtype"):
        SimulatorClient(dtype="invalid")  # pyright: ignore[reportArgumentType]
