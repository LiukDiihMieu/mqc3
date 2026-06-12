"""Run Gaussian circuit representations with PyTorch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import acos, cos, exp, sin, sqrt, tan
from typing import TYPE_CHECKING, Literal
from zoneinfo import ZoneInfo

import numpy as np

import mqc3.circuit.ops.intrinsic as intrinsic_ops
import mqc3.circuit.ops.std as std_ops
from mqc3.circuit.result import CircuitOperationMeasuredValue, CircuitResult, CircuitShotMeasuredValue
from mqc3.circuit.state import BosonicState, GaussianState
from mqc3.constant import hbar
from mqc3.feedforward import FeedForward

if TYPE_CHECKING:
    import torch

    from mqc3.circuit.ops._base import CircOpParam, Operation
    from mqc3.circuit.program import CircuitRepr

logger = logging.getLogger(__name__)

TorchDType = Literal["float32", "float64"]


def _import_torch():  # noqa: ANN202
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        msg = """PyTorch is not installed.
Please install the CPU build with:

uv pip install torch --index-url https://download.pytorch.org/whl/cpu
"""
        raise ImportError(msg) from exc
    return torch


def _resolve_dtype(dtype: TorchDType):  # noqa: ANN202
    torch = _import_torch()
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    msg = f"Unsupported torch dtype: {dtype}."
    raise ValueError(msg)


def _parameter_value(param: CircOpParam, measured: dict[int, float]) -> float:
    if isinstance(param, float | int):
        return float(param)
    if isinstance(param, FeedForward):
        mode = param.variable.get_from_operation().opnd().get_ids()[0]
        try:
            value = measured[mode]
        except KeyError as exc:
            msg = f"Feedforward refers to mode {mode}, which has not been measured."
            raise ValueError(msg) from exc
        result = param.func(value)
        if isinstance(result, FeedForward):
            msg = "Feedforward evaluation returned another feedforward expression."
            raise TypeError(msg)
        return float(result)
    msg = f"Invalid parameter: {param}"
    raise ValueError(msg)


@dataclass
class _GaussianState:
    mean: torch.Tensor
    cov: torch.Tensor

    @property
    def n_modes(self) -> int:
        return self.mean.shape[0] // 2

    @classmethod
    def from_circuit(cls, circuit: CircuitRepr, *, dtype: torch.dtype) -> _GaussianState:
        torch = _import_torch()
        n_modes = circuit.n_modes
        mean = torch.empty(2 * n_modes, dtype=dtype, device="cpu")
        cov = torch.zeros((2 * n_modes, 2 * n_modes), dtype=dtype, device="cpu")

        for mode in range(n_modes):
            state = circuit.get_initial_state(mode)
            if not isinstance(state, BosonicState):
                msg = "Initial state must be 'BosonicState' instance."
                raise TypeError(msg)
            if state.n_peaks != 1:
                msg = "Only single-peak initial states are supported."
                raise ValueError(msg)

            gaussian = state.get_gaussian_state(0)
            if not np.allclose(gaussian.mean.imag, 0):
                msg = "Torch Gaussian backend supports only real-valued quadrature means."
                raise ValueError(msg)

            indices = [mode, mode + n_modes]
            mean[indices] = torch.as_tensor(gaussian.mean.real, dtype=dtype)
            cov[np.ix_(indices, indices)] = torch.as_tensor(gaussian.cov, dtype=dtype)

        return cls(mean=mean, cov=cov)

    def transform(self, modes: list[int], matrix: torch.Tensor, displacement: torch.Tensor | None = None) -> None:
        torch = _import_torch()
        indices = modes + [mode + self.n_modes for mode in modes]
        transform = torch.eye(2 * self.n_modes, dtype=self.mean.dtype, device=self.mean.device)
        transform[np.ix_(indices, indices)] = matrix
        self.mean = transform @ self.mean
        if displacement is not None:
            self.mean[indices] += displacement
        self.cov = transform @ self.cov @ transform.T
        self.cov = (self.cov + self.cov.T) / 2

    def measure(self, mode: int, theta: float, *, generator: torch.Generator) -> float:
        torch = _import_torch()
        n_modes = self.n_modes
        measured_indices = [mode, mode + n_modes]
        remaining_indices = [i for i in range(2 * n_modes) if i not in measured_indices]
        direction = torch.tensor([sin(theta), cos(theta)], dtype=self.mean.dtype)

        measured_mean = self.mean[measured_indices]
        measured_cov = self.cov[np.ix_(measured_indices, measured_indices)]
        mean = direction @ measured_mean
        variance = direction @ measured_cov @ direction
        tolerance = 10 * torch.finfo(self.mean.dtype).eps
        if variance < -tolerance:
            msg = f"Homodyne variance is negative ({variance.item()})."
            raise ValueError(msg)
        variance = variance.clamp_min(0)
        sample = mean + torch.sqrt(variance) * torch.randn((), dtype=self.mean.dtype, generator=generator)

        if remaining_indices:
            cross_cov = self.cov[np.ix_(remaining_indices, measured_indices)]
            covariance_with_measurement = cross_cov @ direction
            if variance > tolerance:
                gain = covariance_with_measurement / variance
                self.mean[remaining_indices] += gain * (sample - mean)
                self.cov[np.ix_(remaining_indices, remaining_indices)] -= (
                    torch.outer(
                        covariance_with_measurement,
                        covariance_with_measurement,
                    )
                    / variance
                )

        # Strawberry Fields leaves measured modes as independent vacuum modes.
        self.mean[measured_indices] = 0
        self.cov[measured_indices, :] = 0
        self.cov[:, measured_indices] = 0
        self.cov[np.ix_(measured_indices, measured_indices)] = torch.eye(
            2,
            dtype=self.mean.dtype,
        ) * (hbar / 2)
        self.cov = (self.cov + self.cov.T) / 2
        return float(sample.item())

    def to_bosonic_state(self) -> BosonicState:
        return BosonicState(
            np.array([1.0 + 0.0j], dtype=np.complex128),
            [
                GaussianState(
                    self.mean.detach().cpu().numpy().astype(np.complex128),
                    self.cov.detach().cpu().numpy().astype(np.float64),
                ),
            ],
        )


def _rotation(phi: float, *, dtype: torch.dtype) -> torch.Tensor:
    torch = _import_torch()
    return torch.tensor([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]], dtype=dtype)


def _std_beam_splitter(theta: float, phi: float, *, dtype: torch.dtype) -> torch.Tensor:
    torch = _import_torch()
    c, s = cos(theta), sin(theta)
    cp, sp = cos(phi), sin(phi)
    return torch.tensor(
        [
            [c, -s * cp, 0, -s * sp],
            [s * cp, c, -s * sp, 0],
            [0, s * sp, c, -s * cp],
            [s * sp, 0, s * cp, c],
        ],
        dtype=dtype,
    )


def _intrinsic_beam_splitter(sqrt_r: float, theta_rel: float, *, dtype: torch.dtype) -> torch.Tensor:
    torch = _import_torch()
    alpha = (theta_rel + acos(sqrt_r)) / 2
    beta = (theta_rel - acos(sqrt_r)) / 2
    plus, minus = alpha + beta, alpha - beta
    cp, cm, sp, sm = cos(plus), cos(minus), sin(plus), sin(minus)
    return torch.tensor(
        [
            [cp * cm, sp * sm, -sp * cm, sm * cp],
            [sp * sm, cp * cm, sm * cp, -sp * cm],
            [sp * cm, -sm * cp, cp * cm, sp * sm],
            [-sm * cp, sp * cm, sp * sm, cp * cm],
        ],
        dtype=dtype,
    )


def _apply_operation(  # noqa: C901, PLR0912, PLR0914
    state: _GaussianState,
    operation: Operation,
    measured: dict[int, float],
    *,
    generator: torch.Generator,
) -> None:
    torch = _import_torch()
    dtype = state.mean.dtype
    modes = operation.opnd().get_ids()
    value = lambda param: _parameter_value(param, measured)  # noqa: E731

    if isinstance(operation, intrinsic_ops.Measurement):
        measured[modes[0]] = state.measure(modes[0], value(operation.theta), generator=generator)
    elif isinstance(operation, intrinsic_ops.Displacement):
        displacement = torch.tensor([value(operation.x), value(operation.p)], dtype=dtype)
        state.transform(modes, torch.eye(2, dtype=dtype), displacement)
    elif isinstance(operation, intrinsic_ops.PhaseRotation):
        state.transform(modes, _rotation(value(operation.phi), dtype=dtype))
    elif isinstance(operation, intrinsic_ops.ShearXInvariant):
        kappa = value(operation.kappa)
        state.transform(modes, torch.tensor([[1, 0], [2 * kappa, 1]], dtype=dtype))
    elif isinstance(operation, intrinsic_ops.ShearPInvariant):
        eta = value(operation.eta)
        state.transform(modes, torch.tensor([[1, 2 * eta], [0, 1]], dtype=dtype))
    elif isinstance(operation, intrinsic_ops.Squeezing):
        theta = value(operation.theta)
        state.transform(modes, torch.tensor([[0, tan(theta)], [-1 / tan(theta), 0]], dtype=dtype))
    elif isinstance(operation, intrinsic_ops.Squeezing45):
        theta = value(operation.theta)
        coeff = 1 / sqrt(2)
        before = torch.tensor([[coeff, -coeff], [coeff, coeff]], dtype=dtype)
        squeeze = torch.diag(torch.tensor([1 / tan(theta), tan(theta)], dtype=dtype))
        after = torch.tensor([[coeff, coeff], [-coeff, coeff]], dtype=dtype)
        state.transform(modes, after @ squeeze @ before)
    elif isinstance(operation, intrinsic_ops.Arbitrary):
        alpha, beta, lam = value(operation.alpha), value(operation.beta), value(operation.lam)
        squeeze = torch.diag(torch.tensor([exp(-lam), exp(lam)], dtype=dtype))
        state.transform(modes, _rotation(alpha, dtype=dtype) @ squeeze @ _rotation(beta, dtype=dtype))
    elif isinstance(operation, intrinsic_ops.ControlledZ):
        g = value(operation.g)
        state.transform(
            modes,
            torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, g, 1, 0], [g, 0, 0, 1]], dtype=dtype),
        )
    elif isinstance(operation, intrinsic_ops.BeamSplitter):
        state.transform(
            modes,
            _intrinsic_beam_splitter(value(operation.sqrt_r), value(operation.theta_rel), dtype=dtype),
        )
    elif isinstance(operation, intrinsic_ops.TwoModeShear):
        a, b = value(operation.a), value(operation.b)
        state.transform(
            modes,
            torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [2 * a, b, 1, 0], [b, 2 * a, 0, 1]], dtype=dtype),
        )
    elif isinstance(operation, intrinsic_ops.Manual):
        msg = "Circuits containing intrinsic.Manual cannot be executed."
        raise TypeError(msg)
    elif isinstance(operation, std_ops.Squeezing):
        r = value(operation.r)
        state.transform(modes, torch.diag(torch.tensor([exp(-r), exp(r)], dtype=dtype)))
    elif isinstance(operation, std_ops.BeamSplitter):
        state.transform(modes, _std_beam_splitter(value(operation.theta), value(operation.phi), dtype=dtype))
    else:
        msg = f"Unsupported operation type: {operation.name()}."
        raise TypeError(msg)


def _run_shot(
    circuit: CircuitRepr,
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[dict[int, float], BosonicState]:
    state = _GaussianState.from_circuit(circuit, dtype=dtype)
    measured: dict[int, float] = {}
    for operation in circuit:
        _apply_operation(state, operation, measured, generator=generator)
    return measured, state.to_bosonic_state()


@dataclass(frozen=True)
class TorchLocalResult:
    """The result of executing a quantum circuit with PyTorch."""

    execution_time: timedelta
    circuit_result: CircuitResult
    states: list[BosonicState]


def local_run(
    n_shots: int,
    state_save_policy: str,
    circuit: CircuitRepr,
    *,
    dtype: TorchDType = "float64",
    seed: int | None = None,
) -> TorchLocalResult:
    """Run a single-peak Gaussian circuit with PyTorch.

    Returns:
        TorchLocalResult: Measurement values and optionally saved states.
    """
    torch = _import_torch()
    torch_dtype = _resolve_dtype(dtype)
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()

    started_at = datetime.now(ZoneInfo("Asia/Tokyo"))
    shot_values: list[CircuitShotMeasuredValue] = []
    states: list[BosonicState] = []
    for shot in range(n_shots):
        measured, state = _run_shot(circuit, dtype=torch_dtype, generator=generator)
        shot_values.append(
            CircuitShotMeasuredValue(
                CircuitOperationMeasuredValue(index=mode, value=value) for mode, value in sorted(measured.items())
            ),
        )
        if state_save_policy == "all" or (shot == 0 and state_save_policy == "first_only"):
            states.append(state)
    finished_at = datetime.now(started_at.tzinfo)

    return TorchLocalResult(
        execution_time=finished_at - started_at,
        circuit_result=CircuitResult(shot_values),
        states=states,
    )
