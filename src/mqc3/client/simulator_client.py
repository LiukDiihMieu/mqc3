"""Local simulator client for optical quantum computing.

This client runs circuit representations on the local Strawberry Fields or
PyTorch backend. The remote (cloud) execution path has been removed in MQC-mini.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Literal, SupportsIndex
from zoneinfo import ZoneInfo

from mqc3.circuit import CircuitRepr, CircuitResult
from mqc3.circuit.state import BosonicState
from mqc3.client.abstract import AbstractClient, AbstractClientResult, MeasuredValue, ReprType, ResultType
from mqc3.graph import GraphResult
from mqc3.machinery import MachineryResult

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_TZ_UTC = timezone.utc  # noqa: UP017


@dataclass
class JobTimeline:
    """Timeline of a local simulation run."""

    submitted_at: datetime | None = None
    execution_started_at: datetime | None = None
    execution_finished_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def execution_time(self) -> timedelta | None:
        """Return the execution time of the simulation.

        Returns:
            timedelta | None: Execution time of the simulation.
        """
        if self.execution_started_at is None or self.execution_finished_at is None:
            return None
        return self.execution_finished_at - self.execution_started_at

    @property
    def total_time(self) -> timedelta | None:
        """Return the total time of the simulation.

        Returns:
            timedelta | None: Total time of the simulation.
        """
        if self.submitted_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.submitted_at


@dataclass
class ExecutionDetails:
    """Version and timeline information for a local simulation run."""

    simulator_version: str = ""
    timeline: JobTimeline | None = None


StateSavePolicy = Literal["all", "first_only", "none"]
"""Policy for saving states after simulation.

- ``all``: Save states of all shots.
- ``first_only``: Save only the state of the first shot.
- ``none``: Do not save any states.
"""

SimulatorBackend = Literal["sf", "torch"]
"""Available local simulator backends."""

TorchDType = Literal["float32", "float64"]
"""Floating-point precision used by the PyTorch backend."""


@dataclass(frozen=True)
class SimulatorClientResult(AbstractClientResult):
    """The result of executing a representation with a SimulatorClient."""

    execution_details: ExecutionDetails
    """Execution details including the simulator version and the timeline of the run."""

    _circuit_result: CircuitResult | None
    """Measurement results after circuit execution."""

    _graph_result: GraphResult | None
    """Measurement results after graph execution."""

    _machinery_result: MachineryResult | None
    """Measurement results after machinery execution."""

    states: list[BosonicState]
    """States after circuit execution."""

    @property
    def circuit_result(self) -> CircuitResult:
        """Return the circuit execution result.

        Returns:
            CircuitResult: The circuit execution result.

        Raises:
            ValueError: If no circuit result is available. This usually means the
                circuit has not been executed yet or execution did not produce a
                circuit-level result.
        """
        if self._circuit_result is None:
            msg = (
                "Circuit result is not available. Run the circuit first or verify "
                "that execution produced a circuit result."
            )
            raise ValueError(msg)
        return self._circuit_result

    @property
    def graph_result(self) -> GraphResult:
        """Return the graph execution result.

        Returns:
            GraphResult: The graph execution result.

        Raises:
            ValueError: If no graph result is available. This usually means the
                graph-level execution has not been performed yet or it produced no
                result.
        """
        if self._graph_result is None:
            msg = (
                "Graph result is not available. Ensure that graph execution has been performed and produced a result."
            )
            raise ValueError(msg)
        return self._graph_result

    @property
    def machinery_result(self) -> MachineryResult:
        """Return the machinery execution result.

        Returns:
            MachineryResult: The machinery execution result.

        Raises:
            ValueError: If no machinery result is available. This usually means
                machinery execution has not been performed yet or it produced
                no result.
        """
        if self._machinery_result is None:
            msg = (
                "Machinery result is not available. Ensure that machinery "
                "execution has been performed and produced a result."
            )
            raise ValueError(msg)
        return self._machinery_result

    @property
    def execution_result(self) -> ResultType:
        """Get the result of the top layer in the execution.

        Returns:
            CircuitResult: Result.

        Examples:
            >>> from mqc3.client import SimulatorClient
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> from mqc3.circuit.state import BosonicState
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | intrinsic.Displacement(0.1, 0.2)
            [QuMode(id=0)]
            >>> circuit.Q(0)| intrinsic.Measurement(0.0)
            Var([intrinsic.measurement] [0.0] [QuMode(id=0)])
            >>> circuit.set_initial_state(0, BosonicState.vacuum())
            >>> client = SimulatorClient(n_shots=1)
            >>> result = client.run(circuit)  # doctest: +SKIP
            >>> circuit_result = result.execution_result  # doctest: +SKIP

        Raises:
            ValueError: Raised when no execution result is available, i.e.,
                none of ``circuit_result``, ``graph_result``, or ``machinery_result``
                has been set (for example, if the circuit has not been run).
        """
        if self._circuit_result is not None:
            return self._circuit_result
        if self._graph_result is not None:
            return self._graph_result
        if self._machinery_result is not None:
            return self._machinery_result
        msg = ""
        raise ValueError(msg)

    @property
    def n_shots(self) -> int:
        """Return the number of shots.

        Returns:
            int: The number of shots.
        """
        return self.execution_result.n_shots()

    @property
    def execution_time(self) -> timedelta | None:
        """Return the execution time of the simulation.

        Returns:
            timedelta | None: The execution time of the simulation.
        """
        if self.execution_details.timeline is None:
            return None

        return self.execution_details.timeline.execution_time

    @property
    def total_time(self) -> timedelta | None:
        """Return the total time of the simulation.

        Returns:
            timedelta | None: The total time of the simulation.
        """
        if self.execution_details.timeline is None:
            return None

        return self.execution_details.timeline.total_time

    def __len__(self) -> int:
        """Return the number of shots.

        Returns:
            int: The number of shots.
        """
        return self.n_shots

    def __iter__(self) -> Iterator:
        """Iterator of the result.

        The iterator is the same as that of ``self.execution_result``.

        Yields:
            Iterator: Iterator of the result.
        """
        yield from self.execution_result

    def get_shot_measured_value(self, index: int) -> MeasuredValue:
        """Get the measured value of the shot at the index.

        This function gets values from ``self.execution_result``.

        Args:
            index (int): Shot index.

        Returns:
            MeasuredValue: Measured value.
        """
        return self.execution_result.get_shot_measured_value(index)

    def __getitem__(
        self,
        index: int | slice | SupportsIndex,
    ) -> MeasuredValue | Sequence[MeasuredValue]:
        """Get the measured value of the shot at the index.

        This function gets values from ``self.execution_result``.

        Args:
            index (int | slice | SupportsIndex): Index or slice.

        Returns:
            MeasuredValue | Sequence[MeasuredValue]: Measured value or sequence of measured values.
        """
        return self.execution_result.measured_vals[index]


class SimulatorClient(AbstractClient):
    """Local simulator client for optical quantum computing.

    Representations are simulated locally with Strawberry Fields or PyTorch.

    Limitations:

    - Supports only circuit representations.
    - Circuits containing ``intrinsic.Manual`` cannot be executed.
    - All modes must have explicitly defined initial states.
    - Each mode's initial state must contain exactly one peak.
    """

    state_save_policy: StateSavePolicy
    """Policy for saving states after simulation.

    See :data:`StateSavePolicy`.
    """

    timezone: timezone | ZoneInfo
    """Timezone to use."""

    backend: SimulatorBackend
    """Local simulator backend."""

    dtype: TorchDType
    """Floating-point precision used by the PyTorch backend."""

    seed: int | None
    """Random seed used by the PyTorch backend."""

    def __init__(  # noqa: PLR0913
        self,
        n_shots: int = 1024,
        state_save_policy: StateSavePolicy = "none",
        *,
        backend: SimulatorBackend = "sf",
        dtype: TorchDType = "float64",
        seed: int | None = None,
        timezone: timezone | ZoneInfo = _TZ_UTC,
    ) -> None:
        """SimulatorClient constructor.

        Args:
            n_shots (int): The number of shots.
            state_save_policy (:data:`StateSavePolicy`): Policy for saving states after simulation
                (all, first_only or none).
            backend (:data:`SimulatorBackend`): Local simulator backend.
            dtype (:data:`TorchDType`): Floating-point precision used by the PyTorch backend.
            seed (int | None): Random seed used by the PyTorch backend.
            timezone (timezone | ZoneInfo): Timezone to use (default: UTC).

        Examples:
            >>> from mqc3.client import SimulatorClient
            >>> client = SimulatorClient(
            ...     n_shots=1024,
            ...     state_save_policy="first_only",
            ... )

        Raises:
            ValueError: If the backend or dtype is unsupported.
        """
        AbstractClient.__init__(self, n_shots=n_shots)
        if backend not in {"sf", "torch"}:
            msg = f"Unsupported simulator backend: {backend}."
            raise ValueError(msg)
        if dtype not in {"float32", "float64"}:
            msg = f"Unsupported torch dtype: {dtype}."
            raise ValueError(msg)
        self.state_save_policy = state_save_policy
        self.backend = backend
        self.dtype = dtype
        self.seed = seed
        self.timezone = timezone

    def _run_local(self, circuit: CircuitRepr) -> SimulatorClientResult:
        execution_started_at = datetime.now(tz=self.timezone)
        if self.backend == "sf":
            from mqc3.client._local_simulator import local_run  # noqa: PLC0415

            local_result = local_run(self.n_shots, self.state_save_policy, circuit)
        else:
            from mqc3.client._torch_simulator import local_run  # noqa: PLC0415

            local_result = local_run(
                self.n_shots,
                self.state_save_policy,
                circuit,
                dtype=self.dtype,
                seed=self.seed,
            )

        execution_finished_at = datetime.now(tz=self.timezone)

        return SimulatorClientResult(
            execution_details=ExecutionDetails(
                simulator_version=self.backend,
                timeline=JobTimeline(
                    submitted_at=execution_started_at,
                    execution_started_at=execution_started_at,
                    execution_finished_at=execution_finished_at,
                    finished_at=execution_finished_at,
                ),
            ),
            _circuit_result=local_result.circuit_result,
            _machinery_result=None,
            _graph_result=None,
            states=local_result.states,
        )

    def run(self, representation: ReprType) -> SimulatorClientResult:
        """Run a representation with the local simulator.

        Note:
            This method is synchronous in the sense that it runs the simulation and waits for the result.

        Note:
            Supports only circuit representations.

        Args:
            representation (ReprType): Representation.

        Returns:
            SimulatorClientResult: Result of the running.

        Raises:
            ValueError: If `representation` is not a `CircuitRepr`.

        Example:
            >>> from numpy import pi
            >>> from mqc3.client import SimulatorClient
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> from mqc3.circuit.state import BosonicState
            >>> client = SimulatorClient(n_shots=1)
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | intrinsic.PhaseRotation(pi / 2)
            [QuMode(id=0)]
            >>> circuit.Q(0) | intrinsic.Measurement(0.0)
            Var([intrinsic.measurement] [0.0] [QuMode(id=0)])
            >>> circuit.set_initial_state(0, BosonicState.vacuum())
            >>> result = client.run(circuit)  # doctest: +SKIP
        """
        if not isinstance(representation, CircuitRepr):
            msg = (
                f"Local simulator supports only CircuitRepr (got {type(representation).__name__}). "
                "Convert the input to CircuitRepr."
            )
            raise ValueError(msg)
        return self._run_local(representation)
