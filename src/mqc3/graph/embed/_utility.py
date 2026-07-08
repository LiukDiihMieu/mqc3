import mqc3.circuit.ops.intrinsic as cops
import mqc3.graph.ops as gops
from mqc3.circuit.program import CircOpParam, CircuitRepr
from mqc3.circuit.program import Operation as CircOp
from mqc3.feedforward import FeedForward
from mqc3.graph.ops import GraphOpParam, ModeMeasuredVariable
from mqc3.graph.program import Operation as GraphOp


def _is_displacement(op: CircOp) -> bool:
    return op.name() == "intrinsic.displacement"


def count_n_ops_except_displacement(circuit: CircuitRepr) -> int:
    return sum(not _is_displacement(op) for op in circuit)


def op_indices_except_displacement(circuit: CircuitRepr) -> list[int]:
    return [i for i, op in enumerate(circuit) if not _is_displacement(op)]


def convert_param(param: CircOpParam) -> GraphOpParam:
    """Convert an operation parameter from the circuit representation to the graph one.

    Constant values are returned unchanged. A feedforward parameter is
    re-keyed: at the circuit level its variable references the measurement *operation
    object* it comes from, but that object does not exist at the graph level, so the
    variable is replaced with a ``ModeMeasuredVariable`` carrying only the measured
    mode id ("the measurement outcome of mode m"). The feedforward function chain is
    preserved by re-applying ``param.func`` to the new variable. The mode id is later
    resolved back to a concrete measurement node via ``mode_to_measurement`` when
    building the dependency DAG (see ``_DependencyBuilder._apply_feedforward``).

    Args:
        param (CircOpParam): Operation parameter in the circuit representation.

    Returns:
        GraphOpParam: Operation parameter in the graph representation.

    Raises:
        ValueError: The feedforward references a measurement whose operand has no modes.
    """
    if not isinstance(param, FeedForward):
        return param

    op = param.variable.get_from_operation()
    if not op.opnd().get_ids():
        msg = "Cannot convert feedforward with empty IDs."
        raise ValueError(msg)

    mode = op.opnd().get_ids()[0]

    ff = ModeMeasuredVariable(mode)
    return param.func(ff)


def convert_intrinsic_op(cop: cops.Intrinsic, coord: tuple[int, int]) -> list[GraphOp]:  # noqa: C901, PLR0911
    """Convert operation object to a new operation object in the graph representation.

    Args:
        cop (Operation): Operation object in the circuit representation.
        coord (tuple[int, int]): Coordinate of the macronode to apply or start the operation
            in the graph representation.

    Returns:
        GOperation: Operation object in the graph representation.

    Raises:
        RuntimeError: The input operation is not supported
    """
    if isinstance(cop, cops.Measurement):
        return [gops.Measurement(coord, convert_param(cop.theta))]
    if isinstance(cop, cops.PhaseRotation):
        return [gops.PhaseRotation(coord, convert_param(cop.phi), swap=False)]
    if isinstance(cop, cops.ShearXInvariant):
        return [gops.ShearXInvariant(coord, convert_param(cop.kappa), swap=False)]
    if isinstance(cop, cops.ShearPInvariant):
        return [gops.ShearPInvariant(coord, convert_param(cop.eta), swap=False)]
    if isinstance(cop, cops.Squeezing):
        return [gops.Squeezing(coord, convert_param(cop.theta), swap=False)]
    if isinstance(cop, cops.Squeezing45):
        return [gops.Squeezing45(coord, convert_param(cop.theta), swap=False)]
    if isinstance(cop, cops.Arbitrary):
        arb_first = gops.ArbitraryFirst(
            coord,
            convert_param(cop.alpha),
            convert_param(cop.beta),
            convert_param(cop.lam),
            swap=False,
        )
        arb_second = gops.ArbitrarySecond(
            coord,
            convert_param(cop.alpha),
            convert_param(cop.beta),
            convert_param(cop.lam),
            swap=False,
        )
        return [arb_first, arb_second]
    if isinstance(cop, cops.ControlledZ):
        return [gops.ControlledZ(coord, convert_param(cop.g), swap=False)]
    if isinstance(cop, cops.BeamSplitter):
        return [gops.BeamSplitter(coord, convert_param(cop.sqrt_r), convert_param(cop.theta_rel), swap=False)]
    if isinstance(cop, cops.TwoModeShear):
        return [gops.TwoModeShear(coord, convert_param(cop.a), convert_param(cop.b), swap=False)]
    if isinstance(cop, cops.Manual):
        return [
            gops.Manual(
                coord,
                convert_param(cop.theta_a),
                convert_param(cop.theta_b),
                convert_param(cop.theta_c),
                convert_param(cop.theta_d),
                swap=False,
            )
        ]
    msg = "This operation is not supported"
    raise RuntimeError(msg)


def convert_op(cop: CircOp, coord: tuple[int, int]) -> list[GraphOp]:
    """Convert operation object to a new operation object in the graph representation.

    Args:
        cop (Operation): Operation object in the circuit representation.
        coord (tuple[int, int]): Coordinate of the macronode to apply or start the operation
            in the graph representation.

    Returns:
        GOperation: Operation object in the graph representation.

    Raises:
        TypeError: One of the return values of to_intrinsic_ops is not an intrinsic operation.
    """
    gop_list = []
    for intrinsic_op in cop.to_intrinsic_ops():
        if not isinstance(intrinsic_op, cops.Intrinsic):
            msg = "One of the return values of to_intrinsic_ops is not an intrinsic operation."
            raise TypeError(msg)
        gop_list.extend(convert_intrinsic_op(intrinsic_op, coord))
    return gop_list
