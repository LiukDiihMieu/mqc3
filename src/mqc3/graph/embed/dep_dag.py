"""Dependency DAG constructor.

Dependency DAG is a Directed Acyclic Graph that represents the dependency of operations.
This module provides a class to construct the dependency graph from a quantum circuit.
"""

from math import pi

import networkx as nx

import mqc3.circuit.ops.intrinsic as cops
import mqc3.graph.ops as gops
from mqc3.circuit.program import CircuitRepr
from mqc3.circuit.state import HardwareConstrainedSqueezedState
from mqc3.feedforward import FeedForward, ff_to_add_constant
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.embed._utility import convert_op, convert_param
from mqc3.graph.ops import GraphOpParam, Measurement, ModeMeasuredVariable
from mqc3.graph.program import GraphRepr
from mqc3.graph.program import Operation as GraphOp
from mqc3.pb.mqc3_cloud.program.v1.graph_pb2 import GraphOperation as PbOperation

DEFAULT_COORDINATE = (-1, -1)


def _sum_displacements(disp1: GraphOpParam, disp2: GraphOpParam) -> GraphOpParam:
    """Sum two displacement parameters into one.

    Called component-wise: each call combines a single x or p component of two
    consecutive displacements on the same mode.

    Args:
        disp1 (GraphOpParam): Displacement component 1.
        disp2 (GraphOpParam): Displacement component 2.

    Returns:
        GraphOpParam: Combined displacement component.

    Raises:
        TypeError: Applying multiple displacements with feedforward is not allowed.
    """
    if isinstance(disp1, FeedForward):
        if isinstance(disp2, FeedForward):
            msg = "Applying multiple displacements with feedforward is not allowed."
            raise TypeError(msg)
        x1, x2 = disp1, disp2
    else:
        x1, x2 = disp2, disp1

    return ff_to_add_constant(x2)(x1)


class DependencyDAG:
    """Directed Acyclic Graph which represents dependency relationship between operations."""

    def __init__(self, circuit_or_graph: CircuitRepr | GraphRepr) -> None:
        """Initializes the DependencyDAG by constructing its internal dependency graph.

        Args:
            circuit_or_graph (CircuitRepr | GraphRepr):
                The input quantum circuit representation or graph representation
                from which to build the DependencyDAG.
        """
        self.dag: nx.DiGraph  # the dependency graph itself, held as a plain networkx DiGraph

        builder = _DependencyBuilder()  # stateful helper that constructs self.dag
        if isinstance(circuit_or_graph, CircuitRepr):
            self.dag = builder.from_circuit(circuit_or_graph)
        else:
            self.dag = builder.from_graph(circuit_or_graph)
        self.n_modes = len(builder.last_op_of_modes)


class _DependencyBuilder:
    # Id assigned to the next DAG node (node ids are consecutive ints: 0, 1, 2, ...).
    next_node_id: int
    # The dependency DAG under construction (see `_add_op_node` for node attributes).
    dep_graph: nx.DiGraph
    # mode -> displacement (x, p) accumulated on that mode but not attached to a node yet;
    # an absent key means no pending displacement. Displacements never become nodes themselves:
    # they are attached to the next operation node on the mode (see `_apply_displacement`).
    last_disp_of_modes: dict[int, tuple[GraphOpParam, GraphOpParam]]
    # mode -> node id of the latest operation on that mode, i.e. the current tail of the
    # mode's wire; the next operation on the mode gets a dependency edge from this node.
    last_op_of_modes: dict[int, int]
    # measured mode -> node id of the Measurement operation on that mode; used to resolve
    # feedforward parameters to the measurement they depend on (see `_apply_feedforward`).
    mode_to_measurement: dict[int, int]

    def __init__(self) -> None:
        self.next_node_id = 0
        self.dep_graph = nx.DiGraph()
        self.last_disp_of_modes = {}
        self.last_op_of_modes = {}
        self.mode_to_measurement = {}

    def _apply_displacement(self, target: int) -> None:
        """Attach the pending displacement of each of the target's modes to the target node."""
        modes = self.dep_graph.nodes[target]["modes"]
        for mode in modes:
            prev_disp = self.last_disp_of_modes.pop(mode, None)
            if prev_disp is not None:
                self.dep_graph.nodes[target]["displacements"].append((mode, prev_disp))

    def _apply_dependency(self, target: int) -> None:
        """Add wire-order edges: on each mode, the previous operation must precede the target."""
        modes = self.dep_graph.nodes[target]["modes"]
        for mode in modes:
            if mode in self.last_op_of_modes:
                self.dep_graph.add_edge(self.last_op_of_modes[mode], target)

            self.last_op_of_modes[mode] = target

    def _apply_feedforward(self, target: int) -> None:
        """Add dependency edges for the feedforward parameters of the target node.

        A feedforward parameter is a function of an earlier measurement outcome, so for
        each such parameter the measurement node M must precede the target in the DAG.
        This is the data dependency itself: edge ``M -> target``.

        In addition, every *other* predecessor of the target is forced to precede M
        (edges ``pred -> M``). This is not a physical requirement but a scheduling
        guarantee for the embedder, which places operations one by one along the 1-D
        macronode index and can never revisit a placement. A feedforward consumer must
        land within ``[i_M + min_dist, i_M + max_dist]`` (``feedforward_distance``), so
        once M is placed the window starts closing. Undershooting is repairable: the
        embedder inserts `through` wirings until the consumer passes ``i_M + min_dist``.
        Overshooting is not: nothing can pull the consumer back under ``i_M + max_dist``.
        Without the extra edges, M and the target's other prerequisites are unordered,
        so the embedder may place M first and then burn through the window while placing
        an arbitrarily long prerequisite chain - stranding the target with no way to
        backtrack. Forcing all other prerequisites before M removes such orders from the
        DAG: the moment M is placed, the target is ready and always fits in the window.
        Direct predecessors suffice because their ancestors precede them by transitivity.
        The cost is that some physically valid schedules are excluded; the benefit is
        that embedding never dead-ends on a feedforward window, and M is scheduled as
        late as possible, which also keeps the result-delivery distance short.

        Raises:
            TypeError: A feedforward parameter's variable is not a ModeMeasuredVariable.
            ValueError: A feedforward parameter references a mode that has no measurement.
        """
        op: GraphOp = self.dep_graph.nodes[target]["op"]

        # Feedforward may sit in the operation's own parameters or in an attached displacement.
        params = list(op.parameters)
        params.extend(self.dep_graph.nodes[target]["displacements"])
        for p in params:
            if not isinstance(p, FeedForward):
                continue
            if not isinstance(p.variable, ModeMeasuredVariable):
                msg = "The type of parameters of operation does not match."
                raise TypeError(msg)
            if p.variable.mode not in self.mode_to_measurement:
                msg = "The mode of ModeMeasuredVariable does not match."
                raise ValueError(msg)
            measurement_node = self.mode_to_measurement[p.variable.mode]

            # Force the target's other prerequisites before M. Adding edges while
            # iterating is safe here: the new edges (pred -> M) never touch the
            # target's predecessor set (M -> target is only added after the loop).
            reachable_from_m = nx.descendants(self.dep_graph, measurement_node)
            for pred_node in self.dep_graph.predecessors(target):
                if pred_node == measurement_node or pred_node in reachable_from_m:
                    # M already precedes this predecessor (e.g. it consumes M's result
                    # itself); adding pred -> M would create a cycle, so keep that order.
                    continue
                self.dep_graph.add_edge(pred_node, measurement_node)

            # The data dependency itself: M's outcome feeds the target's parameter.
            self.dep_graph.add_edge(measurement_node, target)
            self.dep_graph.nodes[target]["has_nlff"] = True

    def _add_op_node(self, op: GraphOp, modes: list[int]) -> None:
        self.dep_graph.add_node(self.next_node_id, op=op, modes=modes, displacements=[])

        if isinstance(op, Measurement):
            self.mode_to_measurement[modes[0]] = self.next_node_id

        # The order is load-bearing: `_apply_feedforward` scans the displacements that
        # `_apply_displacement` attaches, and iterates the predecessor edges that
        # `_apply_dependency` creates.
        self._apply_displacement(self.next_node_id)
        self._apply_dependency(self.next_node_id)
        self._apply_feedforward(self.next_node_id)

        self.next_node_id += 1

    def _add_initialization(self, mode: int, theta: float = pi / 2) -> None:
        init = gops.Initialization(DEFAULT_COORDINATE, theta=theta, initialized_modes=(BLANK_MODE, mode))
        self._add_op_node(init, [mode])

    def _add_displacement(self, disp: cops.Displacement) -> None:
        """Accumulate a circuit displacement instead of adding a node.

        Displacement is an operation in the circuit representation but not an independent
        operation in the graph representation: it stays pending in `last_disp_of_modes`
        (consecutive displacements are summed) until the next operation node on the mode
        picks it up.
        """
        mode = disp.opnd().get_ids()[0]

        prev_disp = self.last_disp_of_modes.get(mode)
        if prev_disp is None:
            self.last_disp_of_modes[mode] = convert_param(disp.x), convert_param(disp.p)
        else:
            x, p = prev_disp
            self.last_disp_of_modes[mode] = (
                _sum_displacements(convert_param(disp.x), x),
                _sum_displacements(convert_param(disp.p), p),
            )

    def _add_displacement_from_graph(self, mode: int, params: tuple[GraphOpParam, GraphOpParam]) -> None:
        prev_disp = self.last_disp_of_modes.get(mode)
        if prev_disp is None:
            self.last_disp_of_modes[mode] = params
        else:
            self.last_disp_of_modes[mode] = (
                _sum_displacements(params[0], prev_disp[0]),
                _sum_displacements(params[1], prev_disp[1]),
            )

    def _reset_op(self, op: GraphOp) -> None:
        """Strip the layout information from the operation, in place."""
        op.macronode = DEFAULT_COORDINATE
        op.swap = False
        op.displacement_k_minus_1 = (0.0, 0.0)
        op.displacement_k_minus_n = (0.0, 0.0)

    def from_circuit(self, circuit: CircuitRepr) -> nx.DiGraph:
        """Build the dependency DAG from a circuit representation.

        Returns:
            nx.DiGraph: The constructed dependency DAG.
        """
        for mode, initial_state in enumerate(circuit.initial_states):
            if isinstance(initial_state, HardwareConstrainedSqueezedState):
                # The `theta` argument is the measurement angle.
                # `initial_state.phi` is the squeezing angle of the initialized mode.
                self._add_initialization(mode, theta=initial_state.phi - pi / 2)
            else:
                self._add_initialization(mode)

        for op_in_circ in circuit:
            if isinstance(op_in_circ, cops.Displacement):
                self._add_displacement(op_in_circ)
                continue

            modes = op_in_circ.opnd().get_ids()

            # convert_op lowers a circuit operation to one or more graph operations.
            # The coordinate is a placeholder, see https://github.com/LiukDiihMieu/mqc3/issues/16
            for op_in_graph in convert_op(op_in_circ, DEFAULT_COORDINATE):
                self._add_op_node(op_in_graph, modes)

        return self.dep_graph

    def from_graph(self, graph: GraphRepr) -> nx.DiGraph:
        """Build the dependency DAG from an already-laid-out graph representation.

        Replays the macronode grid in index order while tracking which mode occupies
        each rail: `modes[h]` is the mode entering row h from the left, and `modes[-1]`
        is the mode on the vertical rail entering the current macronode from above.
        At each macronode the outputs are derived from the operation type (through
        passes the rails straight, swap crosses them, measurement consumes its input,
        initialization creates modes), which recovers the modes every operation acts
        on. Wiring macronodes do not become nodes; edge displacements are
        re-accumulated and attached to the next operation node on the mode.

        Note: the layout information of the input graph's operations is stripped in
        place (see `_reset_op` and issue #18).

        Args:
            graph (GraphRepr): Graph representation to rebuild the dependency DAG from.

        Returns:
            nx.DiGraph: The constructed dependency DAG.
        """
        modes = [BLANK_MODE] * (graph.n_local_macronodes + 1)
        for w in range(graph.n_steps):
            for h in range(graph.n_local_macronodes):
                left, up = modes[h], modes[-1]
                down, right = up, left
                op = graph.get_operation(h, w)
                if left != BLANK_MODE:
                    self._add_displacement_from_graph(left, op.displacement_k_minus_n)
                if up != BLANK_MODE:
                    self._add_displacement_from_graph(up, op.displacement_k_minus_1)
                if op.type() == PbOperation.OPERATION_TYPE_MEASUREMENT:
                    down, right = BLANK_MODE, BLANK_MODE
                elif op.type() == PbOperation.OPERATION_TYPE_INITIALIZATION:
                    down, right = op.initialized_modes
                elif graph.is_swap_macronode(h, w):
                    down, right = left, up
                if op.type() != PbOperation.OPERATION_TYPE_WIRING:
                    self._reset_op(op)
                    self._add_op_node(op, list({left, right, up, down} - {BLANK_MODE}))

                modes[h] = right
                modes[-1] = down

        return self.dep_graph
