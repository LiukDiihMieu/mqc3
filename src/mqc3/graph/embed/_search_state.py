from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import mqc3.graph.ops as gops
from mqc3.feedforward import FeedForward
from mqc3.graph import GraphRepr
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.embed.embed import GraphEmbedSettings
from mqc3.graph.ops import ModeMeasuredVariable

if TYPE_CHECKING:
    from collections.abc import Generator

    from mqc3.graph import Operation as GraphOp
    from mqc3.graph.embed.dep_dag import DependencyDAG


class SearchState:
    __slots__ = [
        "_left",
        "_mode_row",
        "_op_pos_dict",
        "_swap_pos_set",
        "_up",
        "dep_dag",
        "feedforward_distance",
        "index",
        "mode_to_measurement",
        "n_local_macronodes",
    ]

    def __init__(self, dep_dag: DependencyDAG | None, settings: GraphEmbedSettings) -> None:
        """Search state for the embedding.

        Args:
            dep_dag (DependencyDAG): The dependency DAG to embed. If None, should be set later.
            settings (GraphEmbedConfig): The settings of embedding.

        Attributes:
            _mode_row (dict[int, int]): The rail position of each mode currently in the frontier.
                                         - If the mode is on the left rail (`_left[h] == mode`): its row `h`.
                                         - If the mode is on the up rail (`_up == mode`): `-1` (a sentinel, not a row).
                                         - If the mode is not in the frontier: the key is absent.
        """
        if dep_dag is None:
            return

        # Constant
        self.n_local_macronodes = settings.n_local_macronodes
        self.feedforward_distance = settings.feedforward_distance
        self.dep_dag = dep_dag
        self.mode_to_measurement: dict[int, int] = {}
        for node_id, node_data in self.dep_dag.dag.nodes.items():
            if isinstance(node_data["op"], gops.Measurement):
                self.mode_to_measurement[node_data["modes"][0]] = node_id

        # Position of operations
        self._swap_pos_set: set[tuple[int, int]] = set()
        self._op_pos_dict: dict[int, tuple[int, int]] = {}

        # Macronode index
        self.index = 0

        # Mode
        self._up = BLANK_MODE
        self._left = [BLANK_MODE] * self.n_local_macronodes
        self._mode_row: dict[int, int] = {}

    def get_left_mode(self, index: int) -> int:
        """Get the left mode of the `index`-th macronode.

        Args:
            index (int): Index of the macronode.

        Returns:
            int: The left mode.
        """
        return self._left[index % self.n_local_macronodes]

    def get_coord(self, index: int) -> tuple[int, int]:
        """Get the coordinate of the `index`-th macronode.

        Args:
            index (int): Index of the macronode.

        Returns:
            tuple[int, int]: Coordinate.
        """
        w, h = divmod(index, self.n_local_macronodes)
        return h, w

    def get_index(self, coord: tuple[int, int]) -> int:
        """Get the index of coordinate.

        Args:
            coord (int): coordinate of the macronode

        Returns:
            int: index
        """
        return coord[1] * self.n_local_macronodes + coord[0]

    def search_mode(self, mode: int) -> int:
        """Get the index of the macronode with the specified mode after the current macronode.

        Args:
            mode (int): Mode.

        Returns:
            int: Macronode index.

        Raises:
            RuntimeError: There is no macronode with the specified mode.
        """
        if self._up == mode:
            return self.index

        from_left = self.search_mode_on_left(mode)
        if from_left is not None:
            return from_left

        msg = f"There is no mode: {mode}."
        raise RuntimeError(msg)

    def search_mode_on_left(self, mode: int) -> int | None:
        """Get the index of the macronode, that the specified mode come from left, after the current macronode.

        Args:
            mode (int): Mode.

        Returns:
            int | None: Macronode index or None if not found.
        """
        if mode != BLANK_MODE:  # normal mode, use _mode_row to look up
            if mode not in self._mode_row or self._mode_row[mode] == -1:
                return None
            h = self._mode_row[mode]
            return self.index + (h - self.index) % self.n_local_macronodes

        # (else) blank mode, go over all rows from the current one
        for i in range(self.index, self.index + self.n_local_macronodes):
            if self.get_left_mode(i) == mode:
                return i
        return None

    def blank_mode_count(self) -> int:
        return self.n_local_macronodes + 1 - len(self._mode_row)

    def calc_min_index_to_place_op(self, op_node_id: int) -> int | None:
        """Calculate the smallest macronode index at which `op` could be placed.

        This is a lower bound given the current frontier (the theoretical earliest
        position); the placement actually chosen by the `prepare_*` methods may be later.

        Returns:
            int | None: The smallest macronode index at which `op` could be placed,
                or None if the op cannot be placed currently.
        """
        node_data = self.dep_dag.dag.nodes[op_node_id]

        # 1. initialization
        if isinstance(node_data["op"], gops.Initialization):
            found_first_blank = False
            if self._up == BLANK_MODE:
                found_first_blank = True
            for i in range(self.index, self.index + self.n_local_macronodes):
                if self._left[i % self.n_local_macronodes] == BLANK_MODE:
                    if found_first_blank:
                        return i
                    found_first_blank = True
            return None  # Two blank modes are required to place initialization.
        modes = node_data["modes"]

        # 2. single-mode operation
        if len(modes) == 1:
            mode = modes[0]
            blank_index = self.search_mode(BLANK_MODE)
            target_index = self.search_mode(mode)
            return max(blank_index, target_index)

        # 3. two-mode operation
        target_mode1, target_mode2 = modes
        target_mode1_index = self.search_mode(target_mode1)
        target_mode2_index = self.search_mode(target_mode2)
        target_mode1_row = self.get_coord(target_mode1_index)[0]
        target_mode2_row = self.get_coord(target_mode2_index)[0]
        upper_index, lower_index = (
            (target_mode1_index, target_mode2_index)
            if target_mode1_row < target_mode2_row
            else (target_mode2_index, target_mode1_index)
        )
        if lower_index < upper_index:
            return lower_index + self.n_local_macronodes
        return lower_index

    def advance_index(self) -> None:
        """Increment index and call functions."""
        self.index += 1

    def remove_current_modes(self) -> None:
        """Remove modes on current macronode from SearchState."""
        self._mode_row.pop(self._up, None)
        self._up = BLANK_MODE

        h, _ = self.get_coord(self.index)
        self._mode_row.pop(self._left[h], None)
        self._left[h] = BLANK_MODE

    def insert_through(self, reps: int = 1, *, without_leap: bool = False) -> None:
        """Insert `through` operations.

        If "without_leap" is True, it also inserts minimum required `swap` operations to avoid leap of modes.

        Args:
            reps (int): The number of iterations.
            without_leap (bool): Avoid leap of modes.
        """
        for _ in range(reps):
            if without_leap and self._up != BLANK_MODE and self.get_left_mode(self.index) == BLANK_MODE:
                self.insert_swap()
            else:
                self.advance_index()

    def insert_swap(self) -> None:
        """Insert the `swap` operation.

        Raises:
            RuntimeError: The swap operation is already placed
        """
        coord = self.get_coord(self.index)
        if coord in self._swap_pos_set:
            msg = "The swap operation is already placed."
            raise RuntimeError(msg)
        self._swap_pos_set.add(coord)

        h = coord[0]
        # actual "SWAP" happens here
        self._up, self._left[h] = self._left[h], self._up

        # update status
        if self._up != BLANK_MODE:
            self._mode_row[self._up] = -1
        if self._left[h] != BLANK_MODE:
            self._mode_row[self._left[h]] = h

        self.advance_index()

    def insert_initialization(self, op_node_id: int, *, swap_in_op: bool) -> None:
        """Insert initialization operation.

        Args:
            op_node_id (int): The dependency-DAG node id of the operation to insert.
            swap_in_op (bool): Whether the swap operation is included in the operation.
        """
        op = self.dep_dag.dag.nodes[op_node_id]["op"]
        coord = self.get_coord(self.index)
        h = coord[0]
        if op.initialized_modes[0] != BLANK_MODE:
            new_mode = op.initialized_modes[0]
            self._up = new_mode
            self._mode_row[new_mode] = -1
        if op.initialized_modes[1] != BLANK_MODE:
            new_mode = op.initialized_modes[1]
            self._left[h] = new_mode
            self._mode_row[new_mode] = h
        self._op_pos_dict[op_node_id] = coord
        if swap_in_op:
            self.insert_swap()
        else:
            self.insert_through()

    def insert_single_mode_operation(self, op_node_id: int, *, swap_in_op: bool) -> None:
        """Insert the single-mode operation.

        Args:
            op_node_id (int): The dependency-DAG node id of the operation to insert.
            swap_in_op (bool): Whether the swap operation is included in the operation.

        Raises:
            RuntimeError: Modes are not prepared correctly
        """
        left = self.get_left_mode(self.index)
        up = self._up

        node_data = self.dep_dag.dag.nodes[op_node_id]
        modes = node_data["modes"]
        if BLANK_MODE not in {left, up}:
            msg = "One of the input modes of the current macronode must be blank mode."
            raise RuntimeError(msg)
        if modes[0] not in {left, up}:
            msg = "The input mode of the operation must match the non-blank mode of the macronode."
            raise RuntimeError(msg)

        op = node_data["op"]
        self._op_pos_dict[op_node_id] = self.get_coord(self.index)

        # remove measured modes
        if isinstance(op, gops.Measurement):
            self.remove_current_modes()
        # Up mode should be kept blank as possible
        if swap_in_op:
            self.insert_swap()
        else:
            self.insert_through()

    def insert_two_mode_operation(self, op_node_id: int, *, swap_in_op: bool) -> None:
        """Insert the two-mode operation.

        Args:
            op_node_id (int): The dependency-DAG node id of the operation to insert.
            swap_in_op (bool): Whether the swap operation is included in the operation.

        Raises:
            RuntimeError: Input modes of the operation and the modes of the macronode do not match.
        """
        modes = self.dep_dag.dag.nodes[op_node_id]["modes"]

        left = self.get_left_mode(self.index)
        up = self._up

        if modes[0] not in {left, up} or modes[1] not in {left, up}:
            msg = "Both input modes of the operation must match the macronode's input modes (left, up)."
            raise RuntimeError(msg)

        self._op_pos_dict[op_node_id] = self.get_coord(self.index)
        if swap_in_op:
            self.insert_swap()
        else:
            self.insert_through()

    def is_all_dependency_resolved(self, op_node_id: int) -> bool:
        return all(n in self._op_pos_dict for n in self.dep_dag.dag.predecessors(op_node_id))

    def is_already_placed(self, op_node_id: int) -> bool:
        return op_node_id in self._op_pos_dict

    def calc_placeable_range(self, op_node_id: int) -> tuple[int, int] | None:
        """Calculate the placeable range of an operation.

        Returns:
            tuple[int, int] | None: the lower and upper bound of the placeable range, inclusive.
                if the operation cannot be placed currently, returns None.

        Raises:
            TypeError: The type of parameters of operation does not match
            ValueError: The mode of ModeMeasuredVariable does not match
        """
        if not self.is_all_dependency_resolved(op_node_id):
            return None  # Some of dependencies are not resolved

        inf = 10**9
        min_index = -1
        max_index = inf
        op = self.dep_dag.dag.nodes[op_node_id]["op"]
        for p in op.parameters:
            if not isinstance(p, FeedForward):
                continue
            if not isinstance(p.variable, ModeMeasuredVariable):
                msg = "The type of parameters of operation does not match."
                raise TypeError(msg)
            if p.variable.mode not in self.mode_to_measurement:
                msg = "The mode of ModeMeasuredVariable does not match."
                raise ValueError(msg)
            # the measurement which provides the input of the feedforward parameter
            measurement_node_id = self.mode_to_measurement[p.variable.mode]

            # Check if the measurement for variable is already placed
            from_position = self._op_pos_dict.get(measurement_node_id)
            if from_position is None:
                return None  # Some of operations required for feedforwarding are not placed
            if isinstance(op, gops.Initialization) and self.blank_mode_count() <= 1:
                return None  # There must be two or more blank modes to place initialization
            from_index = self.get_index(from_position)
            min_index = max(min_index, from_index + self.feedforward_distance[0])
            max_index = min(max_index, from_index + self.feedforward_distance[1])
        return (min_index, max_index)

    def place_operation(self, op_node_id: int, *, swap_in_op: bool) -> None:
        """Place the operation into the current state, advancing the frontier as needed.

        Advances the macronode frontier (inserting `through`s) until the earliest
        geometrically feasible index satisfies the feedforward lower bound, verifies it
        still fits within the feedforward upper bound, then arranges the input modes
        (`prepare_*`) and records the operation (`insert_*`).

        Args:
            op_node_id (int): The dependency-DAG node id of the operation to place.
            swap_in_op (bool): Whether the swap operation is included in the operation.

        Raises:
            RuntimeError: The operation is already placed, cannot be placed in the current
                state, or cannot fit within its feedforward window.
        """
        if op_node_id in self._op_pos_dict:
            msg = "This operation has been already placed"
            raise RuntimeError(msg)
        placeable_range = self.calc_placeable_range(op_node_id)
        if placeable_range is None:
            msg = "This operation is not placeable currently."
            raise RuntimeError(msg)
        min_index, max_index = placeable_range

        # through until min_index
        while True:
            # start from theoretical lower bound
            placement_index = self.calc_min_index_to_place_op(op_node_id)
            if placement_index is None:
                msg = "This operation is not placeable currently."
                raise RuntimeError(msg)
            if placement_index >= min_index:
                break
            self.insert_through()
        if placement_index > max_index:
            msg = "Failed in insertion of operation."
            raise RuntimeError(msg)
        modes = self.dep_dag.dag.nodes[op_node_id]["modes"]
        op = self.dep_dag.dag.nodes[op_node_id]["op"]
        if isinstance(op, gops.Initialization):
            self.prepare_initialization()
            self.insert_initialization(op_node_id, swap_in_op=swap_in_op)
        elif len(modes) == 1:
            self.prepare_single_mode_operation(modes[0])
            self.insert_single_mode_operation(op_node_id, swap_in_op=swap_in_op)
        else:
            self.prepare_two_mode_operation(modes[0], modes[1])
            self.insert_two_mode_operation(op_node_id, swap_in_op=swap_in_op)

    def generate_next_states(self) -> Generator[SearchState, None, None]:
        """Yield the states reachable by placing one more operation.

        For each operation whose dependencies are resolved and that is not yet placed,
        yields a copy of this state with that operation placed - once without and once
        with a in-place swap (`swap_in_op` False/True).

        As an early prune, if any still-unplaced operation has already missed its
        feedforward window (the current frontier index is past its latest allowed
        placement index), this state can never be completed, so nothing is yielded.

        Yields:
            SearchState: A copy of this state with one more operation placed.
        """
        candidates = [
            op_node_id
            for op_node_id in self.dep_dag.dag.nodes
            if not self.is_already_placed(op_node_id) and self.is_all_dependency_resolved(op_node_id)
        ]

        # Phase 1: if any candidate has already missed its feedforward window, this state is a
        # dead end (that op can never be placed), so yield no successors.
        for op_node_id in candidates:
            placeable_range = self.calc_placeable_range(op_node_id)
            if placeable_range is not None and self.index > placeable_range[1]:
                return

        # Phase 2: otherwise branch out, placing each candidate once without and once with SWAP.
        for op_node_id, swap_in_op in product(candidates, [False, True]):
            next_state = self.copy()
            next_state.place_operation(op_node_id, swap_in_op=swap_in_op)
            yield next_state

    def output_graph(self) -> GraphRepr:  # noqa: C901
        """Output current state as `GraphRepr`.

        Returns:
            GraphRepr: Current state.
        """
        n_steps = 0
        for _, w in self._op_pos_dict.values():
            n_steps = max(n_steps, w + 1)
        for _, w in self._swap_pos_set:
            n_steps = max(n_steps, w + 1)

        graph = GraphRepr(self.n_local_macronodes, n_steps)

        for ind, coord in self._op_pos_dict.items():
            op: GraphOp = self.dep_dag.dag.nodes[ind]["op"]
            op.macronode = coord
            graph.place_operation(op)

        for coord in self._swap_pos_set:
            op = graph.get_operation(*coord)
            if isinstance(op, gops.Initialization):  # Initialization does not have swap attribute
                op.initialized_modes[0], op.initialized_modes[1] = op.initialized_modes[1], op.initialized_modes[0]
            else:
                op.swap = True

        io_modes_dict = None  # to improve performance, dict is created only when necessary

        for ind, coord in self._op_pos_dict.items():
            op = graph.get_operation(*coord)
            for mode, disps in self.dep_dag.dag.nodes[ind]["displacements"]:
                if io_modes_dict is None:
                    io_modes_dict = graph.io_modes_dict()
                left, up, _, _ = io_modes_dict[coord]
                if mode == left:
                    op.displacement_k_minus_n = disps
                elif mode == up:
                    op.displacement_k_minus_1 = disps

        return graph

    def evaluate(self) -> int:
        """Return the evaluate value of current state (larger means better)."""
        return len(self._op_pos_dict)

    def __lt__(self, other: SearchState) -> int:
        return self.evaluate() < other.evaluate()

    def is_all_done(self) -> bool:
        """Return true if all operations are done."""
        return all(n in self._op_pos_dict for n in self.dep_dag.dag.nodes)

    def prepare_initialization(self) -> None:
        """Use through or swap to create a macronode with input modes as (blank, blank).

        Raises:
            RuntimeError: Two blank nodes are required to place initialization.
        """
        first_blank_index = self.search_mode(BLANK_MODE)
        self.insert_through(first_blank_index - self.index)
        if self._up == BLANK_MODE and self.get_left_mode(self.index) == BLANK_MODE:
            return
        if self._up != BLANK_MODE:
            self.insert_swap()
        second_blank_index = self.search_mode_on_left(BLANK_MODE)
        if second_blank_index is None:
            msg = "Two blank nodes are required to place initialization."
            raise RuntimeError(msg)
        self.insert_through(second_blank_index - self.index)

    def prepare_single_mode_operation(self, target_mode: int) -> None:
        """Use through or swap to create a macronode with input modes as (blank, target).

        Args:
            target_mode (int): Mode used for operation. Must be a real (non-blank) mode.

        Raises:
            ValueError: `target_mode` is the blank mode.
            RuntimeError: Mode is not found.
        """
        if target_mode == BLANK_MODE:
            msg = "`target_mode` must be a real (non-blank) mode."
            raise ValueError(msg)

        blank_index = self.search_mode(BLANK_MODE)
        target_left_index = self.search_mode_on_left(target_mode)  # if already in _up, returns None

        # target is neither on the up-rail nor the left-rail, i.e. not in the current frontier
        if self._up != target_mode and target_left_index is None:
            msg = f"`target_mode` {target_mode} is not found."
            raise RuntimeError(msg)

        # up occupied by another mode, perform SWAP when encounters the first blank or target
        if self._up not in {target_mode, BLANK_MODE}:
            next_index = min(target_left_index, blank_index)
            self.insert_through(next_index - self.index, without_leap=True)
            self.insert_swap()  # resulting blank or target in _up

        # up should be occupied by target or BLANK now
        if self._up not in {target_mode, BLANK_MODE}:
            msg = f"`target_mode` {target_mode} is not found."
            raise RuntimeError(msg)

        # final case: up is already prepared into either blank or target, prepare left.
        next_index = blank_index if target_left_index is None else max(target_left_index, blank_index)
        self.insert_through(next_index - self.index)

    def prepare_two_mode_operation(self, target_mode1: int, target_mode2: int) -> None:
        """Place `through` or `swap` operations to make space for the target two-mode operation.

        Args:
            target_mode1 (int): The first input mode of the operation.
            target_mode2 (int): The second input mode of the operation.

        Raises:
            RuntimeError: The target two-mode operation cannot be placed.
        """
        # both already prepared
        if (self._up == target_mode1 and self.get_left_mode(self.index) == target_mode2) or (
            self._up == target_mode2 and self.get_left_mode(self.index) == target_mode1
        ):
            return

        # if a target sits on the up-rail but is the lower one (the other target is above it),
        # swap it down to the left rail so that both targets end up on the left
        if self._up in {target_mode1, target_mode2}:
            other_mode = target_mode2 if self._up == target_mode1 else target_mode1
            target_other_left_index = self.search_mode_on_left(other_mode)
            if target_other_left_index is None:
                msg = f"{other_mode} is not found."
                raise RuntimeError(msg)
            if self.get_coord(target_other_left_index)[0] < self.get_coord(self.index)[0]:
                # the other mode comes before, swap target in up to left
                self.insert_swap()

        # if both targets are on the left, lift the upper one (smaller row) onto the up-rail
        if self._up not in {target_mode1, target_mode2}:
            target_mode1_left_index = self.search_mode_on_left(target_mode1)
            if target_mode1_left_index is None:
                msg = f"{target_mode1} is not found."
                raise RuntimeError(msg)

            target_mode2_left_index = self.search_mode_on_left(target_mode2)
            if target_mode2_left_index is None:
                msg = f"{target_mode2} is not found."
                raise RuntimeError(msg)

            # almost the same as using _mode_row? duplicated.
            target_mode1_row = self.get_coord(target_mode1_left_index)[0]
            target_mode2_row = self.get_coord(target_mode2_left_index)[0]
            upper_index = target_mode1_left_index if target_mode1_row < target_mode2_row else target_mode2_left_index
            self.insert_through(upper_index - self.index, without_leap=True)  # advance to the swap point
            self.insert_swap()

        # one target is now on the up-rail and is the upper one; advance until the other target appears on the left
        other_mode = target_mode2 if self._up == target_mode1 else target_mode1
        target_other_left_index = self.search_mode_on_left(other_mode)
        if target_other_left_index is None:
            msg = f"{other_mode} is not found."
            raise RuntimeError(msg)
        self.insert_through(target_other_left_index - self.index)  # advance to the final placement macronode

    def copy(self) -> SearchState:
        """Copy the search state.

        Returns:
            SearchState: Copied search state.
        """
        copied = SearchState(None, GraphEmbedSettings(self.n_local_macronodes, self.feedforward_distance))

        copied._left = self._left.copy()
        copied._mode_row = self._mode_row.copy()
        copied._swap_pos_set = self._swap_pos_set.copy()
        copied._op_pos_dict = self._op_pos_dict.copy()
        copied._up = self._up
        copied.index = self.index
        copied.n_local_macronodes = self.n_local_macronodes
        copied.feedforward_distance = self.feedforward_distance
        copied.dep_dag = self.dep_dag
        copied.mode_to_measurement = self.mode_to_measurement

        return copied
