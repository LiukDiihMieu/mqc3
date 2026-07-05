"""Convert circuit representation to graph representation using beam search."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heappushpop
from typing import TYPE_CHECKING

from mqc3.graph.embed._search_state import SearchState
from mqc3.graph.embed.embed import GraphEmbedder, GraphEmbedResult, GraphEmbedSettings

if TYPE_CHECKING:
    from .dep_dag import DependencyDAG


@dataclass
class BeamSearchEmbedSettings(GraphEmbedSettings):
    """Settings for embedding a dependency DAG into a graph representation with beam search strategy."""

    beam_width: int = 10
    """The number of candidate solutions to keep at each step of the beam search.
    A larger value leads to a more precise solution but increases computation time."""


class BeamSearchEmbedder(GraphEmbedder):
    """Embed a dependency DAG into a graph representation with beam search strategy."""

    _settings: BeamSearchEmbedSettings

    def __init__(self, settings: BeamSearchEmbedSettings) -> None:
        """Initialize embedder with settings.

        Args:
            settings (BeamSearchEmbedSettings): Settings for embedding a dependency DAG into a GraphRepr.
        """
        super().__init__(settings)

    def _push_into_beam(self, search_nodes: list[list[SearchState]], state: SearchState) -> None:
        """Insert `state` into the beam at its macronode index.

        Grows `search_nodes` so the index exists, then keeps at most `beam_width` best
        states in that beam (a min-heap ordered by `SearchState.evaluate`).

        Args:
            search_nodes (list[list[SearchState]]): Beams indexed by macronode index.
            state (SearchState): The state to insert.
        """
        while len(search_nodes) <= state.index:
            search_nodes.append([])
        beam = search_nodes[state.index]
        if len(beam) < self._settings.beam_width:
            heappush(beam, state)  # only add
        else:
            heappushpop(beam, state)  # remove the worst one and then add

    def _embed_impl(self, dep_dag: DependencyDAG) -> GraphEmbedResult:
        """Embed a Dependency DAG into a graph representation with beam search strategy.

        Args:
            dep_dag: Dependency DAG to embed.

        Returns:
            GraphEmbedResult: Result of the embedding.

        Raises:
            RuntimeError: If the embedding fails.
        """
        # search_nodes[i] is the beam for macronode index i.
        # search_nodes[i] has length "beam_width"
        search_nodes: list[list[SearchState]] = [[SearchState(dep_dag, self._settings)]]
        next_macronode_index = 0

        while True:
            for bs_state in search_nodes[next_macronode_index]:
                if bs_state.is_all_done():
                    return GraphEmbedResult(graph=bs_state.output_graph())
                for next_state in bs_state.generate_next_states():
                    self._push_into_beam(search_nodes, next_state)
            next_macronode_index += 1
            if next_macronode_index == len(search_nodes):
                msg = "Failed to convert the circuit to graph representation using beam search."
                raise RuntimeError(msg)
