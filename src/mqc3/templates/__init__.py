"""Templates of common continuous-variable circuit constructs.

Each template appends a parameterized sub-circuit to an existing
:class:`~mqc3.circuit.CircuitRepr` and returns the same instance,
so templates can be chained and combined with hand-written gates.
"""

from mqc3.templates.bs_net import bs_triangle, gaussian_network, squeezing_layer
from mqc3.templates.qaoa import qaoa

__all__ = ["bs_triangle", "gaussian_network", "qaoa", "squeezing_layer"]
