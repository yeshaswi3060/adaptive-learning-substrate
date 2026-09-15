"""Stage-1 Adaptive Learning Substrate simulator.

The package deliberately uses explicit Python objects and NumPy only.  It does
not depend on an automatic-differentiation framework.
"""

from .environments import DelayedChainEnvironment, MicroProgram3BitEnvironment
from .graph import Edge, FixedSparseEventGraph, ForwardResult
from .learning import CausalCreditFlow, EligibilityTraceBaseline
from .recurrent import RecurrentEventGraph, build_experiment_000_graph

__all__ = [
    "CausalCreditFlow",
    "DelayedChainEnvironment",
    "Edge",
    "EligibilityTraceBaseline",
    "FixedSparseEventGraph",
    "ForwardResult",
    "MicroProgram3BitEnvironment",
    "RecurrentEventGraph",
    "build_experiment_000_graph",
]

__version__ = "0.1.0"
