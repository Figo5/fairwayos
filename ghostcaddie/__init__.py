"""FairwayOS — a golf shot-analytics engine vertical slice.

The internal ``ghostcaddie`` package name remains unchanged for compatibility.

Pure Python 3.9 stdlib only. Analytics engine (geometry/hazards/dispersion/
expected_strokes/simulation/decision) is fully separated from the single
rendering module (overlay); the pipeline wires them in strict order.
"""

__version__ = "0.1.0"
