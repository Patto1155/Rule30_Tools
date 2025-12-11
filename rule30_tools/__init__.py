"""
Rule 30 Tools: A comprehensive toolkit for solving the Wolfram Rule 30 Prize Problems.

This package provides tools for:
- Simulating Rule 30 at large scales
- Analyzing center column sequences
- Testing hypotheses about periodicity, frequency, and randomness
- Attempting formal proofs
- Agent-friendly APIs for automated problem solving
"""

__version__ = "0.1.0"

from rule30_tools.agent.api import Rule30AgentAPI
from rule30_tools.core.simulator import Rule30Simulator
from rule30_tools.core.bit_array import BitArray

__all__ = [
    'Rule30AgentAPI',
    'Rule30Simulator',
    'BitArray',
]

