"""Hypothesis generation and testing framework."""

from rule30_tools.experimentation.hypothesis import Hypothesis, HypothesisGenerator
from rule30_tools.experimentation.experiments import ExperimentRunner, ExperimentResult

try:
    from rule30_tools.experimentation.counterexample import CounterexampleFinder
    __all__ = [
        'Hypothesis',
        'HypothesisGenerator',
        'ExperimentRunner',
        'ExperimentResult',
        'CounterexampleFinder',
    ]
except ImportError:
    __all__ = [
        'Hypothesis',
        'HypothesisGenerator',
        'ExperimentRunner',
        'ExperimentResult',
    ]

