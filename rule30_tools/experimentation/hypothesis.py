"""Hypothesis generation for systematic testing."""

from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from rule30_tools.core.bit_array import BitArray


class HypothesisType(Enum):
    """Types of hypotheses."""
    PERIODICITY = "periodicity"
    FREQUENCY = "frequency"
    PATTERN = "pattern"
    STRUCTURAL = "structural"
    INVARIANT = "invariant"


@dataclass
class Evidence:
    """Evidence supporting or refuting a hypothesis."""
    source: str
    data: Dict[str, Any]
    supports: bool
    confidence: float


@dataclass
class Hypothesis:
    """A testable hypothesis about Rule 30."""
    id: str
    statement: str
    hypothesis_type: HypothesisType
    test_method: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    evidence: List[Evidence] = field(default_factory=list)
    
    def add_evidence(self, evidence: Evidence):
        """Add evidence to this hypothesis."""
        self.evidence.append(evidence)
        # Update confidence based on evidence
        if evidence.supports:
            self.confidence = min(1.0, self.confidence + 0.1 * evidence.confidence)
        else:
            self.confidence = max(0.0, self.confidence - 0.1 * evidence.confidence)


class HypothesisGenerator:
    """Systematically generate testable hypotheses about Rule 30."""
    
    def __init__(self):
        self.hypothesis_counter = 0
    
    def generate_periodicity_hypotheses(self, n_tested: int) -> List[Hypothesis]:
        """
        Generate periodicity-related hypotheses.
        
        Args:
            n_tested: Number of steps already tested
            
        Returns:
            List of periodicity hypotheses
        """
        hypotheses = []
        
        # Hypothesis: Sequence becomes periodic after N steps
        for threshold in [n_tested * 10, n_tested * 100, n_tested * 1000]:
            self.hypothesis_counter += 1
            hypotheses.append(Hypothesis(
                id=f"period_{self.hypothesis_counter}",
                statement=f"The center column becomes periodic after {threshold} steps",
                hypothesis_type=HypothesisType.PERIODICITY,
                parameters={'threshold': threshold, 'n_tested': n_tested},
                confidence=0.3
            ))
        
        # Hypothesis: Sequence has near-periodic patterns
        self.hypothesis_counter += 1
        hypotheses.append(Hypothesis(
            id=f"period_{self.hypothesis_counter}",
            statement="The center column has near-periodic patterns with period < 1000",
            hypothesis_type=HypothesisType.PERIODICITY,
            parameters={'max_period': 1000},
            confidence=0.2
        ))
        
        return hypotheses
    
    def generate_frequency_hypotheses(self, observed_data: Dict) -> List[Hypothesis]:
        """
        Generate frequency-related hypotheses.
        
        Args:
            observed_data: Dictionary with observed frequency data
            
        Returns:
            List of frequency hypotheses
        """
        hypotheses = []
        
        ratio_ones = observed_data.get('ratio_ones', 0.5)
        deviation = abs(ratio_ones - 0.5)
        
        # Hypothesis: Ratio converges to 0.5
        self.hypothesis_counter += 1
        hypotheses.append(Hypothesis(
            id=f"freq_{self.hypothesis_counter}",
            statement="The ratio of ones converges to 0.5 as n approaches infinity",
            hypothesis_type=HypothesisType.FREQUENCY,
            parameters={'target_ratio': 0.5, 'current_deviation': deviation},
            confidence=0.7 if deviation < 0.01 else 0.5
        ))
        
        # Hypothesis: Ratio converges to a different value
        if deviation > 0.01:
            self.hypothesis_counter += 1
            hypotheses.append(Hypothesis(
                id=f"freq_{self.hypothesis_counter}",
                statement=f"The ratio of ones converges to {ratio_ones:.4f}",
                hypothesis_type=HypothesisType.FREQUENCY,
                parameters={'target_ratio': ratio_ones, 'current_deviation': deviation},
                confidence=0.3
            ))
        
        # Hypothesis: Convergence rate
        self.hypothesis_counter += 1
        hypotheses.append(Hypothesis(
            id=f"freq_{self.hypothesis_counter}",
            statement="The frequency converges at rate O(1/sqrt(n))",
            hypothesis_type=HypothesisType.FREQUENCY,
            parameters={'convergence_rate': 'sqrt'},
            confidence=0.4
        ))
        
        return hypotheses
    
    def generate_pattern_hypotheses(self, patterns: List[Dict]) -> List[Hypothesis]:
        """
        Generate pattern-related hypotheses.
        
        Args:
            patterns: List of discovered patterns
            
        Returns:
            List of pattern hypotheses
        """
        hypotheses = []
        
        # Hypothesis: Specific patterns appear with certain frequencies
        for pattern in patterns[:5]:  # Top 5 patterns
            self.hypothesis_counter += 1
            hypotheses.append(Hypothesis(
                id=f"pattern_{self.hypothesis_counter}",
                statement=f"Pattern {pattern['pattern']} appears with frequency {pattern['frequency']:.4f}",
                hypothesis_type=HypothesisType.PATTERN,
                parameters={'pattern': pattern['pattern'], 'observed_frequency': pattern['frequency']},
                confidence=0.6
            ))
        
        return hypotheses
    
    def generate_structural_hypotheses(self, structure_data: Dict) -> List[Hypothesis]:
        """
        Generate structural hypotheses.
        
        Args:
            structure_data: Dictionary with structure analysis results
            
        Returns:
            List of structural hypotheses
        """
        hypotheses = []
        
        entropy = structure_data.get('entropy', 0.5)
        
        # Hypothesis: Sequence is random-like
        self.hypothesis_counter += 1
        hypotheses.append(Hypothesis(
            id=f"struct_{self.hypothesis_counter}",
            statement="The center column is statistically random",
            hypothesis_type=HypothesisType.STRUCTURAL,
            parameters={'entropy': entropy},
            confidence=0.6 if entropy > 0.9 else 0.3
        ))
        
        # Hypothesis: Sequence has hidden structure
        self.hypothesis_counter += 1
        hypotheses.append(Hypothesis(
            id=f"struct_{self.hypothesis_counter}",
            statement="The center column has hidden deterministic structure",
            hypothesis_type=HypothesisType.STRUCTURAL,
            parameters={'entropy': entropy, 'compression_ratio': structure_data.get('compression_ratio', 1.0)},
            confidence=0.4 if entropy < 0.7 else 0.2
        ))
        
        return hypotheses
    
    def rank_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """
        Rank hypotheses by priority (confidence, testability, etc.).
        
        Args:
            hypotheses: List of hypotheses to rank
            
        Returns:
            Ranked list of hypotheses
        """
        # Sort by confidence (descending), then by type priority
        type_priority = {
            HypothesisType.PERIODICITY: 1,
            HypothesisType.FREQUENCY: 2,
            HypothesisType.PATTERN: 3,
            HypothesisType.STRUCTURAL: 4,
            HypothesisType.INVARIANT: 5,
        }
        
        ranked = sorted(
            hypotheses,
            key=lambda h: (h.confidence, -type_priority.get(h.hypothesis_type, 10)),
            reverse=True
        )
        
        return ranked

