"""
Strategy generator for suggesting high-level approaches.

Helps agents:
- Choose the right strategy for each problem
- Adapt strategies based on what's been tried
- Combine multiple strategies
- Learn which strategies work best
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from rule30_tools.agent.history import ExperimentHistory


class StrategyType(Enum):
    """Types of problem-solving strategies."""
    DIRECT_COMPUTATION = "direct_computation"
    PATTERN_ANALYSIS = "pattern_analysis"
    MATHEMATICAL_ANALYSIS = "mathematical_analysis"
    HYBRID = "hybrid"
    COUNTEREXAMPLE_SEARCH = "counterexample_search"
    PROOF_ATTEMPT = "proof_attempt"


@dataclass
class Strategy:
    """A problem-solving strategy."""
    strategy_type: StrategyType
    description: str
    problem_number: Optional[int]
    steps: List[str]
    estimated_time: str
    success_probability: float
    resource_requirements: Dict[str, Any]


class StrategyGenerator:
    """
    Suggest high-level strategies based on problem analysis.
    
    Enables agents to:
    - Quickly identify promising approaches
    - Avoid strategies that have failed before
    - Combine multiple strategies
    - Adapt based on results
    """
    
    def __init__(self, history: Optional[ExperimentHistory] = None):
        """
        Initialize strategy generator.
        
        Args:
            history: Optional experiment history for learning
        """
        self.history = history
    
    def analyze_problem(self, problem_number: int) -> Dict[str, Any]:
        """
        Analyze a problem and return analysis.
        
        Args:
            problem_number: Problem number (1, 2, or 3)
            
        Returns:
            Problem analysis dictionary
        """
        analysis = {
            'problem_number': problem_number,
            'complexity': self._estimate_complexity(problem_number),
            'best_approaches': self._get_best_approaches(problem_number),
            'known_failures': self._get_known_failures(problem_number),
            'recommended_scale': self._recommend_scale(problem_number)
        }
        
        return analysis
    
    def suggest_strategies(
        self,
        problem_number: Optional[int] = None,
        analysis: Optional[Dict] = None
    ) -> List[Strategy]:
        """
        Suggest strategies for solving a problem.
        
        Args:
            problem_number: Problem number (1, 2, or 3)
            analysis: Optional problem analysis (will compute if not provided)
            
        Returns:
            List of suggested strategies, ranked by priority
        """
        if analysis is None and problem_number:
            analysis = self.analyze_problem(problem_number)
        
        strategies = []
        
        if problem_number == 1:
            # Problem 1: Periodicity
            strategies.extend([
                Strategy(
                    strategy_type=StrategyType.DIRECT_COMPUTATION,
                    description="Compute very large sequences and check for periodicity",
                    problem_number=1,
                    steps=[
                        "Simulate Rule 30 for 10^9+ steps",
                        "Extract center column",
                        "Run periodicity detection",
                        "Check for near-periodic patterns"
                    ],
                    estimated_time="Hours to days",
                    success_probability=0.3,
                    resource_requirements={'memory': 'high', 'cpu': 'high', 'time': 'long'}
                ),
                Strategy(
                    strategy_type=StrategyType.PATTERN_ANALYSIS,
                    description="Look for patterns that might indicate periodicity",
                    problem_number=1,
                    steps=[
                        "Analyze autocorrelation",
                        "Look for repeating blocks",
                        "Check suffix arrays",
                        "Fourier analysis"
                    ],
                    estimated_time="Minutes to hours",
                    success_probability=0.2,
                    resource_requirements={'memory': 'medium', 'cpu': 'medium', 'time': 'medium'}
                ),
                Strategy(
                    strategy_type=StrategyType.MATHEMATICAL_ANALYSIS,
                    description="Prove non-periodicity using mathematical properties",
                    problem_number=1,
                    steps=[
                        "Identify invariants",
                        "Show invariants prevent periodicity",
                        "Use contradiction proof",
                        "Formal verification"
                    ],
                    estimated_time="Days to weeks",
                    success_probability=0.1,
                    resource_requirements={'memory': 'low', 'cpu': 'low', 'time': 'very_long'}
                )
            ])
        
        elif problem_number == 2:
            # Problem 2: Frequency
            strategies.extend([
                Strategy(
                    strategy_type=StrategyType.DIRECT_COMPUTATION,
                    description="Compute large sequences and analyze frequency convergence",
                    problem_number=2,
                    steps=[
                        "Simulate Rule 30 for 10^12+ steps",
                        "Compute running average of ones",
                        "Test convergence rate",
                        "Statistical tests (chi-squared, etc.)"
                    ],
                    estimated_time="Hours to days",
                    success_probability=0.4,
                    resource_requirements={'memory': 'high', 'cpu': 'high', 'time': 'long'}
                ),
                Strategy(
                    strategy_type=StrategyType.MATHEMATICAL_ANALYSIS,
                    description="Prove convergence using limit theorems",
                    problem_number=2,
                    steps=[
                        "Model as stochastic process",
                        "Apply ergodic theorems",
                        "Prove convergence",
                        "Formal verification"
                    ],
                    estimated_time="Days to weeks",
                    success_probability=0.2,
                    resource_requirements={'memory': 'low', 'cpu': 'low', 'time': 'very_long'}
                ),
                Strategy(
                    strategy_type=StrategyType.PATTERN_ANALYSIS,
                    description="Analyze frequency patterns at different scales",
                    problem_number=2,
                    steps=[
                        "Windowed frequency analysis",
                        "Multi-scale analysis",
                        "Look for bias patterns",
                        "Test at multiple scales"
                    ],
                    estimated_time="Minutes to hours",
                    success_probability=0.3,
                    resource_requirements={'memory': 'medium', 'cpu': 'medium', 'time': 'medium'}
                )
            ])
        
        elif problem_number == 3:
            # Problem 3: Randomness
            strategies.extend([
                Strategy(
                    strategy_type=StrategyType.DIRECT_COMPUTATION,
                    description="Run comprehensive randomness test suite",
                    problem_number=3,
                    steps=[
                        "Simulate Rule 30 for 10^12+ steps",
                        "Run all NIST SP 800-22 tests",
                        "Compare to true random",
                        "Test at multiple scales"
                    ],
                    estimated_time="Hours to days",
                    success_probability=0.5,
                    resource_requirements={'memory': 'high', 'cpu': 'high', 'time': 'long'}
                ),
                Strategy(
                    strategy_type=StrategyType.PATTERN_ANALYSIS,
                    description="Look for systematic deviations from randomness",
                    problem_number=3,
                    steps=[
                        "Entropy analysis",
                        "Compression ratio",
                        "Pattern frequency analysis",
                        "Autocorrelation"
                    ],
                    estimated_time="Minutes to hours",
                    success_probability=0.3,
                    resource_requirements={'memory': 'medium', 'cpu': 'medium', 'time': 'medium'}
                ),
                Strategy(
                    strategy_type=StrategyType.COUNTEREXAMPLE_SEARCH,
                    description="Search for patterns that fail randomness tests",
                    problem_number=3,
                    steps=[
                        "Run tests at multiple scales",
                        "Identify failing tests",
                        "Analyze failure patterns",
                        "Find systematic deviations"
                    ],
                    estimated_time="Hours",
                    success_probability=0.4,
                    resource_requirements={'memory': 'medium', 'cpu': 'high', 'time': 'medium'}
                )
            ])
        
        else:
            # General strategies
            strategies.extend([
                Strategy(
                    strategy_type=StrategyType.HYBRID,
                    description="Combine computation and analysis",
                    problem_number=None,
                    steps=[
                        "Run quick tests at small scales",
                        "Identify promising directions",
                        "Scale up successful approaches",
                        "Combine with theoretical analysis"
                    ],
                    estimated_time="Variable",
                    success_probability=0.3,
                    resource_requirements={'memory': 'medium', 'cpu': 'medium', 'time': 'variable'}
                )
            ])
        
        # Adjust based on history
        if self.history:
            strategies = self._adjust_strategies_by_history(strategies, problem_number)
        
        # Rank strategies
        return self.rank_strategies(strategies)
    
    def rank_strategies(self, strategies: List[Strategy]) -> List[Strategy]:
        """
        Rank strategies by priority.
        
        Args:
            strategies: List of strategies
            
        Returns:
            Ranked list of strategies
        """
        # Sort by success probability and resource requirements
        def score(strategy: Strategy) -> float:
            base_score = strategy.success_probability
            
            # Prefer strategies with lower resource requirements
            resource_penalty = {
                'low': 0.0,
                'medium': -0.1,
                'high': -0.2,
                'long': -0.1,
                'very_long': -0.2
            }
            
            for req in strategy.resource_requirements.values():
                if isinstance(req, str):
                    base_score += resource_penalty.get(req, 0.0)
            
            return base_score
        
        return sorted(strategies, key=score, reverse=True)
    
    def execute_strategy(
        self,
        strategy: Strategy,
        api: Any  # Rule30AgentAPI
    ) -> Dict[str, Any]:
        """
        Execute a strategy using the agent API.
        
        Args:
            strategy: Strategy to execute
            api: Rule30AgentAPI instance
            
        Returns:
            Strategy execution results
        """
        results = {
            'strategy': strategy.description,
            'steps_completed': [],
            'results': {},
            'success': False
        }
        
        try:
            # Execute strategy steps
            for step in strategy.steps:
                # Simple step execution (would be more sophisticated in practice)
                if 'Simulate' in step or 'simulate' in step:
                    # Extract step count if possible
                    steps = 1000000  # Default
                    if '10^9' in step:
                        steps = 10**9
                    elif '10^12' in step:
                        steps = 10**12
                    
                    center = api.simulate_rule30(steps)
                    results['steps_completed'].append(step)
                    results['results']['simulation'] = {'steps': steps, 'length': len(center)}
                
                elif 'periodicity' in step.lower():
                    result = api.check_periodicity(steps=1000000)
                    results['steps_completed'].append(step)
                    results['results']['periodicity'] = result
                
                elif 'frequency' in step.lower() or 'convergence' in step.lower():
                    result = api.test_convergence(steps=1000000)
                    results['steps_completed'].append(step)
                    results['results']['frequency'] = result
                
                elif 'randomness' in step.lower() or 'test' in step.lower():
                    result = api.test_randomness(steps=1000000)
                    results['steps_completed'].append(step)
                    results['results']['randomness'] = result
                
                else:
                    results['steps_completed'].append(step)
            
            # Determine success
            if strategy.problem_number:
                problem_result = api.problem1_check(1000000) if strategy.problem_number == 1 else \
                                api.problem2_check(1000000) if strategy.problem_number == 2 else \
                                api.problem3_check(1000000)
                results['success'] = True  # Simplified
        
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _estimate_complexity(self, problem_number: int) -> str:
        """Estimate problem complexity."""
        complexities = {
            1: "High - requires checking infinite sequence",
            2: "High - requires asymptotic analysis",
            3: "Medium - requires statistical testing"
        }
        return complexities.get(problem_number, "Unknown")
    
    def _get_best_approaches(self, problem_number: int) -> List[str]:
        """Get best approaches for a problem."""
        approaches = {
            1: ["Large-scale computation", "Pattern analysis", "Mathematical proof"],
            2: ["Convergence analysis", "Statistical testing", "Asymptotic analysis"],
            3: ["Comprehensive randomness tests", "Pattern detection", "Comparison to random"]
        }
        return approaches.get(problem_number, [])
    
    def _get_known_failures(self, problem_number: int) -> List[str]:
        """Get known failure modes."""
        if self.history:
            failures = self.history.get_failed_approaches(problem_number)
            return [f['type'] for f in failures[:3]]
        return []
    
    def _recommend_scale(self, problem_number: int) -> int:
        """Recommend starting scale for computation."""
        scales = {
            1: 10**6,   # 1 million steps
            2: 10**7,   # 10 million steps
            3: 10**8    # 100 million steps
        }
        return scales.get(problem_number, 10**6)
    
    def _adjust_strategies_by_history(
        self,
        strategies: List[Strategy],
        problem_number: Optional[int]
    ) -> List[Strategy]:
        """Adjust strategy probabilities based on history."""
        if not self.history or not problem_number:
            return strategies
        
        failed_approaches = self.history.get_failed_approaches(problem_number)
        failed_types = {f['type'] for f in failed_approaches}
        
        # Reduce probability of strategies that have failed
        # Note: Strategies don't have experiment_type, this would need to be
        # mapped from strategy description or type
        # For now, we'll skip this adjustment
        pass
        
        return strategies

