"""
High-level API for coding agents to use Rule 30 tools.

This API is designed for rapid experimentation and iteration:
- Simple, intuitive function calls
- Automatic result caching
- Built-in experiment tracking
- Quick hypothesis testing
"""

from typing import Optional, Dict, List
from rule30_tools.core.simulator import Rule30Simulator
from rule30_tools.core.bit_array import BitArray
from rule30_tools.core.center_column import CenterColumnExtractor
from rule30_tools.analysis.periodicity import PeriodicityDetector
from rule30_tools.analysis.frequency import FrequencyAnalyzer
from rule30_tools.analysis.randomness import RandomnessTestSuite
from rule30_tools.analysis.patterns import PatternRecognizer
from rule30_tools.agent.history import ExperimentHistory


class Rule30AgentAPI:
    """
    Simple, intuitive API for agents to use Rule 30 tools.
    
    Designed for rapid iteration:
    - One-line function calls for common tasks
    - Automatic caching of expensive computations
    - Built-in experiment tracking
    - Quick hypothesis testing
    """
    
    def __init__(self, enable_history: bool = True, cache_dir: Optional[str] = None):
        """
        Initialize the agent API.
        
        Args:
            enable_history: Whether to track experiment history
            cache_dir: Optional directory for caching results
        """
        self.simulator = Rule30Simulator()
        self.periodicity_detector = PeriodicityDetector()
        self.frequency_analyzer = FrequencyAnalyzer()
        self.randomness_suite = RandomnessTestSuite()
        self.pattern_recognizer = PatternRecognizer()
        self.extractor = CenterColumnExtractor()
        
        self.history = ExperimentHistory() if enable_history else None
        self.cache_dir = cache_dir
        self._cache: Dict[str, Any] = {}
    
    def simulate_rule30(self, steps: int, use_cache: bool = True) -> BitArray:
        """
        Simulate Rule 30 for given number of steps, return center column.
        
        Args:
            steps: Number of steps to simulate
            use_cache: Whether to use cached results if available
            
        Returns:
            Center column sequence as BitArray
        """
        cache_key = f"simulate_{steps}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        center = self.simulator.compute_center_column(steps)
        
        if use_cache:
            self._cache[cache_key] = center
        
        if self.history:
            self.history.log_simulation(steps, len(center))
        
        return center
    
    def check_periodicity(
        self,
        sequence: Optional[BitArray] = None,
        steps: Optional[int] = None,
        max_period: Optional[int] = None
    ) -> Dict:
        """
        Check if sequence is periodic.
        
        Args:
            sequence: Optional BitArray to check (if None, will simulate)
            steps: Number of steps to simulate if sequence not provided
            max_period: Maximum period to check
            
        Returns:
            Dictionary with periodicity analysis
        """
        if sequence is None:
            if steps is None:
                steps = 10000  # Default
            sequence = self.simulate_rule30(steps)
        
        result = self.periodicity_detector.check_aperiodicity(sequence)
        
        if self.history:
            self.history.log_analysis('periodicity', result)
        
        return result
    
    def analyze_frequency(
        self,
        sequence: Optional[BitArray] = None,
        steps: Optional[int] = None
    ) -> Dict:
        """
        Analyze frequency distribution.
        
        Args:
            sequence: Optional BitArray to analyze
            steps: Number of steps to simulate if sequence not provided
            
        Returns:
            Dictionary with frequency statistics
        """
        if sequence is None:
            if steps is None:
                steps = 10000
            sequence = self.simulate_rule30(steps)
        
        result = self.frequency_analyzer.compute_frequencies(sequence)
        
        if self.history:
            self.history.log_analysis('frequency', result)
        
        return result
    
    def test_convergence(
        self,
        sequence: Optional[BitArray] = None,
        steps: Optional[int] = None
    ) -> Dict:
        """
        Test if frequency converges to 0.5.
        
        Args:
            sequence: Optional BitArray to test
            steps: Number of steps to simulate if sequence not provided
            
        Returns:
            Dictionary with convergence analysis
        """
        if sequence is None:
            if steps is None:
                steps = 100000
            sequence = self.simulate_rule30(steps)
        
        result = self.frequency_analyzer.test_convergence_to_0_5(sequence)
        
        if self.history:
            self.history.log_analysis('convergence', result)
        
        return result
    
    def test_randomness(
        self,
        sequence: Optional[BitArray] = None,
        steps: Optional[int] = None,
        tests: Optional[List[str]] = None
    ) -> Dict:
        """
        Run randomness tests on sequence.
        
        Args:
            sequence: Optional BitArray to test
            steps: Number of steps to simulate if sequence not provided
            tests: Optional list of specific tests to run
            
        Returns:
            Dictionary with randomness test results
        """
        if sequence is None:
            if steps is None:
                steps = 1000000  # Need more steps for randomness tests
            sequence = self.simulate_rule30(steps)
        
        result = self.randomness_suite.run_all_tests(sequence, tests)
        
        if self.history:
            self.history.log_analysis('randomness', result)
        
        return result
    
    def find_patterns(
        self,
        sequence: Optional[BitArray] = None,
        steps: Optional[int] = None,
        min_length: int = 2,
        max_length: int = 10
    ) -> List[Dict]:
        """
        Find repeating patterns in sequence.
        
        Args:
            sequence: Optional BitArray to analyze
            steps: Number of steps to simulate if sequence not provided
            min_length: Minimum pattern length
            max_length: Maximum pattern length
            
        Returns:
            List of discovered patterns
        """
        if sequence is None:
            if steps is None:
                steps = 10000
            sequence = self.simulate_rule30(steps)
        
        patterns = self.pattern_recognizer.find_patterns(sequence, min_length, max_length)
        
        if self.history:
            self.history.log_analysis('patterns', {'patterns': patterns})
        
        return patterns
    
    def problem1_check(self, steps: int) -> Dict:
        """
        Check Problem 1: Does center column remain non-periodic?
        
        Quick one-liner for agents to test Problem 1.
        
        Args:
            steps: Number of steps to check
            
        Returns:
            Dictionary with periodicity check results
        """
        center = self.simulate_rule30(steps)
        periodicity = self.check_periodicity(center)
        
        result = {
            'steps_checked': steps,
            'is_periodic': periodicity['is_periodic'],
            'period': periodicity['period'],
            'confidence': periodicity['confidence'],
            'max_checked': periodicity['max_checked']
        }
        
        if self.history:
            self.history.log_problem_check(1, steps, result)
        
        return result
    
    def problem2_check(self, steps: int) -> Dict:
        """
        Check Problem 2: Do colors occur equally often on average?
        
        Quick one-liner for agents to test Problem 2.
        
        Args:
            steps: Number of steps to check
            
        Returns:
            Dictionary with frequency analysis results
        """
        center = self.simulate_rule30(steps)
        frequency = self.analyze_frequency(center)
        convergence = self.test_convergence(center)
        chi2 = self.frequency_analyzer.chi_squared_test(center)
        
        result = {
            'steps_checked': steps,
            'ratio_ones': frequency['ratio_ones'],
            'deviation_from_0.5': frequency['deviation_from_0.5'],
            'converging': convergence['converging'],
            'convergence_rate': convergence['convergence_rate'],
            'chi_squared_result': chi2
        }
        
        if self.history:
            self.history.log_problem_check(2, steps, result)
        
        return result
    
    def problem3_check(self, steps: int, tests: Optional[List[str]] = None) -> Dict:
        """
        Check Problem 3: Does center column pass randomness tests?
        
        Quick one-liner for agents to test Problem 3.
        
        Args:
            steps: Number of steps to check
            tests: Optional list of specific tests to run
            
        Returns:
            Dictionary with randomness test results
        """
        center = self.simulate_rule30(steps)
        randomness = self.test_randomness(center, tests=tests)
        
        summary = randomness.get('_summary', {})
        result = {
            'steps_checked': steps,
            'tests_passed': summary.get('tests_passed', 0),
            'tests_total': summary.get('tests_total', 0),
            'pass_rate': summary.get('pass_rate', 0.0),
            'test_results': {k: v for k, v in randomness.items() if k != '_summary'}
        }
        
        if self.history:
            self.history.log_problem_check(3, steps, result)
        
        return result
    
    def quick_test_hypothesis(
        self,
        hypothesis_statement: str,
        test_steps: int = 10000
    ) -> Dict:
        """
        Quickly test a hypothesis statement.
        
        This is a convenience method for agents to rapidly test ideas.
        
        Args:
            hypothesis_statement: Natural language hypothesis
            test_steps: Number of steps to use for testing
            
        Returns:
            Dictionary with test results
        """
        center = self.simulate_rule30(test_steps)
        
        # Simple keyword-based hypothesis testing
        hypothesis_lower = hypothesis_statement.lower()
        
        if 'periodic' in hypothesis_lower or 'period' in hypothesis_lower:
            result = self.check_periodicity(center)
            return {
                'hypothesis': hypothesis_statement,
                'test_type': 'periodicity',
                'result': result,
                'supports': not result['is_periodic']
            }
        elif 'frequency' in hypothesis_lower or 'ratio' in hypothesis_lower or 'equal' in hypothesis_lower:
            result = self.analyze_frequency(center)
            convergence = self.test_convergence(center)
            return {
                'hypothesis': hypothesis_statement,
                'test_type': 'frequency',
                'result': result,
                'convergence': convergence,
                'supports': convergence['converging'] and result['deviation_from_0.5'] < 0.01
            }
        elif 'random' in hypothesis_lower:
            result = self.test_randomness(center)
            summary = result.get('_summary', {})
            return {
                'hypothesis': hypothesis_statement,
                'test_type': 'randomness',
                'result': result,
                'supports': summary.get('pass_rate', 0.0) > 0.8
            }
        else:
            # General analysis
            return {
                'hypothesis': hypothesis_statement,
                'test_type': 'general',
                'result': {
                    'periodicity': self.check_periodicity(center),
                    'frequency': self.analyze_frequency(center),
                    'patterns': self.find_patterns(center, max_length=5)
                }
            }
    
    def get_suggestions(self, problem_number: Optional[int] = None) -> List[str]:
        """
        Get suggestions for next experiments based on history.
        
        Args:
            problem_number: Optional problem number (1, 2, or 3)
            
        Returns:
            List of suggested experiment descriptions
        """
        if self.history:
            return self.history.suggest_next_experiments(problem_number)
        return []
    
    def clear_cache(self):
        """Clear the computation cache."""
        self._cache.clear()


# Convenience functions for direct use
def simulate_rule30(steps: int) -> BitArray:
    """Simulate Rule 30 and return center column."""
    api = Rule30AgentAPI(enable_history=False)
    return api.simulate_rule30(steps)


def check_periodicity(sequence: BitArray) -> Dict:
    """Check if sequence is periodic."""
    api = Rule30AgentAPI(enable_history=False)
    return api.check_periodicity(sequence)

