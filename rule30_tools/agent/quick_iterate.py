"""
Quick iteration helper for rapid experimentation.

This module provides utilities that enable agents to:
- Test ideas in seconds, not hours
- Iterate on hypotheses quickly
- Get immediate feedback
- Build up knowledge incrementally
"""

from typing import Dict, List, Optional, Callable, Any
from rule30_tools.agent.api import Rule30AgentAPI
from rule30_tools.agent.history import ExperimentHistory
from rule30_tools.experimentation.hypothesis import Hypothesis, HypothesisGenerator
from rule30_tools.core.bit_array import BitArray


class QuickIterationHelper:
    """
    Helper class for rapid experimentation and iteration.
    
    Key features:
    - Quick hypothesis testing (seconds, not hours)
    - Incremental exploration
    - Automatic result caching
    - Smart suggestions for next steps
    - Learning from each iteration
    """
    
    def __init__(self, api: Optional[Rule30AgentAPI] = None):
        """
        Initialize quick iteration helper.
        
        Args:
            api: Optional Rule30AgentAPI instance (creates new if not provided)
        """
        self.api = api or Rule30AgentAPI(enable_history=True)
        self.history = self.api.history
        self.hypothesis_generator = HypothesisGenerator()
        
        # Quick test configurations
        self.quick_test_steps = 10000      # For quick tests
        self.medium_test_steps = 100000    # For medium tests
        self.thorough_test_steps = 1000000 # For thorough tests
    
    def quick_test(
        self,
        hypothesis: str,
        test_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Quickly test a hypothesis (seconds, not hours).
        
        This is the core rapid iteration method.
        
        Args:
            hypothesis: Natural language hypothesis
            test_type: Optional test type ('periodicity', 'frequency', 'randomness')
            
        Returns:
            Quick test results
        """
        # Use quick test steps for speed
        result = self.api.quick_test_hypothesis(hypothesis, self.quick_test_steps)
        
        result['test_steps'] = self.quick_test_steps
        result['duration_seconds'] = result.get('duration_seconds', 0.0)
        result['next_suggestions'] = self._suggest_next_quick_tests(result)
        
        return result
    
    def incremental_explore(
        self,
        problem_number: int,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Incrementally explore a problem, building up knowledge.
        
        Starts small and scales up based on findings.
        
        Args:
            problem_number: Problem number (1, 2, or 3)
            max_iterations: Maximum number of iterations
            
        Returns:
            Exploration results
        """
        results = {
            'problem_number': problem_number,
            'iterations': [],
            'findings': [],
            'recommendations': []
        }
        
        current_steps = self.quick_test_steps
        
        for iteration in range(max_iterations):
            iteration_result = {
                'iteration': iteration + 1,
                'steps': current_steps,
                'result': None
            }
            
            if problem_number == 1:
                result = self.api.problem1_check(current_steps)
                iteration_result['result'] = result
                if result.get('is_periodic'):
                    results['findings'].append(f"Found periodicity at {current_steps} steps!")
                    break
            
            elif problem_number == 2:
                result = self.api.problem2_check(current_steps)
                iteration_result['result'] = result
                deviation = result.get('deviation_from_0.5', 1.0)
                if deviation < 0.001:
                    results['findings'].append(f"Frequency very close to 0.5 at {current_steps} steps")
            
            elif problem_number == 3:
                result = self.api.problem3_check(current_steps)
                iteration_result['result'] = result
                pass_rate = result.get('pass_rate', 0.0)
                if pass_rate > 0.9:
                    results['findings'].append(f"High randomness test pass rate at {current_steps} steps")
            
            results['iterations'].append(iteration_result)
            
            # Scale up for next iteration
            current_steps *= 10
            if current_steps > 10**12:
                break
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def batch_test_hypotheses(
        self,
        hypotheses: List[str],
        parallel: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Test multiple hypotheses quickly.
        
        Args:
            hypotheses: List of hypothesis statements
            parallel: Whether to run in parallel (future enhancement)
            
        Returns:
            List of test results
        """
        results = []
        for hypothesis in hypotheses:
            result = self.quick_test(hypothesis)
            results.append(result)
        
        return results
    
    def explore_pattern_space(
        self,
        min_length: int = 2,
        max_length: int = 8,
        steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Quickly explore the pattern space.
        
        Args:
            min_length: Minimum pattern length
            max_length: Maximum pattern length
            steps: Number of steps to analyze
            
        Returns:
            Pattern exploration results
        """
        if steps is None:
            steps = self.medium_test_steps
        
        center = self.api.simulate_rule30(steps)
        patterns = self.api.find_patterns(center, min_length, max_length)
        
        # Analyze pattern distribution
        pattern_analysis = {
            'total_patterns': len(patterns),
            'top_patterns': patterns[:10],
            'pattern_lengths': {},
            'unusual_patterns': []
        }
        
        for pattern in patterns:
            length = pattern['length']
            pattern_analysis['pattern_lengths'][length] = \
                pattern_analysis['pattern_lengths'].get(length, 0) + 1
            
            # Find unusual patterns (much more frequent than expected)
            if pattern['frequency'] > pattern['expected'] * 2:
                pattern_analysis['unusual_patterns'].append(pattern)
        
        return {
            'steps_analyzed': steps,
            'patterns': pattern_analysis,
            'suggestions': self._suggest_from_patterns(patterns)
        }
    
    def compare_scales(
        self,
        problem_number: int,
        scales: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Compare results across different scales.
        
        Helps agents understand how properties change with scale.
        
        Args:
            problem_number: Problem number
            scales: List of step counts to test
            
        Returns:
            Comparison results
        """
        if scales is None:
            scales = [1000, 10000, 100000, 1000000]
        
        comparisons = []
        
        for scale in scales:
            if problem_number == 1:
                result = self.api.problem1_check(scale)
            elif problem_number == 2:
                result = self.api.problem2_check(scale)
            elif problem_number == 3:
                result = self.api.problem3_check(scale)
            else:
                continue
            
            comparisons.append({
                'scale': scale,
                'result': result
            })
        
        # Analyze trends
        trends = self._analyze_trends(comparisons, problem_number)
        
        return {
            'problem_number': problem_number,
            'comparisons': comparisons,
            'trends': trends,
            'recommendations': self._recommend_next_scale(comparisons, problem_number)
        }
    
    def rapid_prototype(
        self,
        idea: str,
        test_function: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Rapidly prototype and test an idea.
        
        Args:
            idea: Description of the idea
            test_function: Optional custom test function
            
        Returns:
            Prototype test results
        """
        if test_function:
            # Use custom test function
            result = test_function(self.api)
        else:
            # Auto-detect test type from idea
            result = self.quick_test(idea)
        
        return {
            'idea': idea,
            'result': result,
            'viable': self._assess_viability(result),
            'next_steps': self._suggest_prototype_next_steps(result)
        }
    
    def _suggest_next_quick_tests(self, result: Dict) -> List[str]:
        """Suggest next quick tests based on result."""
        suggestions = []
        
        test_type = result.get('test_type', 'general')
        
        if test_type == 'periodicity':
            if not result.get('result', {}).get('is_periodic'):
                suggestions.append("Test with larger sequence (100k steps)")
                suggestions.append("Check for near-periodic patterns")
            else:
                suggestions.append("Analyze the detected period")
        
        elif test_type == 'frequency':
            deviation = result.get('result', {}).get('deviation_from_0.5', 1.0)
            if deviation > 0.01:
                suggestions.append("Test convergence rate")
                suggestions.append("Check at larger scales")
            else:
                suggestions.append("Verify with statistical tests")
        
        elif test_type == 'randomness':
            pass_rate = result.get('result', {}).get('pass_rate', 0.0)
            if pass_rate < 0.8:
                suggestions.append("Identify which tests failed")
                suggestions.append("Analyze failure patterns")
            else:
                suggestions.append("Test at larger scales")
        
        return suggestions
    
    def _generate_recommendations(self, exploration_results: Dict) -> List[str]:
        """Generate recommendations from exploration."""
        recommendations = []
        
        iterations = exploration_results.get('iterations', [])
        if not iterations:
            return recommendations
        
        # Check if we should scale up
        last_result = iterations[-1].get('result', {})
        if 'deviation_from_0.5' in last_result:
            deviation = last_result['deviation_from_0.5']
            if deviation > 0.01:
                recommendations.append("Continue scaling up - deviation still significant")
        
        # Check for interesting findings
        findings = exploration_results.get('findings', [])
        if findings:
            recommendations.append("Investigate findings more deeply")
        
        return recommendations
    
    def _suggest_from_patterns(self, patterns: List[Dict]) -> List[str]:
        """Suggest next steps based on patterns found."""
        suggestions = []
        
        if not patterns:
            suggestions.append("No patterns found - test at larger scale")
            return suggestions
        
        # Check for unusual patterns
        unusual = [p for p in patterns if p['frequency'] > p['expected'] * 2]
        if unusual:
            suggestions.append(f"Found {len(unusual)} unusual patterns - investigate further")
        
        # Check pattern distribution
        if len(patterns) > 50:
            suggestions.append("Many patterns found - analyze pattern space structure")
        
        return suggestions
    
    def _analyze_trends(
        self,
        comparisons: List[Dict],
        problem_number: int
    ) -> Dict[str, Any]:
        """Analyze trends across scales."""
        trends = {
            'stable': [],
            'changing': [],
            'converging': []
        }
        
        if problem_number == 2:
            # Frequency convergence
            deviations = [
                c['result'].get('deviation_from_0.5', 1.0)
                for c in comparisons
            ]
            if len(deviations) > 1 and deviations[-1] < deviations[0]:
                trends['converging'].append('Frequency appears to be converging')
        
        return trends
    
    def _recommend_next_scale(
        self,
        comparisons: List[Dict],
        problem_number: int
    ) -> List[str]:
        """Recommend next scale to test."""
        if not comparisons:
            return ["Start with 10k steps"]
        
        max_scale = max(c['scale'] for c in comparisons)
        return [f"Test at {max_scale * 10} steps to continue scaling"]
    
    def _assess_viability(self, result: Dict) -> bool:
        """Assess if an idea is viable based on quick test."""
        # Simple viability assessment
        if 'supports' in result:
            return result['supports']
        if 'pass_rate' in result.get('result', {}):
            return result['result']['pass_rate'] > 0.7
        return True  # Default to viable if unclear
    
    def _suggest_prototype_next_steps(self, result: Dict) -> List[str]:
        """Suggest next steps for a prototype."""
        if self._assess_viability(result):
            return [
                "Prototype looks promising - scale up test",
                "Refine the idea based on results",
                "Test variations of the approach"
            ]
        else:
            return [
                "Prototype needs refinement",
                "Try a different approach",
                "Check if assumptions are correct"
            ]

