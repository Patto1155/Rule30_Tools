"""
Experiment history and learning system.

Tracks what's been tried, learns from failures, suggests next steps.
This enables agents to:
- Avoid repeating failed experiments
- Learn from what worked
- Get intelligent suggestions for next steps
- Build on previous discoveries
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import os
from collections import defaultdict

from rule30_tools.experimentation.hypothesis import Hypothesis, HypothesisType


@dataclass
class ExperimentRecord:
    """Record of a single experiment."""
    timestamp: datetime
    problem_number: Optional[int]
    experiment_type: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class ExperimentHistory:
    """
    Track experiments and learn from them.
    
    Key features for rapid iteration:
    - Fast lookup of similar experiments
    - Automatic suggestion of next steps
    - Learning from patterns in failures
    - Resource usage tracking
    """
    
    def __init__(self, history_file: Optional[str] = None):
        """
        Initialize experiment history.
        
        Args:
            history_file: Optional file to persist history
        """
        self.history_file = history_file or "rule30_experiments.json"
        self.experiments: List[ExperimentRecord] = []
        self._load_history()
        
        # Indexes for fast lookup
        self._by_problem: Dict[int, List[ExperimentRecord]] = defaultdict(list)
        self._by_type: Dict[str, List[ExperimentRecord]] = defaultdict(list)
        self._successful: List[ExperimentRecord] = []
        self._failed: List[ExperimentRecord] = []
        
        self._rebuild_indexes()
    
    def log_simulation(self, steps: int, sequence_length: int):
        """Log a simulation run."""
        record = ExperimentRecord(
            timestamp=datetime.now(),
            problem_number=None,
            experiment_type='simulation',
            parameters={'steps': steps, 'sequence_length': sequence_length},
            result={'status': 'completed'},
            success=True
        )
        self._add_record(record)
    
    def log_analysis(self, analysis_type: str, result: Dict):
        """Log an analysis result."""
        record = ExperimentRecord(
            timestamp=datetime.now(),
            problem_number=None,
            experiment_type=f'analysis_{analysis_type}',
            parameters={},
            result=result,
            success=True
        )
        self._add_record(record)
    
    def log_problem_check(self, problem_number: int, steps: int, result: Dict):
        """Log a problem check."""
        # Determine success based on problem
        if problem_number == 1:
            success = not result.get('is_periodic', True)  # Success if non-periodic
        elif problem_number == 2:
            success = result.get('converging', False) and result.get('deviation_from_0.5', 1.0) < 0.01
        elif problem_number == 3:
            success = result.get('pass_rate', 0.0) > 0.8
        else:
            success = True
        
        record = ExperimentRecord(
            timestamp=datetime.now(),
            problem_number=problem_number,
            experiment_type=f'problem_{problem_number}_check',
            parameters={'steps': steps},
            result=result,
            success=success
        )
        self._add_record(record)
    
    def log_experiment(
        self,
        experiment_type: str,
        parameters: Dict,
        result: Dict,
        success: bool,
        problem_number: Optional[int] = None
    ):
        """Log a general experiment."""
        record = ExperimentRecord(
            timestamp=datetime.now(),
            problem_number=problem_number,
            experiment_type=experiment_type,
            parameters=parameters,
            result=result,
            success=success
        )
        self._add_record(record)
    
    def get_similar_experiments(
        self,
        experiment_type: str,
        parameters: Dict,
        limit: int = 5
    ) -> List[ExperimentRecord]:
        """
        Find similar experiments that have been tried before.
        
        This helps agents avoid repeating work.
        
        Args:
            experiment_type: Type of experiment
            parameters: Experiment parameters
            limit: Maximum number of results
            
        Returns:
            List of similar experiments
        """
        similar = []
        
        # Find experiments of same type
        candidates = self._by_type.get(experiment_type, [])
        
        # Score by parameter similarity
        for exp in candidates:
            score = self._similarity_score(parameters, exp.parameters)
            if score > 0.5:  # Threshold for similarity
                similar.append((score, exp))
        
        # Sort by similarity and return top results
        similar.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in similar[:limit]]
    
    def get_failed_approaches(self, problem_number: int) -> List[Dict]:
        """
        Get approaches that failed for a problem.
        
        Helps agents avoid repeating failures.
        
        Args:
            problem_number: Problem number (1, 2, or 3)
            
        Returns:
            List of failed approach summaries
        """
        failed = [
            exp for exp in self._by_problem.get(problem_number, [])
            if not exp.success
        ]
        
        # Group by experiment type
        by_type = defaultdict(list)
        for exp in failed:
            by_type[exp.experiment_type].append(exp)
        
        # Summarize
        summaries = []
        for exp_type, exps in by_type.items():
            summaries.append({
                'type': exp_type,
                'count': len(exps),
                'last_tried': max(exp.timestamp for exp in exps).isoformat(),
                'common_parameters': self._common_parameters(exps)
            })
        
        return summaries
    
    def suggest_next_experiments(
        self,
        problem_number: Optional[int] = None,
        limit: int = 5
    ) -> List[str]:
        """
        Suggest next experiments to try.
        
        This is a key feature for rapid iteration - agents get intelligent
        suggestions based on what's been tried.
        
        Args:
            problem_number: Optional problem number to focus on
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested experiment descriptions
        """
        suggestions = []
        
        if problem_number == 1:
            # Problem 1: Periodicity
            max_steps_tested = self._max_steps_tested(problem_number)
            if max_steps_tested < 1000000:
                suggestions.append(f"Test periodicity with {max_steps_tested * 10} steps")
            suggestions.append("Check for near-periodic patterns with period < 1000")
            suggestions.append("Analyze autocorrelation at various lags")
        
        elif problem_number == 2:
            # Problem 2: Frequency
            max_steps_tested = self._max_steps_tested(problem_number)
            if max_steps_tested < 1000000:
                suggestions.append(f"Test frequency convergence with {max_steps_tested * 10} steps")
            suggestions.append("Analyze convergence rate (O(1/sqrt(n)) vs O(1/n))")
            suggestions.append("Test chi-squared at multiple scales")
        
        elif problem_number == 3:
            # Problem 3: Randomness
            max_steps_tested = self._max_steps_tested(problem_number)
            if max_steps_tested < 1000000:
                suggestions.append(f"Run full randomness test suite with {max_steps_tested * 10} steps")
            suggestions.append("Compare to true random sequences")
            suggestions.append("Test at multiple sequence lengths")
        
        else:
            # General suggestions
            suggestions.append("Test all three problems with increasing step counts")
            suggestions.append("Look for patterns in the center column")
            suggestions.append("Analyze structure and entropy")
        
        # Add suggestions based on failures
        if problem_number:
            failed = self.get_failed_approaches(problem_number)
            for failure in failed[:2]:  # Top 2 failures
                suggestions.append(f"Avoid: {failure['type']} (failed {failure['count']} times)")
        
        return suggestions[:limit]
    
    def generate_insights(self) -> List[str]:
        """
        Generate insights from experiment history.
        
        Returns:
            List of insights
        """
        insights = []
        
        # Success rate by problem
        for problem_num in [1, 2, 3]:
            exps = self._by_problem.get(problem_num, [])
            if exps:
                success_rate = sum(1 for e in exps if e.success) / len(exps)
                insights.append(
                    f"Problem {problem_num}: {len(exps)} experiments, "
                    f"{success_rate:.1%} success rate"
                )
        
        # Most tested step counts
        step_counts = defaultdict(int)
        for exp in self.experiments:
            if 'steps' in exp.parameters:
                step_counts[exp.parameters['steps']] += 1
        
        if step_counts:
            most_common = max(step_counts.items(), key=lambda x: x[1])
            insights.append(f"Most tested step count: {most_common[0]:,} ({most_common[1]} times)")
        
        # Time trends
        if len(self.experiments) > 1:
            recent = [e for e in self.experiments if e.success][-10:]
            if recent:
                recent_success_rate = sum(1 for e in recent if e.success) / len(recent)
                insights.append(f"Recent success rate: {recent_success_rate:.1%}")
        
        return insights
    
    def _add_record(self, record: ExperimentRecord):
        """Add a record and update indexes."""
        self.experiments.append(record)
        
        if record.problem_number:
            self._by_problem[record.problem_number].append(record)
        
        self._by_type[record.experiment_type].append(record)
        
        if record.success:
            self._successful.append(record)
        else:
            self._failed.append(record)
        
        self._save_history()
    
    def _rebuild_indexes(self):
        """Rebuild all indexes."""
        self._by_problem.clear()
        self._by_type.clear()
        self._successful.clear()
        self._failed.clear()
        
        for exp in self.experiments:
            if exp.problem_number:
                self._by_problem[exp.problem_number].append(exp)
            self._by_type[exp.experiment_type].append(exp)
            if exp.success:
                self._successful.append(exp)
            else:
                self._failed.append(exp)
    
    def _similarity_score(self, params1: Dict, params2: Dict) -> float:
        """Compute similarity score between parameter sets."""
        if not params1 or not params2:
            return 0.0
        
        common_keys = set(params1.keys()) & set(params2.keys())
        if not common_keys:
            return 0.0
        
        matches = sum(
            1 for k in common_keys
            if params1[k] == params2[k]
        )
        
        return matches / max(len(params1), len(params2))
    
    def _common_parameters(self, experiments: List[ExperimentRecord]) -> Dict:
        """Find common parameters across experiments."""
        if not experiments:
            return {}
        
        common = {}
        first_params = experiments[0].parameters
        
        for key in first_params:
            if all(key in exp.parameters and exp.parameters[key] == first_params[key]
                   for exp in experiments):
                common[key] = first_params[key]
        
        return common
    
    def _max_steps_tested(self, problem_number: int) -> int:
        """Get maximum steps tested for a problem."""
        exps = self._by_problem.get(problem_number, [])
        max_steps = 0
        for exp in exps:
            if 'steps' in exp.parameters:
                max_steps = max(max_steps, exp.parameters['steps'])
        return max_steps
    
    def _save_history(self):
        """Save history to file."""
        if self.history_file:
            try:
                with open(self.history_file, 'w') as f:
                    json.dump(
                        [exp.to_dict() for exp in self.experiments],
                        f,
                        indent=2
                    )
            except Exception:
                pass  # Don't fail if can't save
    
    def _load_history(self):
        """Load history from file."""
        if self.history_file and os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        record = ExperimentRecord(
                            timestamp=datetime.fromisoformat(item['timestamp']),
                            problem_number=item.get('problem_number'),
                            experiment_type=item['experiment_type'],
                            parameters=item['parameters'],
                            result=item['result'],
                            success=item['success'],
                            duration_seconds=item.get('duration_seconds', 0.0)
                        )
                        self.experiments.append(record)
            except Exception:
                pass  # Start fresh if can't load

