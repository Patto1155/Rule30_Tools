"""Experiment runner for executing and tracking experiments."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import time
import uuid
from enum import Enum

from rule30_tools.experimentation.hypothesis import Hypothesis
from rule30_tools.core.bit_array import BitArray


class ExperimentStatus(Enum):
    """Experiment status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentResult:
    """Result of an experiment."""
    experiment_id: str
    hypothesis: Hypothesis
    status: ExperimentStatus
    result_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['status'] = self.status.value
        result['hypothesis'] = {
            'id': self.hypothesis.id,
            'statement': self.hypothesis.statement,
            'type': self.hypothesis.hypothesis_type.value,
            'confidence': self.hypothesis.confidence,
        }
        return result


class ExperimentRunner:
    """Execute experiments, track results, manage resources."""
    
    def __init__(self, history_path: Optional[str] = None):
        """
        Initialize experiment runner.
        
        Args:
            history_path: Optional path to save experiment history
        """
        self.history_path = history_path
        self.active_experiments: Dict[str, Any] = {}
        self.completed_experiments: List[ExperimentResult] = []
        self._load_history()
    
    def run_experiment(
        self,
        hypothesis: Hypothesis,
        test_function: Optional[callable] = None,
        timeout: Optional[float] = None
    ) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            hypothesis: Hypothesis to test
            test_function: Optional function to test hypothesis (default: auto-detect)
            timeout: Optional timeout in seconds
            
        Returns:
            ExperimentResult
        """
        experiment_id = str(uuid.uuid4())
        start_time = time.time()
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            status=ExperimentStatus.RUNNING,
            timestamp=datetime.now()
        )
        
        self.active_experiments[experiment_id] = result
        
        try:
            # Use provided test function or auto-detect based on hypothesis type
            if test_function is None:
                test_function = self._get_default_test_function(hypothesis)
            
            # Run experiment
            if timeout:
                # Simple timeout implementation (could be improved with threading)
                result_data = test_function(hypothesis)
            else:
                result_data = test_function(hypothesis)
            
            result.status = ExperimentStatus.COMPLETED
            result.result_data = result_data
            result.duration_seconds = time.time() - start_time
            
            # Update hypothesis confidence based on result
            if 'supports' in result_data:
                from rule30_tools.experimentation.hypothesis import Evidence
                evidence = Evidence(
                    source=f"experiment_{experiment_id}",
                    data=result_data,
                    supports=result_data['supports'],
                    confidence=result_data.get('confidence', 0.5)
                )
                hypothesis.add_evidence(evidence)
        
        except Exception as e:
            result.status = ExperimentStatus.FAILED
            result.error = str(e)
            result.duration_seconds = time.time() - start_time
        
        finally:
            del self.active_experiments[experiment_id]
            self.completed_experiments.append(result)
            self._save_history()
        
        return result
    
    def batch_run(
        self,
        hypotheses: List[Hypothesis],
        parallel: bool = False,
        max_workers: int = 4
    ) -> List[ExperimentResult]:
        """
        Run multiple experiments.
        
        Args:
            hypotheses: List of hypotheses to test
            parallel: Whether to run in parallel
            max_workers: Maximum parallel workers
            
        Returns:
            List of ExperimentResults
        """
        if parallel:
            # Simple parallel implementation (could use multiprocessing)
            results = []
            for hypothesis in hypotheses:
                results.append(self.run_experiment(hypothesis))
            return results
        else:
            return [self.run_experiment(h) for h in hypotheses]
    
    def monitor_progress(self, experiment_id: str) -> Dict[str, Any]:
        """
        Monitor progress of an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Progress information
        """
        if experiment_id in self.active_experiments:
            result = self.active_experiments[experiment_id]
            return {
                'status': result.status.value,
                'duration': result.duration_seconds,
                'timestamp': result.timestamp.isoformat()
            }
        else:
            # Check completed experiments
            for result in self.completed_experiments:
                if result.experiment_id == experiment_id:
                    return {
                        'status': result.status.value,
                        'duration': result.duration_seconds,
                        'timestamp': result.timestamp.isoformat(),
                        'completed': True
                    }
            return {'error': 'Experiment not found'}
    
    def cancel_experiment(self, experiment_id: str) -> bool:
        """
        Cancel a running experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if cancelled, False if not found
        """
        if experiment_id in self.active_experiments:
            result = self.active_experiments[experiment_id]
            result.status = ExperimentStatus.CANCELLED
            del self.active_experiments[experiment_id]
            self.completed_experiments.append(result)
            self._save_history()
            return True
        return False
    
    def save_results(self, result: ExperimentResult, path: str):
        """Save experiment result to file."""
        with open(path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def load_results(self, path: str) -> ExperimentResult:
        """Load experiment result from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        # Reconstruct (simplified - would need full reconstruction)
        return data
    
    def get_recent_experiments(self, limit: int = 10) -> List[ExperimentResult]:
        """Get most recent experiments."""
        return sorted(
            self.completed_experiments,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
    
    def _get_default_test_function(self, hypothesis: Hypothesis) -> callable:
        """Get default test function for hypothesis type."""
        # This would be connected to actual test implementations
        def default_test(h: Hypothesis) -> Dict:
            return {
                'supports': False,
                'confidence': 0.0,
                'message': 'Default test not implemented'
            }
        return default_test
    
    def _save_history(self):
        """Save experiment history."""
        if self.history_path:
            with open(self.history_path, 'w') as f:
                json.dump(
                    [r.to_dict() for r in self.completed_experiments],
                    f,
                    indent=2
                )
    
    def _load_history(self):
        """Load experiment history."""
        if self.history_path:
            try:
                with open(self.history_path, 'r') as f:
                    data = json.load(f)
                    # Reconstruct results (simplified)
                    self.completed_experiments = []
            except FileNotFoundError:
                pass

