# Rule 30 Tools: Implementation Guide

This document provides a practical implementation guide with code examples and project structure.

## Project Structure

```
rule30_tools/
├── core/
│   ├── __init__.py
│   ├── simulator.py          # Rule 30 simulation engine
│   ├── center_column.py      # Center column extraction
│   └── bit_array.py          # Efficient bit array implementation
├── analysis/
│   ├── __init__.py
│   ├── periodicity.py        # Periodicity detection
│   ├── frequency.py          # Frequency analysis
│   ├── randomness.py         # Randomness tests
│   └── patterns.py           # Pattern recognition
├── experimentation/
│   ├── __init__.py
│   ├── hypothesis.py         # Hypothesis generation
│   ├── experiments.py        # Experiment runner
│   └── counterexample.py     # Counterexample search
├── proof/
│   ├── __init__.py
│   ├── proof_generator.py    # Proof attempt generation
│   ├── formal_verify.py      # SMT/formal verification
│   └── invariants.py         # Invariant discovery
├── agent/
│   ├── __init__.py
│   ├── api.py                # High-level agent API
│   ├── history.py            # Experiment history
│   └── strategies.py         # Strategy generation
├── visualization/
│   ├── __init__.py
│   ├── plots.py              # Plotting functions
│   └── dashboard.py          # Interactive dashboards
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py         # Checkpoint management
│   ├── storage.py            # Efficient storage
│   └── parallel.py           # Parallel processing
├── tests/
│   ├── test_simulator.py
│   ├── test_analysis.py
│   └── ...
├── examples/
│   ├── problem1_periodicity.py
│   ├── problem2_frequency.py
│   └── problem3_randomness.py
├── docs/
│   ├── api_reference.md
│   └── tutorials.md
└── README.md
```

## Core Implementation Examples

### 1. Efficient Rule 30 Simulator

```python
# core/bit_array.py
from typing import List
import numpy as np

class BitArray:
    """Efficient bit array using numpy for storage."""
    
    def __init__(self, size: int = 0, data: np.ndarray = None):
        if data is not None:
            self.data = data
        else:
            self.data = np.zeros((size + 7) // 8, dtype=np.uint8)
        self.size = size if size else len(self.data) * 8
    
    def __getitem__(self, index: int) -> bool:
        byte_idx = index // 8
        bit_idx = index % 8
        return bool(self.data[byte_idx] & (1 << bit_idx))
    
    def __setitem__(self, index: int, value: bool):
        byte_idx = index // 8
        bit_idx = index % 8
        if value:
            self.data[byte_idx] |= (1 << bit_idx)
        else:
            self.data[byte_idx] &= ~(1 << bit_idx)
    
    def to_list(self) -> List[bool]:
        return [self[i] for i in range(self.size)]
    
    def count_ones(self) -> int:
        return sum(bin(byte).count('1') for byte in self.data)
```

```python
# core/simulator.py
from typing import Generator, Optional
from .bit_array import BitArray
import numpy as np

class Rule30Simulator:
    """High-performance Rule 30 cellular automaton simulator."""
    
    def __init__(self, initial_state: Optional[BitArray] = None):
        if initial_state is None:
            # Start with single 1 in center
            self.state = BitArray(1)
            self.state[0] = True
            self.center_offset = 0
        else:
            self.state = initial_state
            self.center_offset = len(initial_state) // 2
    
    def compute_step(self, state: BitArray) -> BitArray:
        """Compute one step of Rule 30 evolution."""
        # Rule 30: new cell = left XOR (center OR right)
        # Implemented efficiently with bit operations
        new_size = len(state) + 2  # Expand by 1 on each side
        new_state = BitArray(new_size)
        
        for i in range(new_size):
            left = state[i-1] if i > 0 else False
            center = state[i] if 0 <= i < len(state) else False
            right = state[i+1] if i < len(state) - 1 else False
            
            # Rule 30: 111→0, 110→0, 101→0, 100→1, 011→1, 010→1, 001→1, 000→0
            new_state[i] = left ^ (center or right)
        
        return new_state
    
    def simulate(self, n_steps: int) -> Generator[BitArray, None, None]:
        """Generate state after each step."""
        current_state = self.state
        yield current_state
        
        for _ in range(n_steps):
            current_state = self.compute_step(current_state)
            yield current_state
    
    def compute_center_column(self, n_steps: int) -> BitArray:
        """Compute only the center column efficiently."""
        center_column = BitArray(n_steps + 1)
        center_column[0] = True  # Initial state
        
        # Use a sliding window approach for efficiency
        current_state = self.state
        center_idx = self.center_offset
        
        for step in range(1, n_steps + 1):
            # Expand state
            new_state = self.compute_step(current_state)
            
            # Track center position (always at index 0 in our representation)
            center_idx = center_idx + 1  # Center moves as we expand
            center_column[step] = new_state[center_idx]
            current_state = new_state
        
        return center_column
```

### 2. Periodicity Detection

```python
# analysis/periodicity.py
from typing import Optional, List
from core.bit_array import BitArray

class PeriodicityDetector:
    """Detect periodic patterns in sequences."""
    
    def detect_period(self, sequence: BitArray, max_period: Optional[int] = None) -> Optional[int]:
        """Detect if sequence has a period, return period length or None."""
        if max_period is None:
            max_period = len(sequence) // 2
        
        for period in range(1, min(max_period + 1, len(sequence) // 2 + 1)):
            if self._has_period(sequence, period):
                return period
        return None
    
    def _has_period(self, sequence: BitArray, period: int) -> bool:
        """Check if sequence has the given period."""
        for i in range(period, len(sequence)):
            if sequence[i] != sequence[i % period]:
                return False
        return True
    
    def check_aperiodicity(self, sequence: BitArray) -> dict:
        """Comprehensive aperiodicity check."""
        max_checked = len(sequence)
        detected_period = self.detect_period(sequence)
        
        return {
            'is_periodic': detected_period is not None,
            'period': detected_period,
            'max_checked': max_checked,
            'confidence': self._compute_confidence(sequence, detected_period)
        }
    
    def _compute_confidence(self, sequence: BitArray, period: Optional[int]) -> float:
        """Compute confidence in periodicity result."""
        if period is None:
            # Check for near-periodic patterns
            # Use autocorrelation as confidence measure
            return self._autocorrelation_confidence(sequence)
        else:
            return 1.0  # Period found, high confidence
    
    def _autocorrelation_confidence(self, sequence: BitArray) -> float:
        """Compute autocorrelation-based confidence."""
        # Simplified autocorrelation at lag 1
        matches = sum(
            1 for i in range(len(sequence) - 1)
            if sequence[i] == sequence[i + 1]
        )
        return 1.0 - (matches / (len(sequence) - 1))
```

### 3. Frequency Analysis

```python
# analysis/frequency.py
from typing import Dict, List
from core.bit_array import BitArray
from scipy import stats
import numpy as np

class FrequencyAnalyzer:
    """Analyze frequency distribution of bits."""
    
    def compute_frequencies(self, sequence: BitArray) -> Dict:
        """Compute frequency statistics."""
        ones = sequence.count_ones()
        zeros = len(sequence) - ones
        
        return {
            'total': len(sequence),
            'ones': ones,
            'zeros': zeros,
            'ratio_ones': ones / len(sequence) if len(sequence) > 0 else 0.0,
            'ratio_zeros': zeros / len(sequence) if len(sequence) > 0 else 0.0,
            'deviation_from_0.5': abs(ones / len(sequence) - 0.5) if len(sequence) > 0 else 0.0
        }
    
    def compute_running_average(self, sequence: BitArray, window: int = 1000) -> List[float]:
        """Compute running average of ones ratio."""
        running_avg = []
        ones_count = 0
        
        for i in range(len(sequence)):
            if sequence[i]:
                ones_count += 1
            
            if i >= window:
                if sequence[i - window]:
                    ones_count -= 1
                window_start = i - window + 1
            else:
                window_start = 0
            
            current_window = i - window_start + 1
            running_avg.append(ones_count / current_window)
        
        return running_avg
    
    def test_convergence_to_0_5(self, sequence: BitArray) -> Dict:
        """Test if frequency converges to 0.5."""
        running_avg = self.compute_running_average(sequence)
        
        # Analyze convergence
        recent_avg = np.mean(running_avg[-len(running_avg)//10:])
        early_avg = np.mean(running_avg[:len(running_avg)//10])
        
        convergence_rate = abs(recent_avg - 0.5) - abs(early_avg - 0.5)
        
        return {
            'converging': convergence_rate < 0,
            'convergence_rate': convergence_rate,
            'current_deviation': abs(recent_avg - 0.5),
            'initial_deviation': abs(early_avg - 0.5),
            'running_average': running_avg
        }
    
    def chi_squared_test(self, sequence: BitArray) -> Dict:
        """Chi-squared test for uniform distribution."""
        ones = sequence.count_ones()
        zeros = len(sequence) - ones
        expected = len(sequence) / 2
        
        chi_squared = ((ones - expected) ** 2 + (zeros - expected) ** 2) / expected
        p_value = 1 - stats.chi2.cdf(chi_squared, df=1)
        
        return {
            'chi_squared': chi_squared,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'conclusion': 'uniform' if p_value >= 0.05 else 'non-uniform'
        }
```

### 4. Agent API

```python
# agent/api.py
"""High-level API for coding agents."""

from typing import Optional, Dict, List
from core.simulator import Rule30Simulator
from core.bit_array import BitArray
from analysis.periodicity import PeriodicityDetector
from analysis.frequency import FrequencyAnalyzer
from analysis.randomness import RandomnessTestSuite

class Rule30AgentAPI:
    """Simple, intuitive API for agents to use Rule 30 tools."""
    
    def __init__(self):
        self.simulator = Rule30Simulator()
        self.periodicity_detector = PeriodicityDetector()
        self.frequency_analyzer = FrequencyAnalyzer()
        self.randomness_suite = RandomnessTestSuite()
    
    def simulate_rule30(self, steps: int) -> BitArray:
        """Simulate Rule 30 for given number of steps, return center column."""
        return self.simulator.compute_center_column(steps)
    
    def check_periodicity(self, sequence: BitArray, max_period: Optional[int] = None) -> Dict:
        """Check if sequence is periodic."""
        return self.periodicity_detector.check_aperiodicity(sequence)
    
    def analyze_frequency(self, sequence: BitArray) -> Dict:
        """Analyze frequency distribution."""
        return self.frequency_analyzer.compute_frequencies(sequence)
    
    def test_convergence(self, sequence: BitArray) -> Dict:
        """Test if frequency converges to 0.5."""
        return self.frequency_analyzer.test_convergence_to_0_5(sequence)
    
    def test_randomness(self, sequence: BitArray, tests: Optional[List[str]] = None) -> Dict:
        """Run randomness tests on sequence."""
        if tests is None:
            return self.randomness_suite.run_all_tests(sequence)
        else:
            return {test: self.randomness_suite.run_single_test(test, sequence) 
                   for test in tests}
    
    def problem1_check(self, steps: int) -> Dict:
        """Check Problem 1: Does center column remain non-periodic?"""
        center = self.simulate_rule30(steps)
        periodicity = self.check_periodicity(center)
        return {
            'steps_checked': steps,
            'is_periodic': periodicity['is_periodic'],
            'period': periodicity['period'],
            'confidence': periodicity['confidence']
        }
    
    def problem2_check(self, steps: int) -> Dict:
        """Check Problem 2: Do colors occur equally often on average?"""
        center = self.simulate_rule30(steps)
        frequency = self.analyze_frequency(center)
        convergence = self.test_convergence(center)
        chi2 = self.frequency_analyzer.chi_squared_test(center)
        
        return {
            'steps_checked': steps,
            'ratio_ones': frequency['ratio_ones'],
            'deviation_from_0.5': frequency['deviation_from_0.5'],
            'converging': convergence['converging'],
            'chi_squared_result': chi2
        }
    
    def problem3_check(self, steps: int) -> Dict:
        """Check Problem 3: Does center column pass randomness tests?"""
        center = self.simulate_rule30(steps)
        randomness = self.test_randomness(center)
        
        passed = sum(1 for result in randomness.values() if result.get('passed', False))
        total = len(randomness)
        
        return {
            'steps_checked': steps,
            'tests_passed': passed,
            'tests_total': total,
            'test_results': randomness
        }

# Convenience functions for direct use
def simulate_rule30(steps: int) -> BitArray:
    """Simulate Rule 30 and return center column."""
    api = Rule30AgentAPI()
    return api.simulate_rule30(steps)

def check_periodicity(sequence: BitArray) -> Dict:
    """Check if sequence is periodic."""
    api = Rule30AgentAPI()
    return api.check_periodicity(sequence)
```

### 5. Example Usage

```python
# examples/problem1_periodicity.py
"""Example: Testing Problem 1 - Periodicity."""

from agent.api import Rule30AgentAPI

def test_periodicity_progressively():
    """Test periodicity with increasing sequence lengths."""
    api = Rule30AgentAPI()
    
    test_steps = [1000, 10000, 100000, 1000000, 10000000]
    
    for steps in test_steps:
        print(f"\nTesting {steps:,} steps...")
        result = api.problem1_check(steps)
        
        print(f"  Periodic: {result['is_periodic']}")
        if result['period']:
            print(f"  Period: {result['period']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        
        # If periodic, we found a counterexample!
        if result['is_periodic']:
            print(f"\n*** COUNTEREXAMPLE FOUND at {steps} steps! ***")
            break

if __name__ == "__main__":
    test_periodicity_progressively()
```

```python
# examples/problem2_frequency.py
"""Example: Testing Problem 2 - Frequency convergence."""

from agent.api import Rule30AgentAPI
import matplotlib.pyplot as plt

def analyze_frequency_convergence():
    """Analyze frequency convergence to 0.5."""
    api = Rule30AgentAPI()
    
    steps = 1000000
    center = api.simulate_rule30(steps)
    
    # Get running average
    convergence = api.test_convergence(center)
    running_avg = convergence['running_average']
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(running_avg, alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Expected: 0.5')
    plt.xlabel('Step')
    plt.ylabel('Running Average (Ratio of 1s)')
    plt.title('Frequency Convergence Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print statistics
    result = api.problem2_check(steps)
    print(f"Steps analyzed: {steps:,}")
    print(f"Final ratio of 1s: {result['ratio_ones']:.6f}")
    print(f"Deviation from 0.5: {result['deviation_from_0.5']:.6f}")
    print(f"Converging: {result['converging']}")
    print(f"Chi-squared p-value: {result['chi_squared_result']['p_value']:.6f}")

if __name__ == "__main__":
    analyze_frequency_convergence()
```

## Next Steps for Implementation

1. **Start with Phase 1**: Implement basic simulator and center column extractor
2. **Add tests**: Write comprehensive tests for each component
3. **Benchmark**: Measure performance and optimize bottlenecks
4. **Iterate**: Use tools on actual problems to refine design
5. **Document**: Write clear documentation and examples

## Integration with Coding Agents

Agents can use these tools by:
1. Importing the high-level API
2. Running experiments systematically
3. Analyzing results
4. Generating new hypotheses
5. Iterating based on findings

The modular design allows agents to:
- Use individual tools as needed
- Compose multiple tools for complex analyses
- Extend functionality through inheritance
- Learn from experiment history

