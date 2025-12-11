"""Frequency analysis tools for Problem 2."""

from typing import Dict, List, Optional
from rule30_tools.core.bit_array import BitArray
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class FrequencyAnalyzer:
    """Analyze frequency distribution of bits."""
    
    def compute_frequencies(self, sequence: BitArray, window_size: Optional[int] = None) -> Dict:
        """
        Compute frequency statistics.
        
        Args:
            sequence: BitArray to analyze
            window_size: Optional window size for windowed analysis
            
        Returns:
            Dictionary with frequency statistics
        """
        ones = sequence.count_ones_simple()
        zeros = len(sequence) - ones
        
        result = {
            'total': len(sequence),
            'ones': ones,
            'zeros': zeros,
            'ratio_ones': ones / len(sequence) if len(sequence) > 0 else 0.0,
            'ratio_zeros': zeros / len(sequence) if len(sequence) > 0 else 0.0,
            'deviation_from_0.5': abs(ones / len(sequence) - 0.5) if len(sequence) > 0 else 0.0
        }
        
        if window_size:
            result['windowed'] = self._compute_windowed_frequencies(sequence, window_size)
        
        return result
    
    def _compute_windowed_frequencies(self, sequence: BitArray, window_size: int) -> List[float]:
        """Compute frequencies in sliding windows."""
        frequencies = []
        ones_count = 0
        
        for i in range(len(sequence)):
            if sequence[i]:
                ones_count += 1
            
            if i >= window_size:
                if sequence[i - window_size]:
                    ones_count -= 1
                window_start = i - window_size + 1
            else:
                window_start = 0
            
            current_window = i - window_start + 1
            frequencies.append(ones_count / current_window if current_window > 0 else 0.0)
        
        return frequencies
    
    def compute_running_average(self, sequence: BitArray, window: int = 1000) -> List[float]:
        """
        Compute running average of ones ratio.
        
        Args:
            sequence: BitArray to analyze
            window: Window size for running average
            
        Returns:
            List of running average values
        """
        return self._compute_windowed_frequencies(sequence, window)
    
    def test_convergence_to_0_5(self, sequence: BitArray) -> Dict:
        """
        Test if frequency converges to 0.5.
        
        Args:
            sequence: BitArray to analyze
            
        Returns:
            Dictionary with convergence analysis
        """
        running_avg = self.compute_running_average(sequence)
        
        if len(running_avg) < 20:
            return {
                'converging': False,
                'convergence_rate': 0.0,
                'current_deviation': abs(running_avg[-1] - 0.5) if running_avg else 0.0,
                'initial_deviation': abs(running_avg[0] - 0.5) if running_avg else 0.0,
                'running_average': running_avg
            }
        
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
        """
        Chi-squared test for uniform distribution.
        
        Args:
            sequence: BitArray to test
            
        Returns:
            Dictionary with chi-squared test results
        """
        ones = sequence.count_ones_simple()
        zeros = len(sequence) - ones
        expected = len(sequence) / 2
        
        chi_squared = ((ones - expected) ** 2 + (zeros - expected) ** 2) / expected if expected > 0 else 0.0
        
        if HAS_SCIPY:
            p_value = 1 - stats.chi2.cdf(chi_squared, df=1)
        else:
            # Approximate p-value using normal approximation
            z = np.sqrt(chi_squared)
            p_value = 2 * (1 - 0.5 * (1 + np.sign(z) * (1 - np.exp(-2 * z**2 / np.pi))))
        
        return {
            'chi_squared': chi_squared,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'conclusion': 'uniform' if p_value >= 0.05 else 'non-uniform'
        }
    
    def compute_confidence_intervals(self, sequence: BitArray, confidence: float = 0.95) -> Dict:
        """
        Compute confidence intervals for the ratio.
        
        Args:
            sequence: BitArray to analyze
            confidence: Confidence level (default 0.95)
            
        Returns:
            Dictionary with confidence intervals
        """
        n = len(sequence)
        if n == 0:
            return {'lower': 0.0, 'upper': 1.0, 'confidence': confidence}
        
        p = sequence.count_ones_simple() / n
        
        # Normal approximation for binomial proportion
        z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
        margin = z * np.sqrt(p * (1 - p) / n)
        
        return {
            'lower': max(0.0, p - margin),
            'upper': min(1.0, p + margin),
            'confidence': confidence,
            'point_estimate': p
        }

