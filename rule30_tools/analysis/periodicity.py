"""Periodicity detection engine for Problem 1."""

from typing import Optional, Dict, List
from rule30_tools.core.bit_array import BitArray
import numpy as np


class PeriodicityDetector:
    """Detect periodic patterns in sequences."""
    
    def detect_period(self, sequence: BitArray, max_period: Optional[int] = None) -> Optional[int]:
        """
        Detect if sequence has a period, return period length or None.
        
        Args:
            sequence: BitArray to analyze
            max_period: Maximum period to check (default: half sequence length)
            
        Returns:
            Period length if found, None otherwise
        """
        if max_period is None:
            max_period = len(sequence) // 2
        
        # Optimize: only check divisors of sequence length
        for period in range(1, min(max_period + 1, len(sequence) // 2 + 1)):
            if self._has_period(sequence, period):
                return period
        return None
    
    def _has_period(self, sequence: BitArray, period: int) -> bool:
        """
        Check if sequence has the given period.
        
        Args:
            sequence: BitArray to check
            period: Period length to test
            
        Returns:
            True if sequence has this period
        """
        if period >= len(sequence):
            return False
        
        # Check if sequence repeats with this period
        for i in range(period, len(sequence)):
            if sequence[i] != sequence[i % period]:
                return False
        return True
    
    def find_all_periods(self, sequence: BitArray, max_period: Optional[int] = None) -> List[int]:
        """
        Find all periods of the sequence.
        
        Args:
            sequence: BitArray to analyze
            max_period: Maximum period to check
            
        Returns:
            List of all periods found
        """
        periods = []
        if max_period is None:
            max_period = len(sequence) // 2
        
        for period in range(1, min(max_period + 1, len(sequence) // 2 + 1)):
            if self._has_period(sequence, period):
                periods.append(period)
        return periods
    
    def check_aperiodicity(self, sequence: BitArray, n_steps: Optional[int] = None) -> Dict:
        """
        Comprehensive aperiodicity check.
        
        Args:
            sequence: BitArray to check
            n_steps: Optional number of steps checked (for reporting)
            
        Returns:
            Dictionary with periodicity analysis results
        """
        max_checked = len(sequence)
        detected_period = self.detect_period(sequence)
        
        return {
            'is_periodic': detected_period is not None,
            'period': detected_period,
            'max_checked': max_checked,
            'n_steps': n_steps if n_steps else max_checked,
            'confidence': self._compute_confidence(sequence, detected_period)
        }
    
    def _compute_confidence(self, sequence: BitArray, period: Optional[int]) -> float:
        """
        Compute confidence in periodicity result.
        
        Args:
            sequence: BitArray analyzed
            period: Detected period (None if aperiodic)
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if period is not None:
            return 1.0  # Period found, high confidence
        
        # Check for near-periodic patterns using autocorrelation
        return self._autocorrelation_confidence(sequence)
    
    def _autocorrelation_confidence(self, sequence: BitArray) -> float:
        """
        Compute autocorrelation-based confidence.
        
        Higher autocorrelation suggests periodicity, lower suggests randomness.
        
        Args:
            sequence: BitArray to analyze
            
        Returns:
            Confidence score
        """
        if len(sequence) < 2:
            return 0.5
        
        # Compute autocorrelation at various lags
        max_lag = min(100, len(sequence) // 10)
        autocorrs = []
        
        for lag in range(1, max_lag + 1):
            matches = sum(
                1 for i in range(len(sequence) - lag)
                if sequence[i] == sequence[i + lag]
            )
            autocorr = matches / (len(sequence) - lag)
            autocorrs.append(autocorr)
        
        # High autocorrelation suggests periodicity
        avg_autocorr = np.mean(autocorrs) if autocorrs else 0.5
        
        # Convert to confidence: if avg_autocorr is close to 0.5 (random), low confidence in periodicity
        # If avg_autocorr is far from 0.5, higher confidence
        confidence = 1.0 - abs(avg_autocorr - 0.5) * 2
        return max(0.0, min(1.0, confidence))
    
    def generate_periodicity_report(self, sequence: BitArray) -> Dict:
        """
        Generate comprehensive periodicity report.
        
        Args:
            sequence: BitArray to analyze
            
        Returns:
            Detailed periodicity report
        """
        result = self.check_aperiodicity(sequence)
        all_periods = self.find_all_periods(sequence)
        autocorr_conf = self._autocorrelation_confidence(sequence)
        
        return {
            **result,
            'all_periods': all_periods,
            'autocorrelation_confidence': autocorr_conf,
            'sequence_length': len(sequence),
        }

