"""Pattern recognition and structure discovery tools."""

from typing import Dict, List, Tuple, Optional
from rule30_tools.core.bit_array import BitArray
import numpy as np
import math


class PatternRecognizer:
    """Discover hidden structures, symmetries, and invariants."""
    
    def find_patterns(self, sequence: BitArray, min_length: int = 2, max_length: int = 10) -> List[Dict]:
        """
        Find repeating patterns in sequence.
        
        Args:
            sequence: BitArray to analyze
            min_length: Minimum pattern length
            max_length: Maximum pattern length
            
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        
        for length in range(min_length, min(max_length + 1, len(sequence) // 2)):
            pattern_counts = {}
            
            # Count all patterns of this length
            for i in range(len(sequence) - length + 1):
                pattern = tuple(sequence[j] for j in range(i, i + length))
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Find patterns that appear more than expected
            expected = (len(sequence) - length + 1) / (2 ** length)
            threshold = expected * 1.5  # 50% more than expected
            
            for pattern, count in pattern_counts.items():
                if count > threshold:
                    patterns.append({
                        'pattern': pattern,
                        'length': length,
                        'count': count,
                        'expected': expected,
                        'frequency': count / (len(sequence) - length + 1)
                    })
        
        # Sort by frequency
        patterns.sort(key=lambda x: x['frequency'], reverse=True)
        return patterns
    
    def detect_structure(self, sequence: BitArray) -> Dict:
        """
        Detect overall structure in sequence.
        
        Args:
            sequence: BitArray to analyze
            
        Returns:
            Structure report
        """
        entropy = self.compute_entropy(sequence)
        compression_ratio = self.analyze_compression_ratio(sequence)
        autocorr = self._compute_autocorrelation(sequence)
        
        return {
            'entropy': entropy,
            'compression_ratio': compression_ratio,
            'autocorrelation': autocorr,
            'is_random_like': entropy > 0.9 and compression_ratio > 0.8,
            'has_structure': entropy < 0.7 or compression_ratio < 0.5
        }
    
    def compute_entropy(self, sequence: BitArray, block_size: int = 1) -> float:
        """
        Compute Shannon entropy.
        
        Args:
            sequence: BitArray to analyze
            block_size: Size of blocks to analyze
            
        Returns:
            Entropy value (0 to 1)
        """
        if block_size == 1:
            ones = sequence.count_ones_simple()
            zeros = len(sequence) - ones
            p1 = ones / len(sequence) if len(sequence) > 0 else 0.0
            p0 = zeros / len(sequence) if len(sequence) > 0 else 0.0
            
            entropy = 0.0
            if p1 > 0:
                entropy -= p1 * math.log2(p1)
            if p0 > 0:
                entropy -= p0 * math.log2(p0)
            
            return entropy
        else:
            # Block entropy
            pattern_counts = {}
            for i in range(len(sequence) - block_size + 1):
                pattern = tuple(sequence[j] for j in range(i, i + block_size))
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            total = len(sequence) - block_size + 1
            entropy = 0.0
            for count in pattern_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)
            
            return entropy / block_size  # Normalize
    
    def analyze_compression_ratio(self, sequence: BitArray) -> float:
        """
        Analyze compression ratio (estimate of Kolmogorov complexity).
        
        Args:
            sequence: BitArray to analyze
            
        Returns:
            Compression ratio (0 to 1, higher = more compressible)
        """
        # Simple run-length encoding
        compressed_size = 0
        current_bit = sequence[0]
        current_run = 1
        
        for i in range(1, len(sequence)):
            if sequence[i] == current_bit:
                current_run += 1
            else:
                # Encode run: need log2(run) bits for length + 1 bit for value
                compressed_size += math.ceil(math.log2(current_run + 1)) + 1
                current_bit = sequence[i]
                current_run = 1
        
        # Final run
        compressed_size += math.ceil(math.log2(current_run + 1)) + 1
        
        original_size = len(sequence)
        ratio = compressed_size / original_size if original_size > 0 else 1.0
        
        return ratio
    
    def _compute_autocorrelation(self, sequence: BitArray, max_lag: int = 100) -> Dict:
        """
        Compute autocorrelation at various lags.
        
        Args:
            sequence: BitArray to analyze
            max_lag: Maximum lag to compute
            
        Returns:
            Dictionary with autocorrelation results
        """
        max_lag = min(max_lag, len(sequence) // 10)
        autocorrs = []
        
        for lag in range(1, max_lag + 1):
            matches = sum(
                1 for i in range(len(sequence) - lag)
                if sequence[i] == sequence[i + lag]
            )
            autocorr = matches / (len(sequence) - lag) if len(sequence) > lag else 0.0
            autocorrs.append(autocorr)
        
        return {
            'lags': list(range(1, max_lag + 1)),
            'values': autocorrs,
            'mean': np.mean(autocorrs) if autocorrs else 0.5,
            'std': np.std(autocorrs) if autocorrs else 0.0
        }
    
    def generate_recurrence_plot(self, sequence: BitArray, threshold: int = 1000) -> np.ndarray:
        """
        Generate recurrence plot data.
        
        Args:
            sequence: BitArray to analyze
            threshold: Maximum size for plot (to avoid memory issues)
            
        Returns:
            2D numpy array representing recurrence plot
        """
        n = min(len(sequence), threshold)
        plot = np.zeros((n, n), dtype=bool)
        
        for i in range(n):
            for j in range(n):
                plot[i, j] = sequence[i] == sequence[j]
        
        return plot

