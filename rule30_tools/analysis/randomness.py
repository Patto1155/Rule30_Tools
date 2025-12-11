"""Randomness testing suite for Problem 3 (NIST SP 800-22 tests)."""

from typing import Dict, List, Optional
from rule30_tools.core.bit_array import BitArray
import numpy as np
import math


class RandomnessTestSuite:
    """Comprehensive statistical tests for randomness."""
    
    def run_all_tests(self, sequence: BitArray, tests: Optional[List[str]] = None) -> Dict:
        """
        Run all randomness tests.
        
        Args:
            sequence: BitArray to test
            tests: Optional list of test names to run (default: all)
            
        Returns:
            Dictionary with all test results
        """
        all_test_names = [
            'frequency_test',
            'block_frequency_test',
            'runs_test',
            'longest_run_test',
            'binary_matrix_rank_test',
            'dft_test',
            'serial_test',
            'approximate_entropy_test',
            'cumulative_sums_test',
        ]
        
        if tests is None:
            tests = all_test_names
        
        results = {}
        for test_name in tests:
            if test_name in all_test_names:
                try:
                    results[test_name] = self.run_single_test(test_name, sequence)
                except Exception as e:
                    results[test_name] = {
                        'passed': False,
                        'error': str(e),
                        'p_value': None
                    }
        
        # Summary
        passed = sum(1 for r in results.values() if r.get('passed', False))
        total = len(results)
        
        results['_summary'] = {
            'tests_passed': passed,
            'tests_total': total,
            'pass_rate': passed / total if total > 0 else 0.0
        }
        
        return results
    
    def run_single_test(self, test_name: str, sequence: BitArray) -> Dict:
        """
        Run a single randomness test.
        
        Args:
            test_name: Name of test to run
            sequence: BitArray to test
            
        Returns:
            Dictionary with test results
        """
        test_methods = {
            'frequency_test': self.frequency_test,
            'block_frequency_test': self.block_frequency_test,
            'runs_test': self.runs_test,
            'longest_run_test': self.longest_run_test,
            'binary_matrix_rank_test': self.binary_matrix_rank_test,
            'dft_test': self.dft_test,
            'serial_test': self.serial_test,
            'approximate_entropy_test': self.approximate_entropy_test,
            'cumulative_sums_test': self.cumulative_sums_test,
        }
        
        if test_name not in test_methods:
            raise ValueError(f"Unknown test: {test_name}")
        
        return test_methods[test_name](sequence)
    
    def frequency_test(self, sequence: BitArray) -> Dict:
        """Monobit frequency test (NIST SP 800-22)."""
        n = len(sequence)
        if n < 100:
            return {'passed': False, 'p_value': None, 'error': 'Sequence too short'}
        
        ones = sequence.count_ones_simple()
        s = 2 * ones - n  # Sum of (2*bit - 1)
        s_obs = abs(s) / np.sqrt(n)
        p_value = math.erfc(s_obs / np.sqrt(2))
        
        return {
            'passed': p_value >= 0.01,
            'p_value': p_value,
            'statistic': s_obs,
            'ones_count': ones,
            'zeros_count': n - ones
        }
    
    def block_frequency_test(self, sequence: BitArray, block_size: int = 128) -> Dict:
        """Block frequency test (NIST SP 800-22)."""
        n = len(sequence)
        if n < block_size:
            return {'passed': False, 'p_value': None, 'error': 'Sequence too short'}
        
        num_blocks = n // block_size
        chi_squared = 0.0
        
        for i in range(num_blocks):
            block = sequence.slice(i * block_size, (i + 1) * block_size)
            ones = block.count_ones_simple()
            pi = ones / block_size
            chi_squared += (pi - 0.5) ** 2
        
        chi_squared = 4 * block_size * chi_squared
        p_value = 1 - self._chi2_cdf(chi_squared, num_blocks)
        
        return {
            'passed': p_value >= 0.01,
            'p_value': p_value,
            'statistic': chi_squared,
            'num_blocks': num_blocks
        }
    
    def runs_test(self, sequence: BitArray) -> Dict:
        """Runs test (NIST SP 800-22)."""
        n = len(sequence)
        if n < 100:
            return {'passed': False, 'p_value': None, 'error': 'Sequence too short'}
        
        ones = sequence.count_ones_simple()
        pi = ones / n
        
        if abs(pi - 0.5) >= 2 / np.sqrt(n):
            return {'passed': False, 'p_value': None, 'error': 'Frequency test prerequisite failed'}
        
        # Count runs
        runs = 1
        for i in range(1, n):
            if sequence[i] != sequence[i-1]:
                runs += 1
        
        # Expected runs
        expected_runs = 2 * n * pi * (1 - pi)
        variance = 2 * n * pi * (1 - pi) * (2 * n * pi * (1 - pi) - n) / (n - 1)
        
        if variance == 0:
            return {'passed': False, 'p_value': None, 'error': 'Zero variance'}
        
        z = (runs - expected_runs) / np.sqrt(variance)
        p_value = math.erfc(abs(z) / np.sqrt(2))
        
        return {
            'passed': p_value >= 0.01,
            'p_value': p_value,
            'statistic': z,
            'runs': runs,
            'expected_runs': expected_runs
        }
    
    def longest_run_test(self, sequence: BitArray) -> Dict:
        """Longest run of ones test (NIST SP 800-22)."""
        n = len(sequence)
        if n < 128:
            return {'passed': False, 'p_value': None, 'error': 'Sequence too short'}
        
        # Divide into blocks
        if n >= 6272:
            block_size = 128
            num_blocks = n // block_size
        elif n >= 750:
            block_size = 8
            num_blocks = n // block_size
        else:
            return {'passed': False, 'p_value': None, 'error': 'Sequence too short'}
        
        # Count longest runs in each block
        longest_runs = []
        for i in range(num_blocks):
            block = sequence.slice(i * block_size, (i + 1) * block_size)
            max_run = 0
            current_run = 0
            for bit in block:
                if bit:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            longest_runs.append(max_run)
        
        # Expected frequencies (simplified)
        # For block_size=128: K=5, v0=0-4, v1=5, v2=6, v3=7+
        # For block_size=8: K=3, v0=0-1, v1=2, v2=3, v3=4+
        if block_size == 128:
            K = 5
            pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        else:  # block_size == 8
            K = 3
            pi = [0.2148, 0.3672, 0.2305, 0.1875]
        
        # Count frequencies
        freq = [0] * (K + 2)
        for run in longest_runs:
            if run <= K:
                freq[run] += 1
            else:
                freq[K + 1] += 1
        
        # Chi-squared test
        chi_squared = 0.0
        for i in range(K + 2):
            expected = num_blocks * pi[i]
            if expected > 0:
                chi_squared += (freq[i] - expected) ** 2 / expected
        
        p_value = 1 - self._chi2_cdf(chi_squared, K + 1)
        
        return {
            'passed': p_value >= 0.01,
            'p_value': p_value,
            'statistic': chi_squared,
            'frequencies': freq
        }
    
    def binary_matrix_rank_test(self, sequence: BitArray, matrix_size: int = 32) -> Dict:
        """Binary matrix rank test (NIST SP 800-22)."""
        n = len(sequence)
        if n < matrix_size * matrix_size:
            return {'passed': False, 'p_value': None, 'error': 'Sequence too short'}
        
        num_matrices = n // (matrix_size * matrix_size)
        full_rank = 0
        full_rank_minus_one = 0
        
        for i in range(num_matrices):
            matrix = self._create_matrix(sequence, i * matrix_size * matrix_size, matrix_size)
            rank = self._matrix_rank(matrix)
            if rank == matrix_size:
                full_rank += 1
            elif rank == matrix_size - 1:
                full_rank_minus_one += 1
        
        # Expected frequencies (simplified)
        # For 32x32: P_full = 0.2888, P_full_minus_1 = 0.5776
        P_full = 0.2888
        P_full_minus_1 = 0.5776
        
        chi_squared = (
            (full_rank - num_matrices * P_full) ** 2 / (num_matrices * P_full) +
            (full_rank_minus_one - num_matrices * P_full_minus_1) ** 2 / (num_matrices * P_full_minus_1) +
            ((num_matrices - full_rank - full_rank_minus_one) - num_matrices * (1 - P_full - P_full_minus_1)) ** 2 /
            (num_matrices * (1 - P_full - P_full_minus_1))
        )
        
        p_value = 1 - self._chi2_cdf(chi_squared, 2)
        
        return {
            'passed': p_value >= 0.01,
            'p_value': p_value,
            'statistic': chi_squared,
            'full_rank_count': full_rank,
            'full_rank_minus_one_count': full_rank_minus_one
        }
    
    def dft_test(self, sequence: BitArray) -> Dict:
        """Discrete Fourier Transform test (NIST SP 800-22)."""
        n = len(sequence)
        if n < 1000:
            return {'passed': False, 'p_value': None, 'error': 'Sequence too short'}
        
        # Convert to -1, +1
        x = np.array([1 if sequence[i] else -1 for i in range(n)])
        
        # Compute DFT
        X = np.fft.fft(x)
        M = np.abs(X[:n//2])
        
        # Expected number of peaks below threshold
        threshold = np.sqrt(np.log(1 / 0.05) * n)
        N0 = 0.95 * n / 2
        N1 = sum(1 for m in M if m < threshold)
        
        d = (N1 - N0) / np.sqrt(0.95 * 0.05 * n / 4)
        p_value = math.erfc(abs(d) / np.sqrt(2))
        
        return {
            'passed': p_value >= 0.01,
            'p_value': p_value,
            'statistic': d,
            'peaks_below_threshold': N1,
            'expected': N0
        }
    
    def serial_test(self, sequence: BitArray, pattern_length: int = 2) -> Dict:
        """Serial test (NIST SP 800-22)."""
        n = len(sequence)
        if n < pattern_length * 2 ** pattern_length:
            return {'passed': False, 'p_value': None, 'error': 'Sequence too short'}
        
        # Count pattern frequencies
        pattern_counts = {}
        for i in range(n - pattern_length + 1):
            pattern = tuple(sequence[j] for j in range(i, i + pattern_length))
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Expected frequency
        expected = (n - pattern_length + 1) / (2 ** pattern_length)
        
        # Chi-squared
        chi_squared = sum(
            (count - expected) ** 2 / expected
            for count in pattern_counts.values()
        )
        
        p_value = 1 - self._chi2_cdf(chi_squared, 2 ** pattern_length - 1)
        
        return {
            'passed': p_value >= 0.01,
            'p_value': p_value,
            'statistic': chi_squared,
            'num_patterns': len(pattern_counts)
        }
    
    def approximate_entropy_test(self, sequence: BitArray, pattern_length: int = 2) -> Dict:
        """Approximate entropy test (NIST SP 800-22)."""
        n = len(sequence)
        if n < pattern_length * 10:
            return {'passed': False, 'p_value': None, 'error': 'Sequence too short'}
        
        def phi(m):
            # Count patterns of length m
            pattern_counts = {}
            for i in range(n - m + 1):
                pattern = tuple(sequence[j] for j in range(i, i + m))
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Compute phi
            phi_val = 0.0
            for count in pattern_counts.values():
                p = count / (n - m + 1)
                if p > 0:
                    phi_val += p * math.log(p)
            return phi_val
        
        apen = phi(pattern_length) - phi(pattern_length + 1)
        chi_squared = 2 * n * (math.log(2) - apen)
        
        p_value = 1 - self._chi2_cdf(chi_squared, 2 ** pattern_length)
        
        return {
            'passed': p_value >= 0.01,
            'p_value': p_value,
            'statistic': chi_squared,
            'approximate_entropy': apen
        }
    
    def cumulative_sums_test(self, sequence: BitArray) -> Dict:
        """Cumulative sums test (NIST SP 800-22)."""
        n = len(sequence)
        if n < 100:
            return {'passed': False, 'p_value': None, 'error': 'Sequence too short'}
        
        # Convert to -1, +1
        x = [1 if sequence[i] else -1 for i in range(n)]
        
        # Forward cumulative sum
        S = 0
        max_S = 0
        for val in x:
            S += val
            max_S = max(max_S, abs(S))
        
        # Backward cumulative sum
        S = 0
        max_S_backward = 0
        for val in reversed(x):
            S += val
            max_S_backward = max(max_S_backward, abs(S))
        
        z_forward = max_S
        z_backward = max_S_backward
        
        # P-value (simplified)
        p_value_forward = 1 - sum(
            (-1) ** k * (math.erfc((4 * k + 1) * z_forward / np.sqrt(n)) -
                         math.erfc((4 * k - 1) * z_forward / np.sqrt(n)))
            for k in range(1, 10)
        )
        
        p_value_backward = 1 - sum(
            (-1) ** k * (math.erfc((4 * k + 1) * z_backward / np.sqrt(n)) -
                         math.erfc((4 * k - 1) * z_backward / np.sqrt(n)))
            for k in range(1, 10)
        )
        
        p_value = min(p_value_forward, p_value_backward)
        
        return {
            'passed': p_value >= 0.01,
            'p_value': p_value,
            'statistic_forward': z_forward,
            'statistic_backward': z_backward
        }
    
    def _chi2_cdf(self, x: float, df: int) -> float:
        """Approximate chi-squared CDF."""
        # Simple approximation
        if x < 0:
            return 0.0
        if df == 1:
            z = np.sqrt(x)
            return math.erf(z / np.sqrt(2))
        # For df > 1, use approximation
        return 1 - math.exp(-x / 2) * (1 + x / 2)
    
    def _create_matrix(self, sequence: BitArray, start: int, size: int) -> np.ndarray:
        """Create binary matrix from sequence."""
        matrix = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                idx = start + i * size + j
                if idx < len(sequence):
                    matrix[i, j] = 1 if sequence[idx] else 0
        return matrix
    
    def _matrix_rank(self, matrix: np.ndarray) -> int:
        """Compute rank of binary matrix (simplified)."""
        # Gaussian elimination over GF(2)
        matrix = matrix.copy()
        rank = 0
        rows, cols = matrix.shape
        
        for col in range(cols):
            # Find pivot
            pivot_row = None
            for row in range(rank, rows):
                if matrix[row, col] == 1:
                    pivot_row = row
                    break
            
            if pivot_row is not None:
                # Swap rows
                if pivot_row != rank:
                    matrix[[rank, pivot_row]] = matrix[[pivot_row, rank]]
                
                # Eliminate
                for row in range(rows):
                    if row != rank and matrix[row, col] == 1:
                        matrix[row] = (matrix[row] + matrix[rank]) % 2
                
                rank += 1
        
        return rank

