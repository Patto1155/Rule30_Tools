"""Center column extraction and analysis utilities."""

from typing import Dict, Optional
from rule30_tools.core.bit_array import BitArray
from rule30_tools.core.simulator import Rule30Simulator
import json
import os
import numpy as np


class CenterColumnExtractor:
    """Extract and analyze center column sequences."""
    
    @staticmethod
    def extract_column(simulator: Rule30Simulator, n_steps: int) -> BitArray:
        """
        Extract center column from simulator.
        
        Args:
            simulator: Rule30Simulator instance
            n_steps: Number of steps to extract
            
        Returns:
            Center column sequence as BitArray
        """
        return simulator.compute_center_column(n_steps)
    
    @staticmethod
    def save_to_file(sequence: BitArray, filepath: str, format: str = 'binary'):
        """
        Save sequence to file.
        
        Args:
            sequence: BitArray to save
            filepath: Path to save file
            format: 'binary' (raw bytes) or 'text' (human-readable)
        """
        if format == 'binary':
            # Save as raw binary
            with open(filepath, 'wb') as f:
                f.write(sequence.data.tobytes())
            # Save metadata
            metadata_path = filepath + '.meta'
            with open(metadata_path, 'w') as f:
                json.dump({'size': sequence.size, 'format': 'binary'}, f)
        elif format == 'text':
            # Save as text (0s and 1s)
            with open(filepath, 'w') as f:
                f.write(sequence.to_string())
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def load_from_file(filepath: str, format: Optional[str] = None) -> BitArray:
        """
        Load sequence from file.
        
        Args:
            filepath: Path to file
            format: 'binary' or 'text'. If None, inferred from extension
            
        Returns:
            Loaded BitArray
        """
        if format is None:
            if filepath.endswith('.bin') or os.path.exists(filepath + '.meta'):
                format = 'binary'
            else:
                format = 'text'
        
        if format == 'binary':
            # Load binary
            with open(filepath, 'rb') as f:
                data_bytes = f.read()
            data = np.frombuffer(data_bytes, dtype=np.uint8)
            
            # Load metadata if available
            metadata_path = filepath + '.meta'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                size = metadata['size']
            else:
                size = len(data) * 8
            
            return BitArray(data=np.array(data))
        elif format == 'text':
            # Load text
            with open(filepath, 'r') as f:
                text = f.read().strip()
            sequence = BitArray(len(text))
            for i, char in enumerate(text):
                sequence[i] = char == '1'
            return sequence
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def compute_statistics(sequence: BitArray) -> Dict:
        """
        Compute basic statistics on sequence.
        
        Args:
            sequence: BitArray to analyze
            
        Returns:
            Dictionary with statistics
        """
        ones = sequence.count_ones_simple()
        zeros = len(sequence) - ones
        
        # Block frequencies (2-bit patterns)
        block2_freq = {'00': 0, '01': 0, '10': 0, '11': 0}
        for i in range(len(sequence) - 1):
            pattern = ('1' if sequence[i] else '0') + ('1' if sequence[i+1] else '0')
            block2_freq[pattern] = block2_freq.get(pattern, 0) + 1
        
        # Runs (consecutive 0s or 1s)
        runs_0 = []
        runs_1 = []
        current_run = 1
        current_bit = sequence[0]
        
        for i in range(1, len(sequence)):
            if sequence[i] == current_bit:
                current_run += 1
            else:
                if current_bit:
                    runs_1.append(current_run)
                else:
                    runs_0.append(current_run)
                current_run = 1
                current_bit = sequence[i]
        
        # Add final run
        if current_bit:
            runs_1.append(current_run)
        else:
            runs_0.append(current_run)
        
        return {
            'total_bits': len(sequence),
            'ones': ones,
            'zeros': zeros,
            'ratio_ones': ones / len(sequence) if len(sequence) > 0 else 0.0,
            'ratio_zeros': zeros / len(sequence) if len(sequence) > 0 else 0.0,
            'block2_frequencies': block2_freq,
            'runs_0': {
                'count': len(runs_0),
                'avg_length': sum(runs_0) / len(runs_0) if runs_0 else 0.0,
                'max_length': max(runs_0) if runs_0 else 0,
            },
            'runs_1': {
                'count': len(runs_1),
                'avg_length': sum(runs_1) / len(runs_1) if runs_1 else 0.0,
                'max_length': max(runs_1) if runs_1 else 0,
            },
        }

