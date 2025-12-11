"""Efficient bit array implementation using numpy for storage."""

from typing import List, Optional
import numpy as np


class BitArray:
    """Efficient bit array using numpy for storage (8x memory reduction vs boolean arrays)."""
    
    def __init__(self, size: int = 0, data: Optional[np.ndarray] = None):
        """
        Initialize a BitArray.
        
        Args:
            size: Number of bits (if data is None)
            data: Optional numpy array of uint8 bytes (if provided, size is inferred)
        """
        if data is not None:
            self.data = data
            self.size = len(self.data) * 8
        else:
            if size < 0:
                raise ValueError("Size must be non-negative")
            self.data = np.zeros((size + 7) // 8, dtype=np.uint8)
            self.size = size
    
    def __len__(self) -> int:
        """Return the number of bits."""
        return self.size
    
    def __getitem__(self, index: int) -> bool:
        """Get bit at index."""
        if index < 0 or index >= self.size:
            raise IndexError(f"Index {index} out of range [0, {self.size})")
        byte_idx = index // 8
        bit_idx = index % 8
        return bool(self.data[byte_idx] & (1 << bit_idx))
    
    def __setitem__(self, index: int, value: bool):
        """Set bit at index."""
        if index < 0 or index >= self.size:
            raise IndexError(f"Index {index} out of range [0, {self.size})")
        byte_idx = index // 8
        bit_idx = index % 8
        if value:
            self.data[byte_idx] |= (1 << bit_idx)
        else:
            self.data[byte_idx] &= ~(1 << bit_idx)
    
    def __iter__(self):
        """Iterate over bits."""
        for i in range(self.size):
            yield self[i]
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BitArray(size={self.size}, ones={self.count_ones()})"
    
    def to_list(self) -> List[bool]:
        """Convert to list of booleans."""
        return [self[i] for i in range(self.size)]
    
    def to_string(self) -> str:
        """Convert to binary string representation."""
        return ''.join('1' if self[i] else '0' for i in range(self.size))
    
    def count_ones(self) -> int:
        """Count number of 1 bits efficiently."""
        return int(np.sum(np.unpackbits(self.data[:self.size // 8])))
        # Handle partial last byte
        if self.size % 8 != 0:
            last_byte = self.data[self.size // 8]
            last_bits = self.size % 8
            for i in range(last_bits):
                if last_byte & (1 << i):
                    return int(np.sum(np.unpackbits(self.data[:self.size // 8]))) + sum(
                        1 for i in range(last_bits) if last_byte & (1 << i)
                    )
        return int(np.sum(np.unpackbits(self.data[:self.size // 8])))
    
    def count_ones_simple(self) -> int:
        """Simple but slower count of ones."""
        return sum(1 for bit in self)
    
    def copy(self) -> 'BitArray':
        """Create a copy of this BitArray."""
        return BitArray(data=self.data.copy())
    
    def extend(self, additional_bits: int):
        """Extend the array by additional_bits (set to False)."""
        new_size = self.size + additional_bits
        new_data_size = (new_size + 7) // 8
        if new_data_size > len(self.data):
            new_data = np.zeros(new_data_size, dtype=np.uint8)
            new_data[:len(self.data)] = self.data
            self.data = new_data
        self.size = new_size
    
    def slice(self, start: int, end: int) -> 'BitArray':
        """Extract a slice of bits."""
        if start < 0:
            start = self.size + start
        if end < 0:
            end = self.size + end
        if start < 0 or end > self.size or start > end:
            raise IndexError(f"Invalid slice [{start}:{end}] for size {self.size}")
        
        result = BitArray(end - start)
        for i in range(start, end):
            result[i - start] = self[i]
        return result

