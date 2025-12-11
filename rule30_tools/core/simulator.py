"""High-performance Rule 30 cellular automaton simulator."""

from typing import Generator, Optional
from rule30_tools.core.bit_array import BitArray
import numpy as np


class Rule30Simulator:
    """High-performance Rule 30 cellular automaton simulator."""
    
    def __init__(self, initial_state: Optional[BitArray] = None):
        """
        Initialize simulator.
        
        Args:
            initial_state: Optional initial state. If None, starts with single 1 in center.
        """
        if initial_state is None:
            # Start with single 1 in center
            self.state = BitArray(1)
            self.state[0] = True
            self.center_offset = 0
        else:
            self.state = initial_state
            self.center_offset = len(initial_state) // 2
        self.step_count = 0
    
    def compute_step(self, state: BitArray) -> BitArray:
        """
        Compute one step of Rule 30 evolution.
        
        Rule 30: new cell = left XOR (center OR right)
        Truth table: 111→0, 110→0, 101→0, 100→1, 011→1, 010→1, 001→1, 000→0
        
        Args:
            state: Current state
            
        Returns:
            New state after one step (expanded by 1 cell on each side)
        """
        new_size = len(state) + 2  # Expand by 1 on each side
        new_state = BitArray(new_size)
        
        for i in range(new_size):
            # Map to original state indices
            orig_idx = i - 1  # Shift by 1 to account for expansion
            
            left = state[orig_idx - 1] if 0 <= orig_idx - 1 < len(state) else False
            center = state[orig_idx] if 0 <= orig_idx < len(state) else False
            right = state[orig_idx + 1] if 0 <= orig_idx + 1 < len(state) else False
            
            # Rule 30: new = left XOR (center OR right)
            new_state[i] = left ^ (center or right)
        
        return new_state
    
    def simulate(self, n_steps: int) -> Generator[BitArray, None, None]:
        """
        Generate state after each step.
        
        Args:
            n_steps: Number of steps to simulate
            
        Yields:
            State after each step (including initial state)
        """
        current_state = self.state
        yield current_state
        
        for _ in range(n_steps):
            current_state = self.compute_step(current_state)
            self.step_count += 1
            yield current_state
    
    def compute_center_column(self, n_steps: int) -> BitArray:
        """
        Compute only the center column efficiently.
        
        This is optimized to track only the center cell as the pattern expands.
        
        Args:
            n_steps: Number of steps to compute
            
        Returns:
            BitArray representing the center column sequence
        """
        center_column = BitArray(n_steps + 1)
        center_column[0] = True  # Initial state
        
        # Use a sliding window approach for efficiency
        current_state = self.state
        center_idx = self.center_offset
        
        for step in range(1, n_steps + 1):
            # Expand state
            new_state = self.compute_step(current_state)
            
            # Track center position
            # As we expand, the center moves: original center at index 0
            # After expansion, center is at index 1 (since we add 1 cell on left)
            center_idx = center_idx + 1
            center_column[step] = new_state[center_idx]
            current_state = new_state
        
        return center_column
    
    def get_state(self) -> BitArray:
        """Get current state."""
        return self.state.copy()
    
    def reset(self, initial_state: Optional[BitArray] = None):
        """Reset simulator to initial state."""
        if initial_state is None:
            self.state = BitArray(1)
            self.state[0] = True
            self.center_offset = 0
        else:
            self.state = initial_state
            self.center_offset = len(initial_state) // 2
        self.step_count = 0

