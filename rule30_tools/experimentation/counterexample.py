"""Counterexample finder for actively searching for counterexamples."""

from typing import Optional, Dict, Callable, Any
from rule30_tools.core.bit_array import BitArray
from rule30_tools.core.simulator import Rule30Simulator
from rule30_tools.analysis.periodicity import PeriodicityDetector
from rule30_tools.analysis.frequency import FrequencyAnalyzer


class Counterexample:
    """A counterexample to a property."""
    def __init__(self, n: int, evidence: Dict[str, Any]):
        self.n = n
        self.evidence = evidence


class CounterexampleFinder:
    """Actively search for counterexamples to conjectures."""
    
    def __init__(self):
        self.simulator = Rule30Simulator()
        self.periodicity_detector = PeriodicityDetector()
        self.frequency_analyzer = FrequencyAnalyzer()
    
    def search_counterexample(
        self,
        property_name: str,
        max_n: int = 10**9,
        start_n: int = 1
    ) -> Optional[Counterexample]:
        """
        Search for a counterexample to a property.
        
        Args:
            property_name: Name of property to test ('periodicity', 'frequency', etc.)
            max_n: Maximum n to test
            start_n: Starting n value
            
        Returns:
            Counterexample if found, None otherwise
        """
        if property_name == 'periodicity':
            return self._search_periodicity_counterexample(max_n, start_n)
        elif property_name == 'frequency':
            return self._search_frequency_counterexample(max_n, start_n)
        else:
            return None
    
    def verify_property(
        self,
        property_name: str,
        n: int
    ) -> bool:
        """
        Verify if a property holds for a given n.
        
        Args:
            property_name: Name of property
            n: Value of n to test
            
        Returns:
            True if property holds, False if counterexample found
        """
        if property_name == 'periodicity':
            center = self.simulator.compute_center_column(n)
            result = self.periodicity_detector.check_aperiodicity(center)
            return not result['is_periodic']
        elif property_name == 'frequency':
            center = self.simulator.compute_center_column(n)
            freq = self.frequency_analyzer.compute_frequencies(center)
            deviation = freq['deviation_from_0.5']
            return deviation < 0.01
        else:
            return True
    
    def find_boundary_case(
        self,
        property_name: str,
        max_n: int = 10**6
    ) -> Optional[int]:
        """
        Find the boundary case where property might change.
        
        Args:
            property_name: Name of property
            max_n: Maximum n to search
            
        Returns:
            Boundary case n if found
        """
        # Binary search for boundary
        low = 1
        high = max_n
        
        while low < high:
            mid = (low + high) // 2
            holds = self.verify_property(property_name, mid)
            
            if holds:
                low = mid + 1
            else:
                high = mid
        
        if low < max_n:
            return low
        return None
    
    def _search_periodicity_counterexample(
        self,
        max_n: int,
        start_n: int
    ) -> Optional[Counterexample]:
        """Search for periodicity counterexample."""
        # Test at increasing scales
        test_points = self._generate_test_points(start_n, max_n)
        
        for n in test_points:
            center = self.simulator.compute_center_column(n)
            result = self.periodicity_detector.check_aperiodicity(center)
            
            if result['is_periodic']:
                return Counterexample(
                    n=n,
                    evidence={
                        'period': result['period'],
                        'is_periodic': True,
                        'confidence': result['confidence']
                    }
                )
        
        return None
    
    def _search_frequency_counterexample(
        self,
        max_n: int,
        start_n: int
    ) -> Optional[Counterexample]:
        """Search for frequency counterexample."""
        test_points = self._generate_test_points(start_n, max_n)
        
        for n in test_points:
            center = self.simulator.compute_center_column(n)
            freq = self.frequency_analyzer.compute_frequencies(center)
            deviation = freq['deviation_from_0.5']
            
            # Counterexample: deviation > 0.1 (significant bias)
            if deviation > 0.1:
                return Counterexample(
                    n=n,
                    evidence={
                        'ratio_ones': freq['ratio_ones'],
                        'deviation': deviation,
                        'significant_bias': True
                    }
                )
        
        return None
    
    def _generate_test_points(self, start: int, end: int, num_points: int = 20) -> list:
        """Generate test points between start and end."""
        if end - start <= num_points:
            return list(range(start, end + 1))
        
        # Logarithmic spacing
        import math
        points = []
        for i in range(num_points):
            ratio = i / (num_points - 1) if num_points > 1 else 0
            n = int(start * (end / start) ** ratio)
            points.append(n)
        
        return sorted(set(points))

