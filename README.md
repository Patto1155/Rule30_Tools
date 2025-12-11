# Rule 30 Tools

A comprehensive toolkit for solving the Wolfram Rule 30 Prize Problems, designed to enable coding agents to experiment, iterate, and eventually solve one or more of the three prize problems:

1. **Problem 1**: Does the center column always remain non-periodic? ($10,000 prize)
2. **Problem 2**: Does each color occur equally often on average in the center column? ($10,000 prize)
3. **Problem 3**: Does the center column pass statistical randomness tests? ($10,000 prize)

## Features

### Core Simulation
- High-performance Rule 30 simulator optimized for billions of steps
- Efficient bit-level operations (8x memory reduction)
- Center column extraction and analysis
- Checkpoint/resume capability for long-running computations

### Analysis Tools
- **Periodicity Detection**: Find repeating patterns (for Problem 1)
- **Frequency Analysis**: Analyze bit distribution and convergence (for Problem 2)
- **Randomness Testing**: Comprehensive NIST SP 800-22 test suite (for Problem 3)
- **Pattern Recognition**: Discover hidden structures and invariants

### Agent Integration (Rapid Experimentation)
- **One-line problem checks**: Test any problem instantly
- **Automatic result caching**: Repeated queries are instant
- **Quick hypothesis testing**: Test ideas in seconds, not hours
- **Incremental exploration**: Build knowledge step-by-step
- **Automatic history tracking**: Learn from every experiment
- **Intelligent suggestions**: Get next steps automatically
- **Strategy recommendations**: Know which approaches to try

### Experimentation Framework
- Hypothesis generation and ranking
- Experiment runner with progress tracking
- Counterexample search
- Batch experiment execution

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Patto1155/Rule30_Tools.git
cd Rule30_Tools

# Install dependencies
pip install numpy scipy
```

### Basic Usage

```python
from rule30_tools import Rule30AgentAPI

# Initialize the API
api = Rule30AgentAPI()

# Check Problem 1: Periodicity
result = api.problem1_check(steps=1_000_000)
print(f"Periodic: {result['is_periodic']}")

# Check Problem 2: Frequency
result = api.problem2_check(steps=1_000_000)
print(f"Ratio: {result['ratio_ones']:.4f}")

# Check Problem 3: Randomness
result = api.problem3_check(steps=1_000_000)
print(f"Tests passed: {result['tests_passed']}/{result['tests_total']}")
```

### Rapid Iteration for Agents

```python
from rule30_tools.agent.quick_iterate import QuickIterationHelper

helper = QuickIterationHelper()

# Quick hypothesis testing (seconds, not hours)
result = helper.quick_test("The center column becomes periodic after 1M steps")

# Incremental exploration
exploration = helper.incremental_explore(problem_number=2, max_iterations=5)

# Get intelligent suggestions
api = Rule30AgentAPI(enable_history=True)
suggestions = api.get_suggestions(problem_number=1)
```

## Project Structure

```
rule30_tools/
├── core/              # Core simulation infrastructure
│   ├── simulator.py   # Rule 30 simulator
│   ├── bit_array.py   # Efficient bit array
│   └── center_column.py
├── analysis/          # Analysis tools
│   ├── periodicity.py
│   ├── frequency.py
│   ├── randomness.py
│   └── patterns.py
├── experimentation/   # Experimentation framework
│   ├── hypothesis.py
│   ├── experiments.py
│   └── counterexample.py
├── agent/            # Agent integration (rapid iteration)
│   ├── api.py        # High-level API
│   ├── history.py    # Experiment history
│   ├── strategies.py # Strategy suggestions
│   └── quick_iterate.py
├── examples/         # Example scripts
└── docs/             # Documentation
```

## Documentation

- **[Quick Reference](RULE30_QUICK_REFERENCE.md)**: Quick overview of the three problems and tools
- **[Implementation Guide](RULE30_IMPLEMENTATION_GUIDE.md)**: Detailed implementation examples
- **[Development Plan](RULE30_TOOLS_PLAN.md)**: Complete development roadmap
- **[Agent Integration Summary](rule30_tools/docs/AGENT_INTEGRATION_SUMMARY.md)**: How agents can rapidly iterate
- **[Agent Rapid Iteration Guide](rule30_tools/docs/AGENT_RAPID_ITERATION.md)**: Detailed guide for rapid experimentation

## Key Design Principles

1. **Modularity**: Each tool works independently
2. **Efficiency**: Optimized for scale (billions of steps)
3. **Extensibility**: Easy to add new analysis methods
4. **Agent-Friendly**: Clear APIs, good error messages
5. **Learning**: Track experiments, learn from failures

## Performance Targets

- Simulate 1 billion steps in < 1 hour
- Memory usage < 16GB for large computations
- Period detection for sequences up to 10^9 bits
- Statistical tests for sequences up to 10^12 bits

## Example: Rapid Iteration

```python
from rule30_tools import Rule30AgentAPI
from rule30_tools.agent.quick_iterate import QuickIterationHelper

api = Rule30AgentAPI(enable_history=True)
helper = QuickIterationHelper(api)

# Iteration 1: Quick test (2 seconds)
result = helper.quick_test("Hypothesis statement")

# Iteration 2: Scale up (15 seconds)
result = api.problem1_check(steps=100_000)

# Iteration 3: Get suggestions (< 1 second)
suggestions = api.get_suggestions(problem_number=1)

# Iteration 4: Follow suggestion (2 minutes)
result = api.problem1_check(steps=1_000_000)

# Total: ~3 minutes for 4 iterations
# Without tools: ~2 hours per iteration
```

## Contributing

This is a research project for solving the Wolfram Rule 30 Prize Problems. Contributions, experiments, and discoveries are welcome!

## License

See LICENSE file for details.

## References

- [Wolfram Rule 30 Prizes](https://writings.stephenwolfram.com/2019/10/announcing-the-rule-30-prizes/)
- [NIST SP 800-22](https://csrc.nist.gov/publications/detail/sp/800-22/rev-1a/final) - Randomness test suite

## Status

✅ Core simulation infrastructure  
✅ Analysis tools (periodicity, frequency, randomness, patterns)  
✅ Experimentation framework  
✅ Agent integration for rapid iteration  
⏳ Proof framework (planned)  
⏳ Visualization tools (planned)  

---

**Goal**: Enable systematic discovery of solutions to the Rule 30 Prize Problems through rapid experimentation and iteration.
