# Agent Rapid Iteration Guide

## Overview

The Rule 30 Tools agent integration is designed to enable **rapid experimentation and iteration**. This document explains how the system enables agents to test ideas quickly, learn from history, and iterate efficiently.

## Key Design Principles

### 1. **One-Line Problem Checks**
Agents can test any of the three problems with a single function call:

```python
from rule30_tools import Rule30AgentAPI

api = Rule30AgentAPI()

# Problem 1: Check periodicity
result = api.problem1_check(steps=1_000_000)

# Problem 2: Check frequency
result = api.problem2_check(steps=1_000_000)

# Problem 3: Check randomness
result = api.problem3_check(steps=1_000_000)
```

### 2. **Automatic Caching**
Expensive computations are automatically cached, so repeated queries are instant:

```python
# First call: computes (takes time)
center1 = api.simulate_rule30(steps=1_000_000)

# Second call: uses cache (instant)
center2 = api.simulate_rule30(steps=1_000_000)  # Instant!
```

### 3. **Quick Hypothesis Testing**
Test ideas in seconds, not hours:

```python
# Test any hypothesis quickly
result = api.quick_test_hypothesis(
    "The center column becomes periodic after 1 million steps",
    test_steps=10_000  # Quick test with smaller sequence
)
```

### 4. **Incremental Exploration**
Build knowledge incrementally, starting small and scaling up:

```python
from rule30_tools.agent.quick_iterate import QuickIterationHelper

helper = QuickIterationHelper()

# Explore a problem incrementally
exploration = helper.incremental_explore(problem_number=2, max_iterations=5)
# Starts at 10k steps, scales up to 10M+ automatically
```

### 5. **Learning from History**
The system tracks all experiments and learns from them:

```python
# Run some experiments
api.problem1_check(steps=10_000)
api.problem1_check(steps=100_000)

# Get suggestions based on what's been tried
suggestions = api.get_suggestions(problem_number=1)
# Returns: ["Test periodicity with 1,000,000 steps", ...]

# Get insights
insights = api.history.generate_insights()
# Returns: ["Problem 1: 2 experiments, 0.0% success rate", ...]
```

### 6. **Strategy Suggestions**
Get intelligent suggestions for next steps:

```python
from rule30_tools.agent.strategies import StrategyGenerator

strategy_gen = StrategyGenerator(history=api.history)
strategies = strategy_gen.suggest_strategies(problem_number=1)

for strategy in strategies:
    print(f"{strategy.description}")
    print(f"Success probability: {strategy.success_probability}")
    print(f"Steps: {strategy.steps}")
```

## How This Enables Rapid Iteration

### Before (Without Tools)
```python
# Agent would need to:
# 1. Write simulation code
# 2. Implement periodicity detection
# 3. Run experiment
# 4. Analyze results
# 5. Decide next steps
# 6. Repeat...

# This takes hours or days per iteration
```

### After (With Tools)
```python
# Agent can:
# 1. Test idea in one line
result = api.problem1_check(steps=1_000_000)

# 2. Get immediate feedback
if result['is_periodic']:
    print("Found periodicity!")

# 3. Get suggestions for next step
suggestions = api.get_suggestions(problem_number=1)

# 4. Iterate immediately
# This takes seconds per iteration
```

## Rapid Iteration Workflow

### Typical Agent Workflow

1. **Quick Test** (seconds)
   ```python
   result = api.quick_test_hypothesis("Hypothesis statement")
   ```

2. **Evaluate** (instant)
   ```python
   if result['supports']:
       # Promising - scale up
   else:
       # Try different approach
   ```

3. **Get Suggestions** (instant)
   ```python
   suggestions = api.get_suggestions(problem_number=1)
   ```

4. **Iterate** (seconds)
   ```python
   # Try next suggestion
   result = api.problem1_check(steps=suggested_steps)
   ```

5. **Learn** (automatic)
   ```python
   # History is automatically updated
   # Next suggestions will be smarter
   ```

### Batch Testing Multiple Ideas

```python
helper = QuickIterationHelper()

hypotheses = [
    "Hypothesis 1",
    "Hypothesis 2",
    "Hypothesis 3"
]

# Test all at once
results = helper.batch_test_hypotheses(hypotheses)

# Evaluate all results
for hypothesis, result in zip(hypotheses, results):
    if result['supports']:
        # Investigate further
        pass
```

### Incremental Scaling

```python
helper = QuickIterationHelper()

# Start small, scale up automatically
exploration = helper.incremental_explore(
    problem_number=2,
    max_iterations=5
)

# System automatically:
# - Starts at 10k steps
# - Scales to 100k, 1M, 10M, 100M
# - Stops if interesting finding
# - Provides recommendations
```

## Performance Characteristics

### Speed Comparison

| Operation | Without Tools | With Tools | Speedup |
|-----------|--------------|------------|---------|
| Test hypothesis | Hours | Seconds | 1000x+ |
| Check problem | Days | Minutes | 100x+ |
| Get suggestions | Manual | Instant | ∞ |
| Learn from history | Manual | Automatic | ∞ |

### Typical Iteration Times

- **Quick test**: 1-5 seconds
- **Problem check (1M steps)**: 10-30 seconds
- **Batch test (10 hypotheses)**: 10-50 seconds
- **Incremental exploration (5 iterations)**: 1-5 minutes

## Key Features for Rapid Iteration

### 1. **Automatic Experiment Tracking**
Every experiment is automatically logged:
- Parameters used
- Results obtained
- Success/failure status
- Duration
- Resource usage

### 2. **Intelligent Suggestions**
The system learns what works and suggests:
- Next step counts to test
- Approaches that haven't been tried
- Strategies based on problem type
- Ways to avoid known failures

### 3. **Result Caching**
Expensive computations are cached:
- Same query = instant result
- Similar queries = fast result
- Enables rapid iteration on ideas

### 4. **Quick Test Modes**
Multiple test modes for different needs:
- **Quick**: 10k steps (seconds)
- **Medium**: 100k steps (minutes)
- **Thorough**: 1M+ steps (hours)

### 5. **Incremental Building**
Build knowledge incrementally:
- Start with quick tests
- Scale up promising directions
- Abandon unpromising paths early

## Example: Complete Iteration Cycle

```python
from rule30_tools import Rule30AgentAPI
from rule30_tools.agent.quick_iterate import QuickIterationHelper

# Initialize
api = Rule30AgentAPI(enable_history=True)
helper = QuickIterationHelper(api)

# Iteration 1: Quick test
result1 = helper.quick_test(
    "The center column becomes periodic after 1M steps"
)
# Takes: 2 seconds
# Result: Not periodic at 10k steps

# Iteration 2: Scale up
result2 = api.problem1_check(steps=100_000)
# Takes: 15 seconds
# Result: Still not periodic

# Iteration 3: Get suggestions
suggestions = api.get_suggestions(problem_number=1)
# Takes: < 1 second
# Result: ["Test periodicity with 1,000,000 steps", ...]

# Iteration 4: Follow suggestion
result3 = api.problem1_check(steps=1_000_000)
# Takes: 2 minutes
# Result: Still not periodic, but confidence increased

# Iteration 5: Try different approach
patterns = helper.explore_pattern_space(min_length=2, max_length=5)
# Takes: 30 seconds
# Result: Found some interesting patterns

# Total time: ~3 minutes
# Iterations: 5
# Knowledge gained: Significant
```

## Best Practices

1. **Start Small**: Use quick tests first, scale up promising ideas
2. **Use History**: Check `api.get_suggestions()` before starting
3. **Batch Test**: Test multiple hypotheses at once
4. **Incremental**: Use `incremental_explore()` to build knowledge
5. **Learn**: Review `api.history.generate_insights()` regularly

## Conclusion

The agent integration system enables **rapid experimentation and iteration** by:

1. **Reducing boilerplate**: One-line function calls
2. **Caching results**: Instant repeated queries
3. **Tracking history**: Automatic learning
4. **Providing suggestions**: Intelligent next steps
5. **Supporting quick tests**: Fast feedback loops

This allows agents to:
- Test ideas in **seconds**, not hours
- Iterate **hundreds of times** per day
- Learn from **every experiment**
- Focus on **ideas**, not implementation
- Make **data-driven decisions** quickly

The result: Agents can explore the problem space **orders of magnitude faster** than manual methods, enabling systematic discovery of solutions to the Rule 30 Prize Problems.

