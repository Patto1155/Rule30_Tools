"""
Example: How agents can use Rule 30 Tools for rapid experimentation and iteration.

This demonstrates the key features that enable fast iteration:
1. One-line problem checks
2. Quick hypothesis testing
3. Incremental exploration
4. Learning from history
5. Strategy suggestions
"""

from rule30_tools.agent.api import Rule30AgentAPI
from rule30_tools.agent.quick_iterate import QuickIterationHelper
from rule30_tools.agent.strategies import StrategyGenerator
from rule30_tools.agent.history import ExperimentHistory


def example_1_quick_problem_check():
    """Example 1: Quick one-liner problem checks."""
    print("=" * 60)
    print("Example 1: Quick Problem Checks")
    print("=" * 60)
    
    api = Rule30AgentAPI()
    
    # Problem 1: One line to check periodicity
    result1 = api.problem1_check(steps=100000)
    print(f"Problem 1: Periodic = {result1['is_periodic']}, "
          f"Confidence = {result1['confidence']:.2f}")
    
    # Problem 2: One line to check frequency
    result2 = api.problem2_check(steps=100000)
    print(f"Problem 2: Ratio = {result2['ratio_ones']:.4f}, "
          f"Converging = {result2['converging']}")
    
    # Problem 3: One line to check randomness
    result3 = api.problem3_check(steps=100000)
    print(f"Problem 3: Tests passed = {result3['tests_passed']}/{result3['tests_total']}")
    
    print()


def example_2_quick_hypothesis_testing():
    """Example 2: Rapid hypothesis testing."""
    print("=" * 60)
    print("Example 2: Quick Hypothesis Testing")
    print("=" * 60)
    
    api = Rule30AgentAPI()
    
    # Test a hypothesis in seconds
    hypotheses = [
        "The center column becomes periodic after 1 million steps",
        "The frequency of ones converges to 0.5",
        "The sequence passes all randomness tests"
    ]
    
    for hypothesis in hypotheses:
        result = api.quick_test_hypothesis(hypothesis, test_steps=10000)
        print(f"\nHypothesis: {hypothesis}")
        print(f"  Supports: {result.get('supports', 'Unknown')}")
        print(f"  Test type: {result.get('test_type', 'Unknown')}")
    
    print()


def example_3_incremental_exploration():
    """Example 3: Incremental exploration building knowledge."""
    print("=" * 60)
    print("Example 3: Incremental Exploration")
    print("=" * 60)
    
    helper = QuickIterationHelper()
    
    # Explore Problem 2 incrementally
    exploration = helper.incremental_explore(problem_number=2, max_iterations=3)
    
    print(f"Explored Problem {exploration['problem_number']} in {len(exploration['iterations'])} iterations")
    for iteration in exploration['iterations']:
        result = iteration['result']
        print(f"  Iteration {iteration['iteration']}: {iteration['steps']:,} steps, "
              f"deviation = {result.get('deviation_from_0.5', 0):.6f}")
    
    print(f"\nFindings: {exploration['findings']}")
    print(f"Recommendations: {exploration['recommendations']}")
    print()


def example_4_learning_from_history():
    """Example 4: Learning from experiment history."""
    print("=" * 60)
    print("Example 4: Learning from History")
    print("=" * 60)
    
    api = Rule30AgentAPI(enable_history=True)
    
    # Run some experiments
    api.problem1_check(steps=10000)
    api.problem1_check(steps=100000)
    api.problem2_check(steps=10000)
    
    # Get suggestions based on what's been tried
    suggestions = api.get_suggestions(problem_number=1)
    print("Suggestions for Problem 1:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")
    
    # Get insights
    if api.history:
        insights = api.history.generate_insights()
        print("\nInsights from history:")
        for insight in insights:
            print(f"  - {insight}")
    
    print()


def example_5_strategy_suggestions():
    """Example 5: Getting strategy suggestions."""
    print("=" * 60)
    print("Example 5: Strategy Suggestions")
    print("=" * 60)
    
    api = Rule30AgentAPI(enable_history=True)
    strategy_gen = StrategyGenerator(history=api.history)
    
    # Get strategies for Problem 1
    strategies = strategy_gen.suggest_strategies(problem_number=1)
    
    print(f"Found {len(strategies)} strategies for Problem 1:\n")
    for i, strategy in enumerate(strategies[:3], 1):
        print(f"{i}. {strategy.description}")
        print(f"   Success probability: {strategy.success_probability:.1%}")
        print(f"   Estimated time: {strategy.estimated_time}")
        print()
    
    print()


def example_6_rapid_prototyping():
    """Example 6: Rapid prototyping of ideas."""
    print("=" * 60)
    print("Example 6: Rapid Prototyping")
    print("=" * 60)
    
    helper = QuickIterationHelper()
    
    # Prototype an idea quickly
    ideas = [
        "Check if periodicity appears at powers of 2",
        "Test frequency convergence at different scales",
        "Look for patterns in the first 1000 bits"
    ]
    
    for idea in ideas:
        result = helper.rapid_prototype(idea)
        print(f"\nIdea: {idea}")
        print(f"  Viable: {result['viable']}")
        print(f"  Next steps: {result['next_steps'][0] if result['next_steps'] else 'None'}")
    
    print()


def example_7_batch_testing():
    """Example 7: Batch testing multiple hypotheses."""
    print("=" * 60)
    print("Example 7: Batch Hypothesis Testing")
    print("=" * 60)
    
    helper = QuickIterationHelper()
    
    # Test multiple hypotheses at once
    hypotheses = [
        "The sequence is periodic with period 100",
        "The sequence is periodic with period 1000",
        "The frequency converges to 0.5",
        "The sequence passes randomness tests"
    ]
    
    results = helper.batch_test_hypotheses(hypotheses)
    
    print(f"Tested {len(hypotheses)} hypotheses:\n")
    for hypothesis, result in zip(hypotheses, results):
        supports = result.get('supports', 'Unknown')
        print(f"  {hypothesis[:50]}... -> {supports}")
    
    print()


def example_8_scale_comparison():
    """Example 8: Comparing results across scales."""
    print("=" * 60)
    print("Example 8: Scale Comparison")
    print("=" * 60)
    
    helper = QuickIterationHelper()
    
    # Compare Problem 2 at different scales
    comparison = helper.compare_scales(problem_number=2, scales=[1000, 10000, 100000])
    
    print("Frequency analysis across scales:")
    for comp in comparison['comparisons']:
        result = comp['result']
        print(f"  {comp['scale']:,} steps: "
              f"ratio = {result['ratio_ones']:.4f}, "
              f"deviation = {result['deviation_from_0.5']:.6f}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Rule 30 Tools: Agent Rapid Iteration Examples")
    print("=" * 60 + "\n")
    
    example_1_quick_problem_check()
    example_2_quick_hypothesis_testing()
    example_3_incremental_exploration()
    example_4_learning_from_history()
    example_5_strategy_suggestions()
    example_6_rapid_prototyping()
    example_7_batch_testing()
    example_8_scale_comparison()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)

