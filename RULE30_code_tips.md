Stick with Python. Here's why:
Your performance targets (1B steps in <1 hour) are achievable with Python + NumPy + Numba JIT compilation. That's not slow—that's perfectly adequate. You're not trying to do microsecond-level trading or render 4K games.
The real speedup comes from better algorithms, not better languages. If you switch to C++, you get maybe 2-4x faster. If you optimize your algorithm (e.g., don't expand the grid naively, use a sliding window), you get 10-100x faster. The latter works in any language and is where your effort should go.
Switching to C++ costs you:

3-5x longer development time
Much harder for an agent to read/modify the code
More maintenance burden
Harder integration with Z3 solver and formal verification tools

The bottleneck in your project won't be the language—it'll be the algorithm design and whether you profile early to find actual slowdowns.
Bottom line: Use Python. Use NumPy for vectorization. Use Numba JIT for hot loops. Optimize your algorithms. You'll hit your targets just fine.