"""Atlas — a multi-agent research platform.

A planner decomposes a question into independent subtasks, researchers run them
in parallel against web search, a critic reviews the findings against the
sources actually retrieved and can send specific branches back, and a
synthesizer writes the cited answer. ``atlas.api`` enqueues runs and relays
their events; ``atlas.worker`` executes them.
"""

__version__ = "0.1.0"
