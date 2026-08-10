"""Atlas evaluation suite.

Deliberately a separate package from ``atlas``: eval code must never ship inside
the runtime, and the separation makes it obvious when something has leaked the
wrong way. Held to the same bar regardless — ruff and ``mypy --strict`` cover it.
"""
