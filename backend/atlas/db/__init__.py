"""Database access layer.

Distinct from the LangGraph checkpointer, which manages its own psycopg pool for
graph state. This package is for Atlas's own tables — starting with the pgvector
embedding store in a later phase.
"""
