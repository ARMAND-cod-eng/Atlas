"""Database access layer.

Distinct from the LangGraph checkpointer, which manages its own psycopg pool for
graph state. Atlas has no tables of its own yet, so today this package backs only
the API's ``/health`` database probe. pgvector is enabled by the compose init
script, but nothing uses it yet.
"""
