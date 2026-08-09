"""Tools the agents can call.

Each tool normalizes its provider's shape into an Atlas model and raises a typed
Atlas error, so a provider swap is contained to one module and callers only ever
catch one exception type.
"""
