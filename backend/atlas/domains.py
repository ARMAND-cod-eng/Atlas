"""Domains: what Atlas researches from, and what each stage is told.

The graph is one engine. A domain is what changes with the use it is put to:
the prompt each stage is given, and whether researchers read the regulation
library before the web.

* ``general`` is Atlas as the 50-item general baseline measured it: web search
  only, prompt for prompt.
* ``compliance`` is the EU AI Act and GDPR copilot: researchers read the
  library first and cite provisions by label; web results are kept as
  secondary sources, never the authority for what the law says.

A run's domain comes from its config (``configurable["domain"]``), falling back
to ``ATLAS_DOMAIN``. Per run, so the eval can score both domains in one process.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from langgraph.config import get_config

from atlas import prompts
from atlas.config import get_settings

DomainName = Literal["general", "compliance"]


@dataclass(frozen=True, slots=True)
class Domain:
    name: DomainName
    planner: str
    researcher: str
    synthesizer: str
    critic: str
    # Researchers read the regulation library before searching the web.
    library: bool


GENERAL = Domain(
    name="general",
    planner=prompts.GENERAL_PLANNER,
    researcher=prompts.GENERAL_RESEARCHER,
    synthesizer=prompts.GENERAL_SYNTHESIZER,
    critic=prompts.GENERAL_CRITIC,
    library=False,
)

COMPLIANCE = Domain(
    name="compliance",
    planner=prompts.COMPLIANCE_PLANNER,
    researcher=prompts.COMPLIANCE_RESEARCHER,
    synthesizer=prompts.COMPLIANCE_SYNTHESIZER,
    critic=prompts.COMPLIANCE_CRITIC,
    library=True,
)

DOMAINS: dict[str, Domain] = {domain.name: domain for domain in (GENERAL, COMPLIANCE)}


# For code that runs a stage outside a graph (the eval's single-call baseline),
# where there is no run config to carry the domain.
_OUTSIDE_A_GRAPH: ContextVar[str | None] = ContextVar("atlas_domain", default=None)


@contextmanager
def using_domain(name: DomainName) -> Iterator[Domain]:
    """Run stages outside a graph in ``name``'s domain."""
    token = _OUTSIDE_A_GRAPH.set(name)
    try:
        yield DOMAINS[name]
    finally:
        _OUTSIDE_A_GRAPH.reset(token)


def current_domain() -> Domain:
    """The running graph's domain; else ``using_domain``'s; else ATLAS_DOMAIN."""
    try:
        name = get_config().get("configurable", {}).get("domain")
    except RuntimeError:
        name = None
    return DOMAINS[name or _OUTSIDE_A_GRAPH.get() or get_settings().atlas_domain]
