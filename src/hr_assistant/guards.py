from __future__ import annotations
from functools import wraps
from typing import Awaitable, Callable

from .config import AGENTS
from .logging import get_logger

log = get_logger("guards")

NodeFn = Callable[[dict], Awaitable[dict]]


def accepts_from_config(agent_name: str):
    """Wrap an async node so it only sees fields in AGENTS[agent_name]['accepts']."""
    allowed = set(AGENTS[agent_name]["accepts"])

    def decorator(fn: NodeFn) -> NodeFn:
        @wraps(fn)
        async def wrapper(state: dict) -> dict:
            stripped = {k: v for k, v in state.items() if k in allowed}
            dropped = set(state.keys()) - allowed - {"routing_path"}
            if dropped:
                log.info("fields_stripped", agent=agent_name, dropped=sorted(dropped))
            return await fn(stripped)
        return wrapper

    return decorator


import re

from .state import Sensitivity

_CPR_PATTERN = re.compile(r"\b\d{6}-?\d{4}\b")
_PATIENT_ID_PATTERN = re.compile(r"\bP-\d{3,}\b", re.IGNORECASE)


def classify_with_reason(query: str) -> tuple[Sensitivity, str | None]:
    if _CPR_PATTERN.search(query):
        return "clinical", "CPR pattern detected"
    if _PATIENT_ID_PATTERN.search(query):
        return "clinical", "patient ID pattern detected"
    return "public", None


def classify_query_sensitivity(query: str) -> Sensitivity:
    level, _ = classify_with_reason(query)
    return level


from .config import SENSITIVE_FIELDS, SENSITIVITY_LEVELS


class SensitivityViolation(Exception):
    pass


def validate_flow(agent_names: list[str], assistant_sensitivity: Sensitivity) -> None:
    """Raise SensitivityViolation if any agent in the flow would expose
    sensitive fields at too low a sensitivity level."""
    assistant_level = SENSITIVITY_LEVELS.index(assistant_sensitivity)
    for name in agent_names:
        spec = AGENTS[name]
        agent_level = SENSITIVITY_LEVELS.index(spec["sensitivity"])
        accepts_sensitive = any(f in spec["accepts"] for f in SENSITIVE_FIELDS)
        if accepts_sensitive and agent_level < assistant_level:
            raise SensitivityViolation(
                f"{name} accepts sensitive fields but is rated "
                f"'{spec['sensitivity']}' (< '{assistant_sensitivity}')"
            )
