"""Constrained action system: vocabulary, parser, input controller.

Model output -> parser (validated) -> ActionController -> uinput backend.
Unrestricted LLM output never reaches the input backend (Rule 3).
"""

from .action_space import Action, ActionSpace, DEFAULT_ACTIONS
from .parser import Decision, parse_decision
from .controller import (
    InputController,
    UInputBackend,
    find_backend_device,
    key_name,
)

__all__ = [
    "Action",
    "ActionSpace",
    "DEFAULT_ACTIONS",
    "Decision",
    "parse_decision",
    "InputController",
    "UInputBackend",
    "find_backend_device",
    "key_name",
]
