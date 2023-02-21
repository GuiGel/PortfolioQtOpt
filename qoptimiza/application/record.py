"""Utilities to keep components output order in a streamlit's `session_state`.
"""
from dataclasses import dataclass
from typing import Any

from loguru import logger
from streamlit.runtime.state.session_state_proxy import SessionStateProxy


@dataclass
class Record:
    """Record a value with an associated positional number."""

    order: int
    """The position of the record in the :data:`~qoptimiza.application.core.memory`"""

    value: Any
    """The value associated with these order."""


def delete_posterior_records(order: int, memory: SessionStateProxy):
    """Remove :class:`~qoptimiza.application.record.Record`s from of
    `streamlit.SessionStateProxy`.

    Given a `st.session_state` object, remove all the keys that hold a
    :class:`Record` such that :attr:`~qoptimiza.application.record.Record.order` is
    superior or equal to `order`.

    Args:
        order (int): The desired order.
        memory (SessionStateProxy): The streamlit current session state.
    """
    for key in memory:
        val = memory[key]
        # TODO Understand why the assertion isinstance(val, Record) doesn't works.
        if str(type(val)) == str(Record) and val.order >= order:
            logger.trace(f"Delete {key} from st.session_state")
            del memory[key]
