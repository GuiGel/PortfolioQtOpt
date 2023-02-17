from dataclasses import dataclass
from typing import Any

from loguru import logger
from streamlit.runtime.state.session_state_proxy import SessionStateProxy


@dataclass
class Record:
    order: int
    value: Any


def delete_posterior_records(order: int, memory: SessionStateProxy):
    # Remove all the key
    for key in memory:
        val = memory[key]
        if str(type(val)) == str(Record) and val.order >= order:
            logger.trace(f"Delete {key} from st.session_state")
            del memory[key]
