from dataclasses import dataclass

import streamlit as st
from typing import Any, List, Dict
from streamlit.runtime.state.session_state_proxy import SessionStateProxy
from loguru import logger

@dataclass
class Record:
    order: int
    value: Any


def delete_posterior_records(order: int, memory: SessionStateProxy):
    # Remove all the key
    for key in memory:
        val = memory[key]
        logger.info(f"{isinstance(val, Record)=}, {val=}")
        if str(type(val)) == str(Record) and val.order >= order:
            del memory[key]
