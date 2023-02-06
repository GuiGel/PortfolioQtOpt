from loguru import logger
from collections import defaultdict
from typing import DefaultDict, Any, Optional, List
from dataclasses import dataclass, field
import streamlit as st

@dataclass
class Item:
    order: int = 0
    calls: int = 0
    args: Any = None
    val: Any = None


def registry(name: str, order: int, parameters: List[str]) -> None:
    logger.debug(f"register_args")

    # Update call number to the form
    logger.info(f"{memo[name]=}")
    memo[name].calls += 1
    memo[name].order = order

    logger.info(f"v1 = {st.session_state.f1_v1}")
    logger.info(f"v2 = {st.session_state.f1_v2}")
    logger.info(f"F1 = {st.session_state.keys()}")

    memo[name].args = tuple(map(st.session_state.get, parameters))
    logger.info(f"{memo[name].args=}")

@dataclass
class controller:
    forms: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_form(self):
        pass




logger.info("Initialize MEMORY")
MEMORY: DefaultDict[str, int] = defaultdict(int)

memo: DefaultDict[str, Item] = defaultdict(Item)


if __name__ == "__main__":
    memo["form1"] = Item(0, 0, "a")
    print(memo)