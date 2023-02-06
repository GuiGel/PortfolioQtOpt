from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict

import streamlit as st
from loguru import logger


@dataclass
class Item:
    order: int = 0
    calls: int = 0
    args: Any = None
    val: Any = None


class Register:

    __MEMORY: DefaultDict[str, Item] = defaultdict(Item)

    def __call__(self, name: str, order: int, *parameters: str) -> None:
        logger.debug(f"call register")

        self.__MEMORY[name].calls += 1
        self.__MEMORY[name].order = order
        self.__MEMORY[name].args = tuple(map(st.session_state.get, parameters))

        logger.info(f"update memory {self.__MEMORY[name]=}")

    def __getattr__(self, name: str) -> Item:
        if self.__MEMORY.get(name, False):
            return self.__MEMORY[name]
        else:
            raise AttributeError(f"The function {name} has not been registered yet!")


register = Register()

if __name__ == "__main__":
    register("f1", 10, ["a", "b"])
    print(register.f1)
