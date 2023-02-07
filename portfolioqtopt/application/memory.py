"""Module that implement to memory logic of the application.

The app use :class:`streamlit.form`s. Outside of forms, any time a user interacts with 
a widget the app's script is rerun. What st.form does is make it so users can 
interact with the widgets as much as they want, without causing a rerun! Instead, to 
update the app, the user should click on the form's submit button.

But by doing this, the fact that previous buttons have been clicked or not disappears.
As the `st.forms` used depend on each other, it becomes necessary to set up a memory.
To ensure that this memory is not reset every time a button is pressed, it must be
initialized outside the scope of the module where our application is defined.
"""
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
    """A simple register.

    Example:

        >>> register = Register()

        To register the function `f1`, and it's position in the order of execution.
        >>> register("f1", 10)
        >>> register.f1
        Item(order=10, calls=1, args=(), val=None)
    """

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
    register("f1", 10)
    print(register.f1)
