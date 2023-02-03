# File to explore ideas about forms and session state
# Steamlit run the 
import compileall
import streamlit as st
from loguru import logger
from typing import Dict, Any, Tuple
from collections import defaultdict
from contextlib import contextmanager
from application.memory import MEMORY, memo, Item

class App:

    @staticmethod
    def register_args(name: str, order: int) -> None:
        logger.debug(f"register_args")

        # Update call number to the form
        logger.info(f"{memo[name]=}")
        memo[name].calls += 1
        memo[name].order = order

        logger.info(f"v1 = {st.session_state.f1_v1}")
        logger.info(f"v2 = {st.session_state.f1_v2}")
        logger.info(f"F1 = {st.session_state.keys()}")

    @staticmethod
    def f1():
        logger.info("enter f1")
        with st.form(key="f1"):
            st.text_input("val1", 10, key="f1_v1")
            st.text_input("val2", 30, key="f1_v2")
            submit = st.form_submit_button(
                "submit values",
                on_click=App.register_args,
                args=("f1", 0),
            )
            logger.info(f"{submit=}")
            if submit:
                MEMORY["f1"] += 1
                logger.info(f"{MEMORY=}")
            else:
                logger.info(f"{MEMORY=}")
            return submit

    @staticmethod
    @st.cache
    def calc_1(a, b) -> float:
        return a + b

    @staticmethod
    def f2():
        logger.info("enter f2")
        with st.form(key="f2"):
            st.text_input("val1", 10, key="f2_v1")
            st.text_input("val2", 30, key="f2_v2")
            submit = st.form_submit_button(
                "submit values",
                on_click=App.register_args,
                args=("f2", 0),
            )
            logger.info(f"submit 1: {submit}")
            if submit:
                MEMORY["f2"] += 1
                logger.info(f"{MEMORY=}")
            else:
                logger.info(f"{MEMORY=}")
            return submit

    @staticmethod
    @st.cache
    def calc_2(a, b, c) -> float:
        return a + b + c

    @staticmethod
    def __call__():
        logger.info(f"{'-':->50}")
        logger.info(f"{' Enter App ':^50}")
        logger.info(f"{'-':->50}")

        submit_1 = App.f1()
        logger.info(f"submit 1: {submit_1}")

        if submit_1:
            args = st.session_state.f1_v1, st.session_state.f1_v2
            c = App.calc_1(*args)
            memo["f1"].val = c  # TODO Not very robust to give the name like this!
            logger.info(f"{c=}")
            logger.info(f"{memo=}")
        else:
            if st.session_state.f1_v1:
                logger.info(f"{st.session_state.f1_v1=}")


        submit_2 = App.f2()
        logger.info(f"submit 2: {submit_2}")
        if submit_2:
            args = st.session_state.f2_v1, st.session_state.f2_v2
            logger.info(
                f"f1_v1, f1_v2 = {st.session_state.f1_v1}, {st.session_state.f1_v2}"
            )
            c = App.calc_2(*args, c)
            logger.info(f"{c=}")
        else:
            if st.session_state.f2_v1:
                logger.info(f"{st.session_state.f2_v1=}")

        logger.info(f"{'-':->50}")
        logger.info(f"{' Done ':^50}")
        logger.info(f"{'-':->50}")

app = App()

if __name__ == "__main__":
    app()