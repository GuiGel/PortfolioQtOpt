# File to explore ideas about forms and session state
# Steamlit run the module evert time a button is click.
import streamlit as st
from loguru import logger
from typing import Dict, Any, Tuple, List
from collections import defaultdict
from contextlib import contextmanager
from application.memory import MEMORY, memo, Item
import pandas as pd

class App:

    @staticmethod
    def register_args(name: str, order: int, parameters: List[str]) -> None:
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

    @staticmethod
    def f1():
        logger.info("enter f1")
        with st.form(key="f1"):
            st.text_input("val1", 10, key="f1_v1")
            st.text_input("val2", 30, key="f1_v2")

            # Prepare calc_f1 arguments
            args = (st.session_state.f1_v1, st.session_state.f1_v2)

            logger.info(f"v1 = {st.session_state.f1_v1}")
            logger.info(f"v2 = {st.session_state.f1_v2}")
            logger.info(f"F1 = {st.session_state.keys()}")

            submit = st.form_submit_button(
                "submit values",
                on_click=App.register_args,
                args=("f1", 0, ["f1_v1", "f1_v2"]),
            )
            logger.info(f"{submit=}")
            return submit

    @staticmethod
    @st.cache
    def calc_1(a, b) -> float:
        logger.debug(f"inside calc_1 {locals()}")
        return a + b

    @staticmethod
    def f2():
        logger.info("enter f2")
        with st.form(key="f2"):
            a = st.text_input("val1", 10, key="f2_v1")
            b = st.text_input("val2", 30, key="f2_v2")
            args = (a, b)
            submit = st.form_submit_button(
                "submit values",
                on_click=App.register_args,
                args=("f2", 1, ["f2_v1", "f2_v2"]),
            )
            logger.info(f"submit 2: {submit}")
            return submit

    @staticmethod
    @st.cache
    def calc_2(a, b, c) -> float:
        logger.debug("inside calc_2")
        return a + b + c

    @staticmethod
    def __call__():
        logger.info(f"{'-':->50}")
        logger.info(f"{' Enter App ':^50}")
        logger.info(f"{'-':->50}")

        logger.info("submit 1")
        submit_1 = App.f1()
        logger.info(f"{submit_1=}")

        if submit_1:
            logger.info(f"Submit 1 {submit_1}. Computation will be made.")
            memo["f1"].val = App.calc_1(*memo["f1"].args)

            logger.info(f"{memo=}")
            logger.info(f"{list(st.session_state.keys())=}")

            st.text("F1 TEST WILL DISAPPEARED WHEN F2 CALL")

        # We collect f2 arguments and call the corresponding function.
        if memo["f1"].calls:

            # Display preceding results and use form to compute new results.
            # When a button is pressed the script is run again and the submit_1 will be
            # false.
            st.text(f"F1 TEST STAY {memo['f1'].val}")


            submit_2 = App.f2()
            logger.info(f"{submit_2=}")

            if submit_2:
                logger.info(f"Submit 1 {submit_2}. Computation will be made.")
                args = memo["f2"].args + (memo["f1"].val,)
                memo["f2"].val = App.calc_2(*args)
                st.text(f"F2 TEST DISAPPEAR {memo['f2'].val}")

            if memo["f2"].calls:
                logger.info(f"previous step done. Display results.")
                st.text(f"F2 TEST STAY {memo['f2'].val}")

        logger.info(f"memo = {memo}")

        logger.info(f"{'-':->50}")
        logger.info(f"{' Done ':^50}")
        logger.info(f"{'-':->50}")

app = App()

if __name__ == "__main__":
    app()