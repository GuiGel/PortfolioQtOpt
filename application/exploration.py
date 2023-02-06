# File to explore ideas about forms and session state
# Streamlit run the module evert time a button is click.
import streamlit as st
from loguru import logger

from application.memory import register


class App:
    @staticmethod
    def f1():
        logger.info("enter f1")
        with st.form(key="f1"):
            st.text_input("val1", 10, key="f1_v1")
            st.text_input("val2", 30, key="f1_v2")

            logger.info(f"F1 = {st.session_state.keys()}")

            submit = st.form_submit_button(
                "submit values",
                on_click=register,
                args=("f1", 0, "f1_v1", "f1_v2"),  # register arguments
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
            st.text_input("val1", 10, key="f2_v1")
            st.text_input("val2", 30, key="f2_v2")

            submit = st.form_submit_button(
                "submit values",
                on_click=register,
                args=("f2", 1, "f2_v1", "f2_v2"),
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

        # ----- collect arguments for step 1
        submit_1 = App.f1()
        logger.info(f"{submit_1=}")

        # ----- argument for step 1 has been collected
        if hasattr(register, "f1"):

            # ----- compute step 1
            if submit_1:
                register.f1.val = App.calc_1(*register.f1.args)

            # ----- display results of step 1
            st.text(f"F1 TEST STAY {register.f1.val}")

            # ----- collect arguments for step 2
            submit_2 = App.f2()
            logger.info(f"{submit_2=}")

            #  ----- the button in step 2 has been pressed
            if submit_2:

                # ----- compute step 2
                args = register.f2.args + (register.f1.val,)
                register.f2.val = App.calc_2(*args)

            # ----- display results step 2
            if hasattr(register, "f2"):
                st.text(f"F2 TEST STAY {register.f2.val}")

        logger.info(f"memo = {register}")

        logger.info(f"{'-':->50}")
        logger.info(f"{' Done ':^50}")
        logger.info(f"{'-':->50}")


app = App()

if __name__ == "__main__":
    app()
