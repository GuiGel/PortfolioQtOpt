import streamlit as st
from typing import Tuple
from loguru import logger
from dataclasses import dataclass


memory = st.session_state

@dataclass
class Record:
    order: int = 0
    count: int = 0
    last: int = 0



def form_submit_button_callback(key: str, order: int) -> None:
    logger.info(f"Callback key: {key=}")
    if not key in memory:
        # The submit button has been pressed
        memory[key] = Record(1, order)
    else:
        memory[key].order += 1
        memory[key].last += 1
    logger.debug(f"memory[{key}]={memory[key]}")


def get_forms(label: str, val: float, key: str, cb: str, order: int) -> Tuple[bool, float, float]:
    with st.form(label):
        val11 = st.slider("val11", 0.0, 1.0, val, key=key)
        val21 = st.number_input("val1_")
        submit = st.form_submit_button(
            "submit", on_click=form_submit_button_callback, args=(cb, order)
        )
        return submit, val11, val21




if __name__ == "__main__":
    logger.info(f"{'-':->50}")
    logger.info(f"{' Enter App ':^50}")
    logger.info(f"{'-':->50}")

    submit1, val11, val12 = get_forms("1", 0.2, "val11", "button1", 1)

    if submit1 or memory.get("button2"):  # The button2 has been pressed at least one time

        val11_compute, val12_compute = str(val11), str(val12)
        st.write(f"{val11_compute=}")
        st.write(f"{val12_compute=}")

        submit2, val21, val22 = get_forms("2", 0.4, "val21", "button2", 2)

        if submit2 or memory.get("button3"):

            if not submit1:
                val21_compute = "+".join([val11_compute, str(val21)])
                val22_compute = "+".join([val12_compute, str(val22)])
                st.write(f"{val21_compute=}")
                st.write(f"{val22_compute=}")

                submit3, val31, val32 = get_forms("3", 0.6, "val31", "button3", 3)

                if submit3:
                    val31_compute = "+".join([val21_compute, str(val31)])
                    val32_compute = "+".join([val22_compute, str(val32)])
                    st.write(f"{val31_compute=}")
                    st.write(f"{val32_compute=}")
    logger.info(f"{'-':->50}")
    logger.info(f"{' Done ':^50}")
    logger.info(f"{'-':->50}")
    logger.info(f"\n\n\n\n")

    # st.write(memory)