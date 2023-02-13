FROM python:3.9

WORKDIR /code

ENV POETRY_VERSION 1.3.1

# COPY ./requirements.txt /code/requirements.txt
COPY ./pyproject.toml /code/pyproject.toml
COPY ./poetry.lock /code/poetry.lock

RUN pip install --no-cache-dir --upgrade pip

# https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
RUN pip install --no-cache-dir `poetry==$POETRY_VERSION`
RUN poetry config virtualenvs.create false
RUN poetry --version

# https://github.com/python-poetry/poetry/issues/3374

RUN --mount=type=cache,target=/home/.cache/pypoetry/cache \
    --mount=type=cache,target=/home/.cache/pypoetry/artifacts \
    poetry install --without test,docs,dev

COPY ./portfolioqtopt /code/portfolioqtopt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV PYTHONPATH "${PYTHONPATH}:/code"

ENV TOKEN_API ""

CMD ["streamlit", "run", "portfolioqtopt/application/app.py", "--server.port=8501", "--server.address=0.0.0.0"]