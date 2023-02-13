FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./portfolioqtopt /code/portfolioqtopt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV PYTHONPATH "${PYTHONPATH}:/code"

ENV TOKEN_API ""

CMD ["streamlit", "run", "portfolioqtopt/application/app.py", "--server.port=8501", "--server.address=0.0.0.0"]