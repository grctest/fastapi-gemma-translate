FROM python:3.14

WORKDIR /code

COPY ./app /code

RUN pip install "fastapi[standard]" "uvicorn[standard]" httpx llama-cpp-python

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]