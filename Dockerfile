FROM python:3.14

WORKDIR /code

COPY ./app /code

# Install dependencies
#RUN apt-get update && apt-get install -y \
#    wget \
#    lsb-release \
#    gnupg \
#    cmake \
#    clang \
#    && bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" \
#    && apt-get clean \
#    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install "fastapi[standard]" "uvicorn[standard]" httpx llama-cpp-python

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]