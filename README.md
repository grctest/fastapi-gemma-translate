# fastapi-gemma-translate

This project provides a robust REST API built with FastAPI and Docker to manage and interact with Google's Gemma Translate AI Models for AI string translations on-device.

## Key Features

*   **Translation services**
*   **Support for multiple models**
*   **Automatic API Docs**: Interactive API documentation powered by Swagger UI and ReDoc.

## Technology Stack

*   [FastAPI](https://github.com/fastapi/fastapi) for the core web framework.
*   [Uvicorn](https://www.uvicorn.org/) as the ASGI server.
*   [Docker](https://www.docker.com/) for containerization and easy deployment.
*   [Pydantic](https://docs.pydantic.dev/) for data validation and settings management.

---

## Getting Started

### Prerequisites

*   [Docker Desktop](https://www.docker.com/products/docker-desktop/)
*   [Conda](https://www.anaconda.com/download) (or another Python environment manager)
*   Python 3.10+

### 1. Set Up the Python Environment

Create and activate a Conda environment:
```bash
conda create -n translate python=3.11
conda activate translate
```

Install the hf tool to download the models:
```
pip install "fastapi[standard]" "uvicorn[standard]" httpx llama-cpp-python huggingface_hub
```

We're going to fetch the GGUF model fromats from these repositories:
```
https://huggingface.co/bullerwins/translategemma-27b-it-GGUF
https://huggingface.co/bullerwins/translategemma-12b-it-GGUF
https://huggingface.co/bullerwins/translategemma-4b-it-GGUF
```

For comparison, here's a table of all above GGUF models and their quantitizations:

| Model Family | Quantization Level | File Name | File Size | Quality Level |
|---|---|---|---|---|
| Gemma 4B | Q3_K_L | translategemma-4b-it-Q3_K_L.gguf | 2.24 GB | Low-Medium |
|  | Q4_K_S | translategemma-4b-it-Q4_K_S.gguf | 2.38 GB | Medium |
|  | Q4_K_M | translategemma-4b-it-Q4_K_M.gguf | 2.49 GB | Balanced |
|  | Q5_K_S | translategemma-4b-it-Q5_K_S.gguf | 2.76 GB | High |
|  | Q5_K_M | translategemma-4b-it-Q5_K_M.gguf | 2.83 GB | High+ |
|  | Q6_K | translategemma-4b-it-Q6_K.gguf | 3.19 GB | Near Lossless |
|  | Q8_0 | translategemma-4b-it-Q8_0.gguf | 4.13 GB | Reference |
| Gemma 12B | Q3_K_L | translategemma-12b-it-Q3_K_L.gguf | 6.48 GB | Medium |
|  | Q4_K_S | translategemma-12b-it-Q4_K_S.gguf | 6.94 GB | High |
|  | Q4_K_M | translategemma-12b-it-Q4_K_M.gguf | 7.30 GB | High (Recommended) |
|  | Q5_K_S | translategemma-12b-it-Q5_K_S.gguf | 8.23 GB | Very High |
|  | Q5_K_M | translategemma-12b-it-Q5_K_M.gguf | 8.45 GB | Very High |
|  | Q6_K | translategemma-12b-it-Q6_K.gguf | 9.66 GB | Near Lossless |
|  | Q8_0 | translategemma-12b-it-Q8_0.gguf | 12.5 GB | Reference |
| Gemma 27B | Q3_K_L | translategemma-27b-it-Q3_K_L.gguf | 14.5 GB | High |
|  | Q4_K_S | translategemma-27b-it-Q4_K_S.gguf | 15.7 GB | Very High |
|  | Q4_K_M | translategemma-27b-it-Q4_K_M.gguf | 16.5 GB | Very High (Recommended) |
|  | Q5_K_S | translategemma-27b-it-Q5_K_S.gguf | 18.8 GB | Excellent |
|  | Q5_K_M | translategemma-27b-it-Q5_K_M.gguf | 19.3 GB | Excellent |
|  | Q6_K | translategemma-27b-it-Q6_K.gguf | 22.2 GB | Near Lossless |
|  | Q8_0 | translategemma-27b-it-Q8_0.gguf | 28.7 GB | Reference |

Download one of the following Gemma Translate models:

Gemma 4B:
```
hf download bullerwins/translategemma-4b-it-GGUF translategemma-4b-it-Q8_0.gguf --local-dir app/models/translategemma-4b-it-Q8_0
hf download bullerwins/translategemma-4b-it-GGUF translategemma-4b-it-Q6_K.gguf --local-dir app/models/translategemma-4b-it-Q6_K
hf download bullerwins/translategemma-4b-it-GGUF translategemma-4b-it-Q5_K_S.gguf --local-dir app/models/translategemma-4b-it-Q5_K_S
hf download bullerwins/translategemma-4b-it-GGUF translategemma-4b-it-Q5_K_M.gguf --local-dir app/models/translategemma-4b-it-Q5_K_M
hf download bullerwins/translategemma-4b-it-GGUF translategemma-4b-it-Q4_K_S.gguf --local-dir app/models/translategemma-4b-it-Q4_K_S
hf download bullerwins/translategemma-4b-it-GGUF translategemma-4b-it-Q4_K_M.gguf --local-dir app/models/translategemma-4b-it-Q4_K_M
hf download bullerwins/translategemma-4b-it-GGUF translategemma-4b-it-Q3_K_L.gguf --local-dir app/models/translategemma-4b-it-Q3_K_L
```

Gemma 12B:
```
hf download bullerwins/translategemma-12b-it-GGUF translategemma-12b-it-Q8_0.gguf --local-dir app/models/translategemma-12b-it-Q8_0
hf download bullerwins/translategemma-12b-it-GGUF translategemma-12b-it-Q6_K.gguf --local-dir app/models/translategemma-12b-it-Q6_K
hf download bullerwins/translategemma-12b-it-GGUF translategemma-12b-it-Q5_K_S.gguf --local-dir app/models/translategemma-12b-it-Q5_K_S
hf download bullerwins/translategemma-12b-it-GGUF translategemma-12b-it-Q5_K_M.gguf --local-dir app/models/translategemma-12b-it-Q5_K_M
hf download bullerwins/translategemma-12b-it-GGUF translategemma-12b-it-Q4_K_S.gguf --local-dir app/models/translategemma-12b-it-Q4_K_S
hf download bullerwins/translategemma-12b-it-GGUF translategemma-12b-it-Q4_K_M.gguf --local-dir app/models/translategemma-12b-it-Q4_K_M
hf download bullerwins/translategemma-12b-it-GGUF translategemma-12b-it-Q3_K_L.gguf --local-dir app/models/translategemma-12b-it-Q3_K_L
```

Gemma 27B:
```
hf download bullerwins/translategemma-27b-it-GGUF translategemma-27b-it-Q8_0.gguf --local-dir app/models/translategemma-27b-it-Q8_0
hf download bullerwins/translategemma-27b-it-GGUF translategemma-27b-it-Q6_K.gguf --local-dir app/models/translategemma-27b-it-Q6_K
hf download bullerwins/translategemma-27b-it-GGUF translategemma-27b-it-Q5_K_S.gguf --local-dir app/models/translategemma-27b-it-Q5_K_S
hf download bullerwins/translategemma-27b-it-GGUF translategemma-27b-it-Q5_K_M.gguf --local-dir app/models/translategemma-27b-it-Q5_K_M
hf download bullerwins/translategemma-27b-it-GGUF translategemma-27b-it-Q4_K_S.gguf --local-dir app/models/translategemma-27b-it-Q4_K_S
hf download bullerwins/translategemma-27b-it-GGUF translategemma-27b-it-Q4_K_M.gguf --local-dir app/models/translategemma-27b-it-Q4_K_M
hf download bullerwins/translategemma-27b-it-GGUF translategemma-27b-it-Q3_K_L.gguf --local-dir app/models/translategemma-27b-it-Q3_K_L
```

---

## Running the Application

### Using Docker (Recommended)

This is the easiest and recommended way to run the application.

1.  **Build the Docker image:**
    ```bash
    docker build -t fastapi_gemma_translate .
    ```

2.  **Run the Docker container:**
    This command runs the container in detached mode (`-d`) and maps port 8080 on your host to port 8080 in the container.
    ```bash
    docker run -d --name ai_container -p 127.0.0.1:8080:8080 -v C:/Users/username/Desktop/git/fastapi-gemma-translate/_models:/code/models fastapi_gemma_translate
    ```

---

Alternatively you can pull and run my docker image:

1. **Pull the Docker image:**
    ```bash
    docker image pull grctest/fastapi_gemma_translate
    ```

2. **Run the Docker container:**
    This command runs the container in detached mode (`-d`) and maps port 8080 on your host to port 8080 in the container.
    ```bash
    docker run -d --name ai_container -p 127.0.0.1:8080:8080 -v C:/Users/username/Desktop/git/fastapi-gemma-translate/_models:/code/models grctest/fastapi_gemma_translate
    ```

---

The above were for CPU-only mode, if you want Nvidia CUDA GPU support you'll need to use the CudaDockerfile, either by:

1. **Building the Docker image:**
   ```bash
   docker build -t fastapi_gemma_translate:legacy -f LegacyCudaDockerfile .
   docker build -t fastapi_gemma_translate:mainstream -f MainstreamCudaDockerfile .
   docker build -t fastapi_gemma_translate:future -f FutureCudaDockerfile .
   ```

   ```bash
   docker build -t grctest/fastapi_gemma_translate:legacy -f LegacyCudaDockerfile .
   docker build -t grctest/fastapi_gemma_translate:mainstream -f MainstreamCudaDockerfile .
   docker build -t grctest/fastapi_gemma_translate:future -f FutureCudaDockerfile .
   ```

2. **Pulling the Docker image:**
    ```bash
    docker image pull grctest/fastapi_gemma_translate:legacy
    ```

    Then you need to use the GPU flag:

    ```bash
    docker run --gpus all -d --name ai_container -p 127.0.0.1:8080:8080 -v C:/Users/username/Desktop/git/fastapi-gemma-translate/_models:/code/models fastapi_gemma_translate:legacy
    ```

    ```bash
    docker run --gpus all -d --name ai_container -p 127.0.0.1:8080:8080 -v C:/Users/username/Desktop/git/fastapi-gemma-translate/_models:/code/models grctest/fastapi_gemma_translate:legacy
    ```

Note: The `C:/Users/username/Desktop/git/fastapi-gemma-translate/_models` folder can be replaced by the path you've downloaded the gguf folders+files to.

### Local Development

For development, you can run the application directly with Uvicorn, which enables auto-reloading.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

---

## API Usage

Once the server is running, you can access the interactive API documentation:

*   **Swagger UI**: [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs)
*   **ReDoc**: [http://127.0.0.1:8080/redoc](http://127.0.0.1:8080/redoc)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.