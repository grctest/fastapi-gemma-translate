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
https://huggingface.co/mradermacher/translategemma-27b-it-GGUF
https://huggingface.co/mradermacher/translategemma-12b-it-GGUF
https://huggingface.co/mradermacher/translategemma-4b-it-GGUF
```

Download one of the following Gemma Translate models:

Gemma 4B:
```
hf download mradermacher/translategemma-4b-it-GGUF translategemma-4b-it.IQ4_XS.gguf --local-dir app/models/translategemma-4b-it.IQ4_XS
hf download mradermacher/translategemma-4b-it-GGUF translategemma-4b-it.Q2_K.gguf --local-dir app/models/translategemma-4b-it.Q2_K
hf download mradermacher/translategemma-4b-it-GGUF translategemma-4b-it.Q3_K_L.gguf --local-dir app/models/translategemma-4b-it.Q3_K_L
hf download mradermacher/translategemma-4b-it-GGUF translategemma-4b-it.Q3_K_M.gguf --local-dir app/models/translategemma-4b-it.Q3_K_M
hf download mradermacher/translategemma-4b-it-GGUF translategemma-4b-it.Q3_K_S.gguf --local-dir app/models/translategemma-4b-it.Q3_K_S
hf download mradermacher/translategemma-4b-it-GGUF translategemma-4b-it.Q4_K_M.gguf --local-dir app/models/translategemma-4b-it.Q4_K_M
hf download mradermacher/translategemma-4b-it-GGUF translategemma-4b-it.Q4_K_S.gguf --local-dir app/models/translategemma-4b-it.Q4_K_S
hf download mradermacher/translategemma-4b-it-GGUF translategemma-4b-it.Q5_K_M.gguf --local-dir app/models/translategemma-4b-it.Q5_K_M
hf download mradermacher/translategemma-4b-it-GGUF translategemma-4b-it.Q5_K_S.gguf --local-dir app/models/translategemma-4b-it.Q5_K_S
hf download mradermacher/translategemma-4b-it-GGUF translategemma-4b-it.Q6_K.gguf --local-dir app/models/translategemma-4b-it.Q6_K
hf download mradermacher/translategemma-4b-it-GGUF translategemma-4b-it.Q8_0.gguf --local-dir app/models/translategemma-4b-it.Q8_0
```

Gemma 12B:
```
hf download mradermacher/translategemma-12b-it-GGUF translategemma-12b-it.IQ4_XS.gguf --local-dir app/models/translategemma-12b-it.IQ4_XS
hf download mradermacher/translategemma-12b-it-GGUF translategemma-12b-it.Q2_K.gguf --local-dir app/models/translategemma-12b-it.Q2_K
hf download mradermacher/translategemma-12b-it-GGUF translategemma-12b-it.Q3_K_L.gguf --local-dir app/models/translategemma-12b-it.Q3_K_L
hf download mradermacher/translategemma-12b-it-GGUF translategemma-12b-it.Q3_K_M.gguf --local-dir app/models/translategemma-12b-it.Q3_K_M
hf download mradermacher/translategemma-12b-it-GGUF translategemma-12b-it.Q3_K_S.gguf --local-dir app/models/translategemma-12b-it.Q3_K_S
hf download mradermacher/translategemma-12b-it-GGUF translategemma-12b-it.Q4_K_M.gguf --local-dir app/models/translategemma-12b-it.Q4_K_M
hf download mradermacher/translategemma-12b-it-GGUF translategemma-12b-it.Q4_K_S.gguf --local-dir app/models/translategemma-12b-it.Q4_K_S
hf download mradermacher/translategemma-12b-it-GGUF translategemma-12b-it.Q5_K_M.gguf --local-dir app/models/translategemma-12b-it.Q5_K_M
hf download mradermacher/translategemma-12b-it-GGUF translategemma-12b-it.Q5_K_S.gguf --local-dir app/models/translategemma-12b-it.Q5_K_S
hf download mradermacher/translategemma-12b-it-GGUF translategemma-12b-it.Q6_K.gguf --local-dir app/models/translategemma-12b-it.Q6_K
hf download mradermacher/translategemma-12b-it-GGUF translategemma-12b-it.Q8_0.gguf --local-dir app/models/translategemma-12b-it.Q8_0
```

Gemma 27B:
```
hf download mradermacher/translategemma-27b-it-GGUF translategemma-27b-it.IQ4_XS.gguf --local-dir app/models/translategemma-27b-it.IQ4_XS
hf download mradermacher/translategemma-27b-it-GGUF translategemma-27b-it.Q2_K.gguf --local-dir app/models/translategemma-27b-it.Q2_K
hf download mradermacher/translategemma-27b-it-GGUF translategemma-27b-it.Q3_K_L.gguf --local-dir app/models/translategemma-27b-it.Q3_K_L
hf download mradermacher/translategemma-27b-it-GGUF translategemma-27b-it.Q3_K_M.gguf --local-dir app/models/translategemma-27b-it.Q3_K_M
hf download mradermacher/translategemma-27b-it-GGUF translategemma-27b-it.Q3_K_S.gguf --local-dir app/models/translategemma-27b-it.Q3_K_S
hf download mradermacher/translategemma-27b-it-GGUF translategemma-27b-it.Q4_K_M.gguf --local-dir app/models/translategemma-27b-it.Q4_K_M
hf download mradermacher/translategemma-27b-it-GGUF translategemma-27b-it.Q4_K_S.gguf --local-dir app/models/translategemma-27b-it.Q4_K_S
hf download mradermacher/translategemma-27b-it-GGUF translategemma-27b-it.Q5_K_M.gguf --local-dir app/models/translategemma-27b-it.Q5_K_M
hf download mradermacher/translategemma-27b-it-GGUF translategemma-27b-it.Q5_K_S.gguf --local-dir app/models/translategemma-27b-it.Q5_K_S
hf download mradermacher/translategemma-27b-it-GGUF translategemma-27b-it.Q6_K.gguf --local-dir app/models/translategemma-27b-it.Q6_K
hf download mradermacher/translategemma-27b-it-GGUF translategemma-27b-it.Q8_0.gguf --local-dir app/models/translategemma-27b-it.Q8_0
```

---

## Running the Application

### Using Docker (Recommended)

This is the easiest and recommended way to run the application.

1.  **Build the Docker image:**
    ```bash
    docker build -t fastapi_gemma_translate .
    docker build -t grctest/fastapi_gemma_translate .
    ```

2.  **Run the Docker container:**
    This command runs the container in detached mode (`-d`) and maps port 8080 on your host to port 8080 in the container.
    ```bash
    docker run -d --name ai_container_cpu -p 127.0.0.1:8080:8080 -v C:/Users/username/Desktop/git/fastapi-gemma-translate/_models:/code/models fastapi_gemma_translate
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
    docker run -d --name ai_container_cpu -p 127.0.0.1:8080:8080 -v C:/Users/username/Desktop/git/fastapi-gemma-translate/_models:/code/models grctest/fastapi_gemma_translate
    ```

---

The above were for CPU-only mode, if you want Nvidia CUDA GPU support you'll need to use the CudaDockerfile, either by:

1. **Building the Docker image:**
   ```bash
   docker build -t fastapi_gemma_translate_cuda:legacy -f LegacyCudaDockerfile .
   docker build -t fastapi_gemma_translate_cuda:mainstream -f MainstreamCudaDockerfile .
   docker build -t fastapi_gemma_translate_cuda:future -f FutureCudaDockerfile .
   ```

   ```bash
   docker build -t grctest/fastapi_gemma_translate_cuda:legacy -f LegacyCudaDockerfile .
   docker build -t grctest/fastapi_gemma_translate_cuda:mainstream -f MainstreamCudaDockerfile .
   docker build -t grctest/fastapi_gemma_translate_cuda:future -f FutureCudaDockerfile .
   ```

2. **Pulling the Docker image:**
    ```bash
    docker image pull grctest/fastapi_gemma_translate_cuda:legacy
    ```

    Then you need to use the GPU flag:

    ```bash
    docker run --gpus all -d --name ai_container_gpu -p 127.0.0.1:8080:8080 -v C:/Users/username/Desktop/git/fastapi-gemma-translate/_models:/code/models -e LLAMA_N_GPU_LAYERS=-1 fastapi_gemma_translate_cuda:legacy
    ```

    ```bash
    docker run --gpus all -d --name ai_container_cuda -p 127.0.0.1:8080:8080 -v C:/Users/username/Desktop/git/fastapi-gemma-translate/_models:/code/models -e LLAMA_N_GPU_LAYERS=-1 grctest/fastapi_gemma_translate_cuda:legacy
    ```

Note: The `C:/Users/username/Desktop/git/fastapi-gemma-translate/_models` folder can be replaced by the path you've downloaded the gguf folders+files to.

Container GPU variants:

* **Legacy**: Pascal / 10xx Nvidia cards

* **Mainstream**: Turing to Ada / 20xx, 30xx, 40xx, A100 Nvidia cards

* **Future**: Blackwell / 50xx Nvidia cards


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

### Model Lifecycle (Required)

Model loading is explicit. Translation endpoints will reject requests unless the requested model is already loaded.

1. Load a model:
```bash
curl -X POST "http://127.0.0.1:8080/model/load" \
    -H "Content-Type: application/json" \
    -d '{"model":"translategemma-4b-it-Q8_0"}'
```

Load a model with vision support (`mmproj` can be relative to the model folder or absolute):
```bash
curl -X POST "http://127.0.0.1:8080/model/load" \
    -H "Content-Type: application/json" \
    -d '{"model":"translategemma-4b-it-Q8_0","mmproj":"translategemma-4b-it.mmproj-f16.gguf"}'
```

2. Check model status:
```bash
curl "http://127.0.0.1:8080/model/status"
curl "http://127.0.0.1:8080/model/status?model=translategemma-4b-it-Q8_0"
```

`/model/status` includes:

* `loaded`: whether any model is loaded
* `loading`: whether a load is currently in progress
* `loaded_model`: currently loaded model name
* `vision_enabled`: whether the currently loaded model can process images

### Text Translation Endpoints

* `POST /translate` (stable locale list)
* `POST /experimental_translation` (stable + experimental locale list)

Both endpoints reject if:

* no model is loaded
* a model is still loading
* the requested `model` does not match the loaded model

### Image Translation Endpoints

This API supports image translation via multipart upload using `llama-cpp-python` vision chat formatting.

* `POST /translate_image` (stable locale list)
* `POST /experimental_translate_image` (stable + experimental locale list)

Notes:

* Upload images with `multipart/form-data` as field `file`.
* The image stays local to the server process and is sent to the model as a Base64 data URI.
* The loaded model must be vision-enabled.
* Vision is enabled only when `/model/load` is called with an `mmproj` value.
* Image translation requests are rejected if the currently loaded model was not loaded with `mmproj`.

Example (stable image route):
```bash
curl -X POST "http://127.0.0.1:8080/translate_image" \
    -F "file=@C:/path/to/image.jpg" \
    -F "model=translategemma-4b-it-Q8_0" \
    -F "source_lang_code=en" \
    -F "target_lang_code=es" \
    -F "max_new_tokens=200"
```

Example (experimental image route):
```bash
curl -X POST "http://127.0.0.1:8080/experimental_translate_image" \
    -F "file=@C:/path/to/image.jpg" \
    -F "model=translategemma-4b-it-Q8_0" \
    -F "source_lang_code=en" \
    -F "target_lang_code=ace" \
    -F "max_new_tokens=200"
```

---

## Real life usage

This FastAPI Gemma Translate Docker container code is used by [Metalglot](https://metalglot.com) software translation tool!

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
