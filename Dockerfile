# Use NVIDIA CUDA 12.3.1 devel image on Ubuntu 22.04
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    libcurl4-openssl-dev \
    curl \
    tesseract-ocr \
    libgl1 \

 && rm -rf /var/lib/apt/lists/*

# Create symbolic link for libcudart.so.11.0 (since some models are looking for CUDA 11)
RUN ln -s /usr/local/cuda/lib64/libcudart.so.12.0 /usr/local/cuda/lib64/libcudart.so.11.0

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1


ENV CMAKE_ARGS="-DGGML_CUDA=ON"
RUN  pip install pynvml pymupdf4llm langchain-community chromadb sentence-transformers tiktoken

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-deps --upgrade "git+https://github.com/JamePeng/llama-cpp-python.git@main"
# # Copy our persistent query script
# COPY run_model.py .

# Mounting the model directory is important so we don't bake the huge model into the image.
# EXPOSE port if you add a web API later

# CMD ["sh", "-c", "python3 ./ocr_bowtie/ocr_with_multi_img.py & python3 ./single_llm_with_api/rag_app_v2.py"]

# CMD ["python3", "./ocr_bowtie/ocr_with_img2table.py"]

# Uses Exact Search with Page number 
# CMD [ "python3", "./single_llm_with_api/rag_app_exact_search.py"] 

# Uses FIASS INDEX SEMANTIC Search 
# CMD [ "python3", "./single_llm_with_api/rag_app_faiss.py"]

# CMD [ "python3", "./single_llm_with_api/rag_app_with_multi_runs.py"]

# CMD [ "python3", "./single_llm_with_api/rag_sobol.py"]

# CMD [ "python3", "./single_llm_with_api/input_sobol.py"]

# CMD [ "python3", "./single_llm_with_api/detached_sobol.py"]

# CMD [ "python3", "./single_llm_with_api/stochastic_experiment.py"]

# CMD [ "python3", "./single_llm_with_api/model_selection.py"]

CMD [ "python3", "./dual_llm_with_api/app.py"]

# CMD [ "python3", "./AlarmDefinitions/generate_def.py"]

# CMD ["python3", "-m", "jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8889", "--no-browser", "--allow-root"]

