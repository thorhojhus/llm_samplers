# Setup llama.cpp server
1. Install the required packages
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp/
make -j$(nproc) GGML_CUDA=1 LLAMA_CURL=1
```

2. Run the server
```bash
llama-server --hf-repo "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" --hf-file Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf --port 8081 -ngl 128 -c 8192 -t 1
```
