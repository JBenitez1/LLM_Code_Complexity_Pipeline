#!/usr/bin/env bash

set -euo pipefail

REPOS=(
    "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
    "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF"
    "bartowski/Ministral-8B-Instruct-2410-GGUF"
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    "bartowski/Mistral-7B-Instruct-v0.3-GGUF"
    "bartowski/codegemma-1.1-7b-it-GGUF"
    "TheBloke/CodeLlama-7B-Instruct-GGUF"
)

MODELS=(
    "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    "DeepSeek-Coder-V2-Lite-Instruct-IQ2_M.gguf"
    "Ministral-8B-Instruct-2410-Q4_K_M.gguf"
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
    "codegemma-1.1-7b-it-Q4_K_M.gguf"
    "codellama-7b-instruct.Q4_K_M.gguf"
)

mkdir -p models

for i in "${!REPOS[@]}"; do
    if [ ! -f "models/${MODELS[$i]}" ]; then
        python3 download_gguf.py --repo-id "${REPOS[$i]}" --filename "${MODELS[$i]}" --local-dir models
    fi
done
