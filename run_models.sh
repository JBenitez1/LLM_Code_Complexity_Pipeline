set -euo pipefail

MODEL_PATHS=(
    "/home/jbeni/Desktop/School/CAP6640/Course Project/LLM_Code_Complexity_Pipeline/models/codegemma-1.1-7b-it-Q4_K_M.gguf"
    "/home/jbeni/Desktop/School/CAP6640/Course Project/LLM_Code_Complexity_Pipeline/models/codellama-7b-instruct.Q4_K_M.gguf"
    "/home/jbeni/Desktop/School/CAP6640/Course Project/LLM_Code_Complexity_Pipeline/models/DeepSeek-Coder-V2-Lite-Instruct-IQ2_M.gguf"
    "/home/jbeni/Desktop/School/CAP6640/Course Project/LLM_Code_Complexity_Pipeline/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "/home/jbeni/Desktop/School/CAP6640/Course Project/LLM_Code_Complexity_Pipeline/models/Ministral-8B-Instruct-2410-Q4_K_M.gguf"
    "/home/jbeni/Desktop/School/CAP6640/Course Project/LLM_Code_Complexity_Pipeline/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
    "/home/jbeni/Desktop/School/CAP6640/Course Project/LLM_Code_Complexity_Pipeline/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    if [ -f "$MODEL_PATH" ]; then
        python3 model_testing.py --model "$MODEL_PATH"
    else
        echo "Model File Not Found: $MODEL_PATH"
    fi
done