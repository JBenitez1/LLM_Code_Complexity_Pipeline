# LLM_Code_Complexity_Pipeline

Benchmark local LLMs on the CodeComplex dataset for time-complexity classification.

The current workflow uses:
- `llama.cpp` through `llama-cpp-python`
- local GGUF model files
- `model_testing.py` for benchmarking
- `download_gguf.py` for downloading GGUF files from Hugging Face

## Files

- `model_testing.py`: runs the benchmark and writes per-example outputs and aggregate stats
- `download_gguf.py`: downloads GGUF files from Hugging Face repos
- `quantize_awq.py`: AWQ quantization helper for a separate workflow
- `vllm_test.py`: older vLLM test script

## Requirements

- Linux
- Anaconda or Miniconda
- NVIDIA GPU with a working driver if you want GPU inference
- CUDA toolkit installed locally for the source build path used here
- Python packages:
  - `llama-cpp-python`
  - `numpy`
  - `tqdm`
  - `datasets`
  - `scikit-learn`
  - `huggingface_hub`

## Dataset

Clone the dataset repo next to this project so these files exist:

```bash
git clone https://github.com/sybaik1/CodeComplex-Data.git
```

Expected paths:

```bash
CodeComplex-Data/java_data.jsonl
CodeComplex-Data/python_data.jsonl
```

## Create the Conda Environment

This project was set up with a CUDA-enabled `llama-cpp-python` build in a dedicated conda environment.

```bash
CONDA_NO_PLUGINS=true conda create --solver classic -n llamacpp-gpu python=3.12 cmake ninja pip -y
```

Install the runtime and benchmark dependencies:

```bash
CONDA_NO_PLUGINS=true conda run -n llamacpp-gpu env CMAKE_ARGS=-DGGML_CUDA=on FORCE_CMAKE=1 pip install --no-cache-dir llama-cpp-python numpy tqdm datasets scikit-learn huggingface_hub
```

Activate the environment:

```bash
conda activate llamacpp-gpu
```

## Verify GPU Support

Check that the build supports GPU offload:

```bash
CONDA_NO_PLUGINS=true conda run -n llamacpp-gpu python -c "import llama_cpp; from llama_cpp import llama_cpp as lib; print(lib.llama_supports_gpu_offload())"
```

If this prints `True`, the installed build supports CUDA offload.

## Download a GGUF Model

List GGUF files in a Hugging Face repo:

```bash
python3 download_gguf.py --repo-id bartowski/Qwen2.5-Coder-7B-Instruct-GGUF --list-files
```

Download one GGUF file by pattern:

```bash
python3 download_gguf.py --repo-id bartowski/Qwen2.5-Coder-7B-Instruct-GGUF --filename "*Q4_K_M.gguf"
```

Files are downloaded into `models/` by default.

## Run the Benchmark

Run inference with GPU offload enabled:

```bash
conda activate llamacpp-gpu
python model_testing.py --model /path/to/model.gguf --n-gpu-layers -1 --verbose
```

Example with a smaller subset:

```bash
conda activate llamacpp-gpu
python model_testing.py --model /path/to/model.gguf --limit 50 --n-gpu-layers -1 --verbose
```

Useful arguments:

- `--model`: path to a local GGUF file
- `--limit`: limit examples per split
- `--n-gpu-layers`: number of layers to offload to GPU, `-1` means all possible layers
- `--n-ctx`: context size
- `--n-batch`: batch size
- `--threads`: CPU thread count
- `--temperature`: generation temperature
- `--max-tokens`: max completion tokens

At startup, `model_testing.py` prints the detected `llama_cpp` module, version, and whether the build supports GPU offload.

## Outputs

The benchmark writes two CSV files to `bench_out_llamacpp/` by default:

- `outputs_<model>.csv`: one row per example
- `stats_<model>.csv`: summary metrics, classification report rows, and confusion matrix rows

Tracked per-example fields include:

- `true_label`
- `pred_label`
- `prompt_tokens`
- `gen_tokens`
- `total_tokens`
- `latency_s`
- `response_text`

Tracked aggregate fields include:

- `accuracy`
- `f1_macro`
- `prompt_tokens_total`
- `gen_tokens_total`
- `elapsed_s_total`
- throughput estimates

## Recommended Starting Models

For limited hardware, a good starting point is a code-focused GGUF model such as:

- `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF`
- `Qwen/Qwen2.5-Coder-3B-Instruct-GGUF`

Use a `Q4_K_M` quantization as the default starting point unless you have a reason to trade more memory for quality.

## Notes

- `model_testing.py` now uses local `llama.cpp` inference, not a vLLM server.
- If GPU offload is requested and CUDA is not healthy, the script fails fast instead of silently falling back.
- If `nvidia-smi` is broken, fix the driver/runtime stack before expecting GPU inference to work.
