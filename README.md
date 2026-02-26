# LLM_Code_Complexity_Pipeline

Determining the capability of LLM to estimate the time complexity of code

## Requirements:

- vllm
- torch
- numpy
- tqdm
- datasets
- sklearn

## Run Instructions:

1. Create an anaconda environment using Python 3.11

   ```bash
   conda create -n vllm311 python=3.11 -y
   conda activate vllm311
   ```
2. Download appropriate torch, torchvision, and torchaudio builds based on CUDA runtime
3. Install dependencies (Would reccomend installation is done in the following order)

   ```bash
   pip install vllm
   pip install flash-attn --no-build-isolation # Skip if download fails
   pip install requests datasets scikit-learn tqdm numpy
   ```
4. Clone dataset repo

   ```bash
   git clone https://github.com/sybaik1/CodeComplex-Data.git
   ```
5. Verify the following paths

   ```bash
   {WORKING_DIR}/CodeComplex-Data/java.jsonl
   {WORKING_DIR}/CodeComplex-Data/java.jsonl
   ```
6. Start vLLM server

   ```bash
   export PYTORCH_ALLOC_CONF=expandable_segments:True

   vllm serve {model-name} \
   	--trust-remote-code \
   	--max-model-len 4096 \
   	--gpu-memory-utilization 0.9 \
   	--enable-prefix-caching
   ```
7. On separate terminal run model_testing.py

   ```bash
   python model_testing.py \
   	--model {model-name}
   ```

### Summary (Just run this):

#### Terminal 1

```bash
conda create -n vllm311 python=3.11 -y
conda activate vllm311

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install vllm
pip install requests datasets scikit-learn tqdm numpy

git clone https://github.com/sybaik1/CodeComplex-Data.git
```

```bash
conda activate vllm311
export PYTORCH_ALLOC_CONF=expandable_segments:True
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching
```

#### Terminal 2

```bash
conda activate vllm311
python bench_vllm_singlefile.py --model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
```
