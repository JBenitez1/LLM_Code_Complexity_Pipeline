# vllm_test.py
from vllm import LLM, SamplingParams

model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"

llm = LLM(model=model_id, trust_remote_code=True)
params = SamplingParams(temperature=0.0, max_tokens=32)

out = llm.generate(["Hello world"], params)
print(out[0].outputs[0].text)