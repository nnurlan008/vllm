import os
import time
from datasets import load_dataset
from transformers import AutoConfig
import datasets as hfds
from datasets import load_dataset

from vllm import LLM, SamplingParams
import pdb


# os.environ['VLLM_USE_FLASHATTN'] = '0'
# supported attentions:
# ['FLASH_ATTN', 'TRITON_ATTN', 'XFORMERS', 'ROCM_ATTN', 
#  'ROCM_AITER_MLA', 'ROCM_AITER_FA', 'TORCH_SDPA', 'FLASHINFER',
#  'FLASHINFER_MLA', 'TRITON_MLA', 'CUTLASS_MLA', 'FLASHMLA', 
#  'FLASHMLA_SPARSE', 'FLASH_ATTN_MLA', 'PALLAS', 'IPEX', 
#  'NO_ATTENTION', 'FLEX_ATTENTION', 'TREE_ATTN', 'ROCM_AITER_UNIFIED_ATTN', 'CPU_ATTN', 'CUSTOM']
os.environ['VLLM_ATTENTION_BACKEND'] = 'TRITON_ATTN'  # or 'TORCH_SDPA'
# os.environ['VLLM_TORCH_COMPILE_LEVEL'] = '0'
max_tokens = 64*1024

dataset = load_dataset('THUDM/LongBench-v2', split='train')
print("dataset: ", dataset)

index_shorts = [i for i in range(len(dataset)) if dataset[i]['length'] == 'short']

# sort indices by context length
sorted_idxs = sorted(index_shorts, key=lambda i: len(dataset[i]['context']))
min_index = sorted_idxs[1]

index = min_index # index_shorts[0]
sample = dataset[min_index]
print("len(sample['context']):", len(sample['context']))
text = f"""
Please read the following text and answer the question below.

{sample['context']}

What is the correct answer to this question: {sample['question']}
Choices:
(A) {sample['choice_A']}
(B) {sample['choice_B']}
(C) {sample['choice_C']}
(D) {sample['choice_D']}

Choose the answer from the choices above. Also explain your answer with a detailed summary of 10000 words."""

#  repetition_penalty=1.2,
#  min_tokens=10,  # Add this - force at least 10 tokens
#  stop=[],  # Ensure no stop tokens
#  min_tokens=1*1024,
sampling_params = SamplingParams(temperature=0.15, top_p=0.95,
                                 repetition_penalty=1.2,
                                 max_tokens=max_tokens)

prompts = [
    text
    # 'The future of AI is'
]

folder = '/home/nnaza008/vllm_results/attention_weights_all_layers/Meta-Llama-3-8B-Instruct/longBench/'
os.makedirs(folder, exist_ok=True)
file_name = folder + 'test.pt'

model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
cfg = AutoConfig.from_pretrained(model_name)
print("max_position_embeddings:", getattr(cfg, "max_position_embeddings", None))
print("model_max_length:", getattr(cfg, "model_max_length", None))
print("other keys:", {k: v for k, v in cfg.to_diff_dict().items() if "max" in k or "pos" in k})

def main():
    # Create an LLM.
    llm = LLM(model=model_name, enforce_eager=True)  # Disable CUDA graphs!
    outputs = llm.generate(prompts, sampling_params, log_file_name=None)
    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)


if __name__ == "__main__":
    print("")
    main()