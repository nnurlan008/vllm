from vllm import LLM, SamplingParams
import os
from datasets import load_dataset
import sys

os.environ['VLLM_ATTENTION_BACKEND'] = 'TRITON_ATTN'  # or 'TORCH_SDPA'

dataset = load_dataset("Crystalcareai/Code-feedback-sharegpt-renamed", split='train')
print("dataset: ", dataset)
print("dataset['id']:", dataset['id'])
print("dataset['messages'][0]:", dataset['messages'][0][0])
print("dataset['id'][0]:", dataset['id'][0])

model = "meta-llama/Llama-3.1-8B-Instruct"   # keep your model or change to an instruct-tuned model if available
STOP_TOKEN = "### END ###"

llm = LLM(model=model, enforce_eager=True)
folder = '/home/nnaza008/vllm_results/attention_weights_all_layers/Meta-Llama-3.1-8B_2/shareGPT/'
os.makedirs(folder, exist_ok=True)

f = open('shareGPT_prompt_results1.txt', 'w')

index_for_file = 0
for chat in dataset['messages']:
    # chat is a column of chats [human propmt and gpt response]
    file_name = 'chat_' + str(index_for_file) + '_maxtoken_' + str(1024) + '.pt' #2_context_1024.pt'
    file_path = folder + file_name 
    # llm = LLM(model=model, file_name=None)
    f.write("Chat " + str(index_for_file) + "\n")
    index_for_file = index_for_file + 1
    f.write("="*200 + "\n")
    prompt_num = 0
    for message in chat:
        
        # print(type(message))
        # print(message.keys())
        # print(message['role'])
        # sys.exit(-1)
        
        if message['role'] == 'human':
            text = message['value']
            print("prompt:", text)
            f.write("prompt " + str(prompt_num) + ": " + message['value'] + "\n")
            prompt_num = prompt_num + 1
            f.write("-"*200 + "\n")
            
            
            # Safer sampling params
            sampling_params = SamplingParams(
                temperature=0.15,        # low randomness for concise factual output
                top_p=0.95,
                # top_k=50,
                max_tokens=32*1024,          # HARD cap on generated tokens (reduce if you still see issues)
                repetition_penalty=1.15, # mild penalty; large values can destabilize generation
                stop=[STOP_TOKEN],       # explicit stop sequence
                # seed=42,                 # reproducible when possible
            )


            print("="*60)
            print("TEST 1: DIRECT REWRITE (Single Turn)")
            print("="*60)

            prompt1 = text

            outputs1 = llm.generate([prompt1], sampling_params, log_file_name=file_path)
            print("OUTPUT:")
            # print("Correct answer: ", sample['answer'])
            print(outputs1[0].outputs[0].text)
            # print("generated response:", "trial")
            f.write("generated response:" + outputs1[0].outputs[0].text + "\n")
            f.write("-"*200 + "\n")
            break 
        else:
            
            print("original response:", message['value'])
            f.write("original response: " + message['value'] + "\n")
            f.write("-"*200 + "\n") 
    llm.reset_importance()
    llm.llm_engine.block_manager.reset()
    # print(llm.llm_engine.get_importance())
    f.write("="*200 + "\n")
            
f.close()