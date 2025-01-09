# _*_ coding:utf-8 _*_
import argparse
import json
import logging
import os
import pprint
import sys
import time
import pandas as pd

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
# from sacrebleu import CHRF, BLEU

# from accelerate import Accelerator
# from accelerate.utils import gather_object



# 'ascii' codec can't encode characters in position 19-29: ordinal not in range(128)
import importlib
importlib.reload(sys)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import pdb

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

logger = logging.getLogger(__name__)

os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["RAY_USE_MULTIPROCESSING_CPU_COUNT"] = "1"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

from tqdm import tqdm
def read_jsonline(file_path):
    result = []
    with open(file_path, "r", encoding="utf-8") as f:
        for l in tqdm(f.readlines()):
            result.append(json.loads(l))
        return result
def write_jsonline(file_path,data,mode='w'):
    with open(file_path,mode=mode,encoding='utf-8')as f:
        for d in data:
            json.dump(d,f,ensure_ascii=False)
            f.write("\n")
        f.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_baseline", # 待评价的模型
        type=str,
        help="Path to baseline model",
    )
    parser.add_argument(
        "--file_path", # 答案
        type=str,
        help="Path to eval dataset",
    )
    parser.add_argument(
        "--output_path", # 预测结果输出路径
        type=str,
        help="Path to baseline model",
    )
    parser.add_argument(
        "--use_vllm", # 预测结果输出路径
        type=bool,
        help="whether using vllm to inference",
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        default="",
        type=str,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Specify num of return sequences",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Specify num of return sequences",
    )
    parser.add_argument(
        "--language", type=str, default="Chinese", choices=["English", "Chinese"]
    )
    parser.add_argument("--eos", type=str, default="</s>")

    args = parser.parse_args()

    return args

def prompt_format(data_list:list,tokenizer:AutoTokenizer):
    sources = []

    for data in tqdm(data_list,desc="data_list"):
        message = [
            # {"role":"system","content":"you are a helpful assistant"},
            {"role":"user","content":data['input']}
        ]
        sources.append({"prompt_input":tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True)})
    return sources, data_list

def generate(
    model,
    tokenizer,
    inputs,
):
    # by default, stop_token_id = tokenizer.eos_token_id
    
    generate_ids = model.generate(
        inputs.input_ids, attention_mask=inputs.attention_mask,
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.eos_token_id,
        temperature=0.4,
        do_sample=True,
        max_new_tokens=1024,
        top_k=40,
        top_p=0.85,
        repetition_penalty=1.05, 
    )
    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return result
    


def evaluate(data,args, model, tokenizer, device, file_path, output_path, batch_size=8):
    

    data_input,data_source = prompt_format(data,tokenizer=tokenizer)
    results, reference, inputs= [], [], []
    total_time = 0
    num_batch = len(data_input) // batch_size if len(data_input) % batch_size == 0 else len(data_input) // batch_size + 1



    for i in tqdm(range(num_batch)):
        batch_prompts = data_input[i * batch_size : (i + 1) * batch_size]


        batch_inputs = [t['prompt_input'] for t in batch_prompts] 


        inputs += batch_inputs
        
        model_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True).to(device)

    


        cur_start_time = time.time()
        outputs = generate(
            model,
            tokenizer,
            model_inputs,
        )
        cur_end_time = time.time()
        total_time += cur_end_time-cur_start_time
        
       
        # outputs_.extend([i for i in outputs])
        

        batch_outputs= []
        for r in outputs:

            results.append(r.split("assistant")[-1].strip())#yi 8*6b模板
           

        for j in range(i * batch_size, (i + 1) * batch_size):
            if j < len(data_input):
                data_source[j]['model_output']=results[j]
                batch_outputs.append(results[j])
            else:
                break
        print(f"\n\nbatch inputs:\n{batch_inputs}\n\nbatch output:{batch_outputs}\n\n")
        
        
    averge_time = total_time/len(data_input)
    print(f"total time:{total_time},averge time:{averge_time}, example numbers:{len(data_input)}")
    

        
    try:
        with open(output_path+"/_output.jsonl", "w",encoding='utf-8') as f:
            f.write(json.dumps({"total time":total_time,"averge time":averge_time,"example numbers":len(data_input),
                                "temperature":0.1,
                                "do_sample":True,
                                "max_new_tokens":250,
                                "top_k":40,
                                "top_p":0.85,
                                "repetition_penalty":1.05,
                                # "metric":sacrebleu_metric_result,
                                },ensure_ascii=False)+"\n")
            for i in data_source:
                f.write(json.dumps(i, ensure_ascii=False) + "\n")
            f.close()
    except Exception as e:
        print(e)
    
def vllm_generate(model_path,data,batch_size,output_path,tokenizer):


    data_input,data_source = prompt_format(data,tokenizer)

    from vllm import LLM,SamplingParams

    sampling_params = SamplingParams(
        stop_token_ids = tokenizer.eos_token,
        max_tokens=1024,
        temperature=0.4,
        top_k=40,
        top_p=0.85,
        repetition_penalty=1.05,
        )
    
    llm = LLM(model_path,tokenizer=model_path,tensor_parallel_size=2,gpu_memory_utilization=0.9,max_model_len=4096,dtype=torch.bfloat16,
              enforce_eager=True
              )

    results, reference, inputs= [], [], []
    
    total_time = 0
    num_batch = len(data_input) // batch_size if len(data_input) % batch_size == 0 else len(data_input) // batch_size + 1


    outputs_= []
    for i in tqdm(range(num_batch)):
        batch_prompts = data_input[i * batch_size : (i + 1) * batch_size]
        batch_inputs = [t['prompt_input'] for t in batch_prompts] 


        cur_start_time = time.time()
        outputs = llm.generate(batch_inputs,sampling_params)
        cur_end_time = time.time()
        total_time += cur_end_time-cur_start_time
        
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
       
        results.extend([i.outputs[0].text for i in outputs])#生成的文本
        

        for j in range(i * batch_size, (i + 1) * batch_size):
            if j < len(data_input):
                data_source[j]['model_output']=results[j]
                print(data_source[j]['model_output'])
            else:
                break
        
    averge_time = total_time/len(data_input)
    print(f"total time:{total_time},averge time:{averge_time}, example numbers:{len(data_input)}")
    
    
   
    try:
   
        with open(output_path+"/_output.jsonl", "w",encoding='utf-8') as f:
            f.write(json.dumps({"total time":total_time,"averge time":averge_time,"example numbers":len(data_source),
                                "temperature":0.4,
                                "do_sample":True,
                                "max_new_tokens":250,
                                "top_k":40,
                                "top_p":0.85,
                                "repetition_penalty":1.05,
                                # "metric":sacrebleu_metric_result,
                                },ensure_ascii=False)+"\n")
            for i in data_source:
                f.write(json.dumps(i, ensure_ascii=False) + "\n")
            f.close()
        
    except Exception as e:
        print(e)




    

def main():
    import pandas
    args = parse_args()
    print(f"args:\n{args}")

    device = torch.device("cuda:0")
    
    print(args.use_vllm)
    print("load data")

    data = json.load(open(args.file_path))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path_baseline,trust_remote_code=True,padding_side="left")
    if args.use_vllm:
        vllm_generate(args.model_name_or_path_baseline,data=data,output_path=args.output_path,batch_size=128,tokenizer=tokenizer)

    else:
        

        print("load model ing...")
        model_baseline = AutoModelForCausalLM.from_pretrained(args.model_name_or_path_baseline,device_map="auto",trust_remote_code=True,torch_dtype=torch.bfloat16)
        # model_baseline = None
        model_baseline.eval()
    
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id


        evaluate(data,args, model_baseline,tokenizer, device, args.file_path, args.output_path,batch_size=2)



if __name__ == "__main__":
    main()