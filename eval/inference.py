import argparse
import json
import logging
import os
import random
from datetime import datetime

import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaTokenizer,
    LlamaTokenizerFast,
)


random.seed(42)
logger = logging.getLogger(__name__)


def format_input_prompt(instruction, tokenizer, system=None):
    """
    This method must be consistent with encode_with_messages_format in train.py
    """
    prompt = ""

    if system is not None:
        message  = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
    else:
        message = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return prompt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--precision", 
        type=str, 
        default="bf16", 
        choices=["fp32", "fp16", "bf16"],
    )
    
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=1536,
        help="Max sequence length for the instruction.",
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=2048,
        help="Max sequence length for generating the response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling, 0.0 means greedy decoding",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.0,
        help="Top-p for sampling, 0.0 means greedy decoding",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for sampling, 1.0 means no penalty",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="if specified, use a subset of alpaca_eval",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help='If specified, only use the first "num_examples" examples in the dataset.',
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="If specified, overwrite the original output file (if exists).",
    )
    parser.add_argument(
        "--continue_output",
        action="store_true",
        help="If specified, continue writing to the original output file (if exists).",
    )
    args = parser.parse_args()

    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
    )

    logger.info("loading data and model...")
    # load some data
    eval_data = json.load(open(args.data, "r", encoding="utf-8"))
    eval_data = eval_data[ :len(eval_data) - len(eval_data) % 8]

    # select the specified subset
    if args.subset is not None:
        eval_data = [x for x in eval_data if x["dataset"] == args.subset]

    if args.num_examples is not None:
        eval_data = eval_data[: args.num_examples]

    logger.info(f"Total evaluation data: {len(eval_data)}")

    '''
    # prev_data = None
    if os.path.exists(args.output_path) or all([os.path.exists(args.output_path+f'.{i}') for i in range(8)]):
        # if args.continue_output:
        #     prev_data = json.load(open(args.output_path, "r", encoding="utf-8"))
        #     prev_data_ids = {x["idx"] for x in prev_data}
        #     logger.warning(
        #         f"Continue writing to {args.output_path}, which already has {len(prev_data)} examples..."
        #     )
        #     eval_data = [x for x in eval_data if x["idx"] not in prev_data_ids]
        # else:
        logger.warning("File %s already exists, exiting...", args.output_path)
        return
    '''
    
    my_outputs = []

    if args.precision == "fp32":
        precision = torch.float32
    elif args.precision == "fp16":
        precision = torch.float16
    elif args.precision == "bf16":
        precision = torch.bfloat16
    else:
        raise ValueError("Unknown precision %s", args.precision)

    if "polylm" in args.model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=precision, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, legacy=False, use_fast=False
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=precision,cache_dir=False
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=False,cache_dir=False)

    # add padding token if not already there (for Llama models)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    logger.info("model and data loaded!")
    logger.info("generating...")

    # generation_config = GenerationConfig.from_pretrained(
    #     args.model,
    #     max_length=args.max_output_length,
    #     top_p=args.top_p,
    #     temperature=args.temperature,
    #     do_sample=do_sample,
    #     repetition_penalty=args.repetition_penalty,
    # )
    logger.warning(
        f"[{datetime.now().strftime('%H:%M:%S')}] <GPU {accelerator.process_index}> Start generating..."
    )

    random.shuffle(eval_data)

    with accelerator.split_between_processes(eval_data) as eval_data_curr_process:

        dataloader = torch.utils.data.DataLoader(
            eval_data_curr_process, batch_size=args.batch_size, shuffle=False
        )

        with torch.inference_mode():
            for samples in tqdm(dataloader, desc=f"GPU {accelerator.process_index}"):

                input_texts = [
                    format_input_prompt(
                        samples["input"][j],
                        tokenizer,
                        system=samples["system"][j] if "system" in samples else None,
                    )
                    for j in range(len(samples["id"]))
                ]
                inputs = tokenizer(
                    input_texts,
                    return_tensors="pt",
                    max_length=args.max_input_length,
                    padding=True, 
                    truncation=True,
                )
                input_ids = inputs.input_ids.to(model.device)
                attention_mask = inputs.attention_mask.to(model.device)
                # try:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=GenerationConfig(
                        max_length=args.max_input_length,
                        max_new_tokens=args.max_output_length,
                        temperature=0.4, 
                        top_p=0.85,
                        top_k=40,
                        repetition_penalty=1.05,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    )
                )
                # except:
                #     logging.warning('CUDA out of memory, skip ...')
                #     continue

                for j in range(len(samples["id"])):
                    output = outputs[j]
                    output_string = tokenizer.decode(
                        output[input_ids.size(1) :], skip_special_tokens=True
                    )
                    my_outputs.append(
                        {
                            "id": samples["id"][j],
                            "generator": f"{args.model}",
                            "input": samples["input"][j],
                            "original_events": samples["original_events"][j],
                            "output": samples["output"][j],
                            "model_output": output_string.strip(),
                        }
                    )

        output_path_curr_process = args.output_path + f".{accelerator.process_index}"
        
        args.output_dir = args.output_path.rsplit('/', 1)[0]
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
            
        json.dump(
            my_outputs,
            open(output_path_curr_process, "w", encoding="utf8"),
            indent=4,
            ensure_ascii=False,
        )

    logger.warning(
        f"[{datetime.now().strftime('%H:%M:%S')}] <GPU {accelerator.process_index}> Finished generation!"
    )

    accelerator.wait_for_everyone()
    '''
    if accelerator.is_main_process:
        # concatenate outputs from all processes
        all_outputs = []
        for i in range(accelerator.num_processes):
            output_path_curr_process = args.output_path + f".{i}"
            all_outputs += json.load(
                open(output_path_curr_process, "r", encoding="utf-8")
            )
            os.remove(output_path_curr_process)

        if prev_data is not None:
            all_outputs += prev_data

        all_outputs = sorted(all_outputs, key=lambda x: x["idx"])
        json.dump(
            all_outputs,
            open(args.output_path, "w", encoding="utf8"),
            indent=4,
            ensure_ascii=False,
        )
        print(f"Saved {len(all_outputs)} examples to {args.output_path}.")
        logger.info(all_outputs[0])
        
        recall, em, f1_score, avg_lens = eval_question_answering(all_outputs)
        
        with open(args.output_path + '.metrics', "w") as metric_file:
            metric_file.write(
                f"---\nTest Set Results\n---\nRecall: {recall}\nEM: {em}\nF1: {f1_score}\nLens: {avg_lens}\n"
            )
    '''

if __name__ == "__main__":
    
    main()
