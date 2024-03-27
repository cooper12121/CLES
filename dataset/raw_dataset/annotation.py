#!/usr/bin/env python
 # -*- coding: utf-8 -*-
'''
Author: gao qiang
Date: 2024-01-19 11:07:17
LastEditTime: 2024-01-22 10:21:54
FilePath: /gaoqiang/EE/annotation.py
Description: 
Copyright (c) 2024 by ${gao qiang} email: ${gaoqiang_mx@163.com}, All Rights Reserved.
'''
# 使用 OmniEvent进行事件标注


from OmniEvent.OmniEvent.infer import get_model_tokenizer,get_result
import OmniEvent.OmniEvent
from process import *
from tqdm import tqdm
import multiprocessing


ed_model,ed_tokenizer,eae_model,eae_tokenizer,device = get_model_tokenizer()


def split():
    infile = "EE/WEC-Zh/train_event.json"
    file_events = load_jsonfile(infile)
    save_jsonfile(f"EE/CDEE/train1.json",file_events[:10000])
    save_jsonfile(f"EE/CDEE/train2.json",file_events[10000:20000])
    save_jsonfile(f"EE/CDEE/train3.json",file_events[20000:30000])
    save_jsonfile(f"EE/CDEE/train4.json",file_events[30000:40000])
    save_jsonfile(f"EE/CDEE/train5.json",file_events[40000:])


def annotation():
    # dev_file = "EE/WEC-Zh/dev_event_validated.json"
    # test_file = "EE/WEC-Zh/test_event_validated.json"
    # train_file = "EE/WEC-Zh/train_event.json"
    infile_list = ["EE/CDEE/train1.json","EE/CDEE/train2.json","EE/CDEE/train3.json","EE/CDEE/train4.json","EE/CDEE/train5.json"]
    for file_path in infile_list:
        file_events = load_jsonfile(file_path)
        save_name = file_path.split("/")[-1]
        new_dev_events = process_event(file_events)
        # save_jsonfile(f"EE/CDEE/dataset/{save_name}",new_dev_events)

def process_event(events):
    new_events = []
    try:
        for event in tqdm(events,desc ="events"):
            text = "".join(event["mention_context"])
            new_event = {}
            new_event["coref_chain"] = event["coref_chain"]
            new_event["coref_link"] = event["coref_link"]
            new_event["coref_chain"] = event["coref_chain"]
            new_event["doc_id"] = event["doc_id"]
            new_event["mention_Type"] = event["mention_Type"]
            new_event["tokens_str"] = event[ "tokens_str"]
            new_event["text"] = text
            results = get_result(text,ed_model,ed_tokenizer,eae_model,eae_tokenizer,device,task="EE")
            new_event["events"]=results[0]["events"]
            new_events.append(new_event)
    except Exception as e:
        print(e)
    return new_events

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    # annotation()
    # split()        
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=annotation)
        jobs.append(p)
        p.start()

    # 等待所有进程完成
    for j in jobs:
        j.join()
