"""
Usage:
    preprocess_embed.py <File> [<File2>] [<File3>]
    preprocess_embed.py <File> [<File2>] [<File3>] [--max=<x>]
    preprocess_embed.py <File> [<File2>] [<File3>] [--cuda=<y>]
    preprocess_embed.py <File> [<File2>] [<File3>] [--max=<x>] [--cuda=<y>]

Options:
    -h --help     Show this screen.
    --max=<x>   Maximum surrounding context [default: 250]
    --cuda=<y>  True/False - Whether to use cuda device or not [default: True].

"""

import multiprocessing
import pickle
import random
import time

import os,sys
from os import path
import torch
from docopt import docopt
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2]) 
sys.path.append(root_path)

from dataobjs.topics import Topics
from utils.embed_utils import EmbedTransformersGenerics



def extract_feature_dict(topics: Topics, embed_model):
    result_train = dict()
    topic_count = len(topics.topics_dict)
    for topic in topics.topics_dict.values():
        mention_count = len(topic.mentions)  #全部mention信息
        for mention in topic.mentions:
            start = time.time()
            # hidden, first_tok, last_tok, ment_size,arg0_hidden,arg0_size,arg1_span,arg1_size,loc_hidden,loc_size,time_hidden,time_size = embed_model.get_mention_full_rep(mention)#编码器编码#,h_mention,h_mention[0],h_mention[-1],mention.shape[0]也即mention的长度
            hidden, first_tok, last_tok, ment_size=embed_model.get_mention_full_rep(mention)
            #hidden:sentence_ids,
            end = time.time()  

            # result_train[mention.mention_id] = (hidden.cpu(), first_tok.cpu(), last_tok.cpu(), ment_size,arg0_hidden.cpu(),arg0_size,arg1_span.cpu(),arg1_size,loc_hidden.cpu(),loc_size,time_hidden.cpu(),time_size) #每一个mention其text对应的编码

            result_train[mention.mention_id] = (hidden.cpu(), first_tok.cpu(), last_tok.cpu(), ment_size)
            # print("To Go: Topics" + str(topic_count) + ", Mentions" + str(mention_count) + ", took-" + str((end - start)))
            mention_count -= 1
        topic_count -= 1

    return result_train


def worker(resource_file, max_surrounding_contx, use_cuda,finetune):
    embed_model = EmbedTransformersGenerics(max_surrounding_contx=max_surrounding_contx, use_cuda=use_cuda,finetune=finetune)
    name = multiprocessing.current_process().name
    print(name, "Starting")

    basename = path.basename(path.splitext(resource_file)[0]) #Dev_Event_gold_mentions #file name has no suffix
    dirname = os.path.dirname(resource_file)
    save_to = dirname + "/embed/" + basename + "_roberta_large.pickle"

    topics = Topics()
    topics.create_from_file(resource_file, keep_order=True) #对于wec，没有topic,topics={None:[所有的mention]}#此处是Mention类
    train_feat = extract_feature_dict(topics, embed_model)#字典{mentionid:（h_mention,h_mention[0],h_mention[-1], )}
    pickle.dump(train_feat, open(save_to, "w+b"))
    print("Done with -" + basename)


if __name__ == '__main__':
    os.chdir(sys.path[0])
    multiprocessing.set_start_method("spawn")
    argv = ['../datasets/WEC-Eng/final_processed/Dev_Event_gold_mentions_validated.json','../datasets/WEC-Eng/final_processed/Test_Event_gold_mentions_validated.json','../datasets/WEC-Eng/final_processed/Train_Event_gold_mentions.json','--cuda',True]
    arguments = docopt(__doc__, argv=argv, help=True, version=None, options_first=False)
    print(arguments)
    _file1 = arguments.get("<File>")
    _file2 = arguments.get("<File2>")
    _file3 = arguments.get("<File3>")
    _max_surrounding_contx = int(arguments.get("--max"))
    _use_cuda = True if arguments.get("--cuda")== True else False

    _all_files = list()
    if _file1:
        _all_files.append(_file1)
    if _file2:
        _all_files.append(_file2)
    if _file3:
        _all_files.append(_file3)

    torch.manual_seed(0)
    random.seed(0)
    if _use_cuda:
        torch.cuda.manual_seed(0)

    print("Processing files-" + str(_all_files))

    jobs = list()
    
    for _resource_file in _all_files:
        finetune=False
        # if(_resource_file=='../datasets/WEC-Eng/arguments/Train_Event_gold_mentions.json'):
        #     finetune=True
        job = multiprocessing.Process(target=worker, args=(_resource_file, _max_surrounding_contx, _use_cuda,finetune))
        jobs.append(job)
        job.start()

    for job in jobs:
        job.join()

    print("DONE!")
