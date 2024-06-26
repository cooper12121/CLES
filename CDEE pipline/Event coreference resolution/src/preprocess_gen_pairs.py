"""

Usage:
    preprocess_gen_pairs.py <File> --dataset=<dataset>
    preprocess_gen_pairs.py <File> --dataset=<dataset> [--split=<set>]
    preprocess_gen_pairs.py <File> --dataset=<dataset> [--split=<set>] [--ratio=<x>] [--topic=<type>]

Options:
    -h --help     Show this screen.
    --dataset=<dataset>   wec/ecb - which dataset to generate for [default: wec]
    --split=<set>    dev/test/train/na (split=na => doesnt matter) [default: na].
    --ratio=<x>  ratio of positive:negative, were negative is the controlled list (ratio=-1 => no ratio) [default: -1]
    --topic=<type>  subtopic/topic/corpus - take pairs only from the same sub-topic, topic or corpus wide [default: corpus]

"""

import os
import pickle
import random
from os import path
import sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2]) #windows 用\
sys.path.append(root_path)
from docopt import docopt

from src.dataobjs.dataset import Split, DataSet
from src.dataobjs.topics import Topics


def generate_pairs():
    positive_, negative_ = _dataset.get_pairwise_feat(_event_validation_file, to_topics=_topic_config)

    validate_pairs(positive_, negative_)

    basename = path.basename(path.splitext(_event_validation_file)[0])
    dirname = os.path.dirname(_event_validation_file)
    # positive_file = dirname + "/pair/" + basename + "_PosPairs.pickle"
    # negative_file = dirname + "/pair/" + basename + "_NegPairs.pickle"
    positive_file = dirname + "/pair/" + basename + "_PosPairs.pickle" #样本比例为-1，即默认
    negative_file = dirname + "/pair/" + basename + "_NegPairs.pickle"

    print("Positive Pairs=" + str(len(positive_)))
    pickle.dump(positive_, open(positive_file, "w+b"))

    print("Negative Pairs=" + str(len(negative_)))
    pickle.dump(negative_, open(negative_file, "w+b"))

    print("Done generating pairs for-" + _event_validation_file)
    print("Positive File created in-" + positive_file)
    print("Negative File created in-" + negative_file)


def validate_pairs(pos_pairs, neg_pairs):
    for men1, men2 in neg_pairs:
        if men1.coref_chain == men2.coref_chain:
            print("NEG BUG!!!!!!!!!!")

    for men1, men2 in pos_pairs:
        if men1.coref_chain != men2.coref_chain:
            print("POS BUG!!!!!!!!!!")

    print("Validation Passed!")


if __name__ == '__main__':
    os.chdir(sys.path[0])

    train_argv = ['../datasets/WEC-Eng/final_processed/Train_Event_gold_mentions.json','--dataset','wec','--split','train','--ratio',10]
    dev_argv = ['../datasets/WEC-Eng/final_processed/Dev_Event_gold_mentions_validated.json','--dataset','wec','--split','dev']
    test_argv = ['../datasets/WEC-Eng/final_processed/Test_Event_gold_mentions_validated.json','--dataset','wec','--split','test']
   
    for argv in [train_argv,dev_argv,test_argv]:
        arguments = docopt(__doc__, argv=argv, help=True, version=None, options_first=False)
        print(arguments)

        _event_validation_file = arguments.get("<File>")
        _ratio = int(arguments.get("--ratio"))
        _split_arg = arguments.get("--split").lower()
        if _split_arg in ["dev", "test"]:
            _split = Split.Dev
        elif _split_arg == "train":
            _split = Split.Train
        else:
            _split = Split.NA

        # subtopic/topic/corpus
        _topic_arg = arguments.get("--topic")
        _topic_config = Topics.get_topic_config(_topic_arg)

        random.seed(0)

        _dataset_arg = arguments.get("--dataset")
        _dataset = DataSet.get_dataset(_dataset_arg, ratio=_ratio, split=_split)

        if _dataset_arg == "wec" and _split == Split.Train and _ratio == -1:
            print("Selected WEC dataset for train with a -1 ratio will generate all possible negative pairs!!")

        print("Generating pairs for file-/" + _event_validation_file)
        generate_pairs()
        print("Process Done!")
