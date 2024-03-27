"""
Usage:
    Lexical_overlap_rate.py --tpf=<TestPosFile> --tnf=<TestNegFile> [--dataset_arg=<d>] [--dataset=<d>]
                   
Options:
    -h --help       Show this screen.
    --dataset_arg=<d>   wec - which dataset to generate for [default: wec]
    --dataset=<d>   wec-eng/wec-zh - which dataset to load [default: wec-eng]
"""

""" 
the experiment explored the influence of lexical overlap rate.
We use the bleu 1 metric to calculate the lexical overlap rate.
"""
import os,sys
root_path = os.path.abspath(__file__)

root_path = '/'.join(root_path.split('/')[:-2]) #windows 用\
sys.path.append(root_path)
from src.dataobjs.dataset import DataSet, Split
from datetime import datetime
from tqdm import tqdm,trange
from docopt import docopt
from src.utils.log_utils import create_logger_with_fh
from src.utils.io_utils import create_and_get_path

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics import jaccard_distance
from nltk import FreqDist

import logging
import pickle

logger = logging.getLogger(__name__)
def load_data():
    test_dataset = DataSet.get_dataset(_dataset_arg, split=Split.Dev)
    test_feat = test_dataset.load_pos_neg_pickle(_test_pos_file, _test_neg_file)
    calculate(test_feat)

def calculate(test_feat,batch_size=10000):
    overlap_0_10 =list()
    overlap_10_30 =list()
    overlap_30_50 =list()
    overlap_50_100 =list()
    for mention_pair in tqdm(test_feat,desc="test pairs"):
        #，.这些标点几乎每个文档都有，可以不用删除，但数量又会影响，删除吧
        doc1 = []
        doc2 = []
        for word in mention_pair[0].mention_context:
            if word.isalpha():doc1.append(word.lower())
        for word in mention_pair[1].mention_context:
            if word.isalpha():doc2.append(word.lower())
        overlap_rate = calculate_whole_doc_bleu(doc1,doc2)
        # overlap_rate = calculate_overlap(doc1,doc2)    
        if overlap_rate < 0.1:
            overlap_0_10.append(mention_pair)
        elif overlap_rate < 0.3:
            overlap_10_30.append(mention_pair)
        elif overlap_rate < 0.5:
            overlap_30_50.append(mention_pair)
        else:
            overlap_50_100.append(mention_pair)
    logger.info(f"overlap 0-20 pairs:{len(overlap_0_10)}, overlap 20-40 pairs:{len(overlap_10_30)}, overlap 40-60 pairs:{len(overlap_30_50)}, overlap 60-100 pairs:{len(overlap_50_100)}")
    pickle.dump(overlap_0_10,open("../experiments/lexical_overlap/overlap_0_10.pickle","w+b"))
    pickle.dump(overlap_10_30,open("../experiments/lexical_overlap/overlap_10_30.pickle","w+b"))
    pickle.dump(overlap_30_50,open("../experiments/lexical_overlap/overlap_30_50.pickle","w+b"))
    pickle.dump(overlap_50_100,open("../experiments/lexical_overlap/overlap_50_100.pickle","w+b"))

def calculate_whole_doc_bleu(doc1, doc2):
    # words_doc1 = doc1.split()
    # words_doc2 = doc2.split()
    words_doc1 = doc1
    words_doc2 = doc2
    bleu = sentence_bleu([words_doc1], words_doc2, weights=(1, 0, 0, 0))
    return bleu



# bleu_score = calculate_whole_doc_bleu(doc1, doc2)
# print(f"Whole Document BLEU-1 Score: {bleu_score}")

def calculate_overlap(doc1, doc2):
    words_doc1 = set(doc1)
    words_doc2 = set(doc2)
    
    common_words = len(words_doc1.intersection(words_doc2))
    total_words = (len(words_doc1) + len(words_doc2)) / 2
    
    overlap_ratio = common_words / total_words
    return overlap_ratio



def jaccard_similarity(doc1, doc2):
    words_doc1 = set(doc1)
    words_doc2 = set(doc2)
    return 1 - jaccard_distance(FreqDist(words_doc1), FreqDist(words_doc2))


# similarity = jaccard_similarity(doc1.split(), doc2.split())
# print(f"Jaccard Similarity: {similarity}")

def test():
    doc1 = ["this",'is',"a","cat"]
    doc2 = ["this","is","not","a","cat"]
    overlap_rate1 = calculate_whole_doc_bleu(doc1,doc2)
    overlap_rate2 = calculate_overlap(doc1,doc2)    
    
    print(overlap_rate1,overlap_rate2)



if __name__ == '__main__':

    os.chdir(sys.path[0])
    argv = ['--tpf','../datasets/WEC-Eng/final_processed/pair/Test_Event_gold_mentions_validated_PosPairs.pickle',
            '--tnf','../datasets/WEC-Eng/final_processed/pair/Test_Event_gold_mentions_validated_NegPairs.pickle',
            '--dataset','wec'
            ]
    _arguments = docopt(__doc__, argv=argv, help=True, version=None, options_first=False)
    start_time = datetime.now()
    dt_string = start_time.strftime("%Y-%m-%d_%H-%M")
    
    _dataset_arg = _arguments.get("--dataset_arg")
    _dataset = _arguments.get("--dataset")
    _test_pos_file = _arguments.get("--tpf")
    _test_neg_file = _arguments.get("--tnf")

    _output_folder = create_and_get_path("checkpoints")
    log_params_str = dt_string+"_ds_" + _dataset_arg + "lexical_overlap_rate"
    create_logger_with_fh(_output_folder + "/depth_experiment_" + log_params_str)

    logger.info(f"experiment of lexical overlap on {_dataset}")

    load_data()
    # test()
    