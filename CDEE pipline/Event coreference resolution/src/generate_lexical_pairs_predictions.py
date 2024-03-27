"""
Usage:
    generate_lexical_pairs_predictions.py --tf=<TestPairFile> --tef=<TestEmbedFile> --ten=<TestEmbedNode> --mf=<ModelFile> --out=<OurPredFile> [--cuda=<y>] [--topic=<type>] [--em=<ExtractMethod>]

Options:
    -h --help                   Show this screen.
    --cuda=<y>                  True/False - Whether to use cuda device or not [default: True]
    --topic=<type>              subtopic/topic/corpus -  take pairs only from the same sub-topic, topic or corpus wide [default: subtopic]
    --em=<ExtractMethod>        pairwize/head_lemma/exact_string - model type to run [default: pairwize]
"""

from datetime import datetime
import logging
import ntpath
import pickle
from itertools import product
from collections import defaultdict

import numpy as np
import random
import torch
from docopt import docopt
import os,sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2]) #windows 用\
sys.path.append(root_path)

from src.coref_system.relation_extraction import HeadLemmaRelationExtractor, RelationTypeEnum
from src.coref_system.relation_extraction import RelationExtraction
from src.dataobjs.dataset import TopicConfig
from src.dataobjs.topics import Topics
from src.utils.embed_utils import EmbedFromFile
from src.dataobjs.dataset import DataSet, Split

from src.preprocess_edu_embed import EmbedNodeHidden
from src.train import accuracy_on_dataset
from src.utils.log_utils import create_logger_with_fh
from src.utils.io_utils import create_and_get_path

logger = logging.getLogger(__name__)
MAX_ALLOWED_BATCH_SIZE = 500


def generate_pred_matrix(inference_model,_pair_file):
    # all_pairs = list(product(topic.mentions, repeat=2))#两两组队
    all_pairs = pickle.load(open(_pair_file, "rb"))
    pairs_chunks = [all_pairs]
    if len(all_pairs) > MAX_ALLOWED_BATCH_SIZE:
        pairs_chunks = [all_pairs[i:i + MAX_ALLOWED_BATCH_SIZE] for i in
                        range(0, len(all_pairs), MAX_ALLOWED_BATCH_SIZE)]#划分为子列表  
    
    with torch.no_grad():
        acc, p, r,f1,all_labels, all_predictions =accuracy_on_dataset("Test", 0, inference_model, all_pairs)
        print(f"accuracy, precision, recall, f1:{acc,p,r,f1}")

   



def predict_and_save(_pair_file):
    all_predictions = list()
    # for topic in _event_topics.topics_dict.values():
    #     logger.info("Evaluating Topic No:" + str(topic.topic_id))
    all_predictions.append(generate_pred_matrix(_model,_pair_file))
    logger.info("Generating prediction file-" + _outfile)
    pickle.dump(all_predictions, open(_outfile, "wb"))


def get_pairwise_model():
    pairwize_model = torch.load(_model_file,map_location={'cuda:0':'cuda:1'})
    pairwize_model.set_embed_utils(EmbedFromFile([_embed_file]))

    # pairwize_model.set_embed_node(embed_node)

    device = torch.device("cuda:1" if torch.cuda.is_available()else "cpu")
    if _use_cuda:
        pairwize_model.cuda()
        # pairwize_model.to(device)

    pairwize_model.eval()
    return pairwize_model


if __name__ == '__main__':
    os.chdir(sys.path[0])

   

    argv=['--tf','../experiments/lexical_overlap/overlap_0_10.pickle',
          '--tef','../datasets/WEC-Eng/final_processed/embed/Test_Event_gold_mentions_validated_roberta_large.pickle',
          '--ten','../datasets/WEC-Eng/final_processed/gat_embed/Test_Event_gold_mentions_validated_roberta_large.pickle',
          '--mf','../model/wec_model.pickle',
          '--out','../experiments/lexical_overlap/pred_pairs/overlap_0_10_pred_pairs.pickle',
          ]
    _arguments = docopt(__doc__, argv=argv, help=True, version=None, options_first=False)
    print(_arguments)
    # _pair_file = _arguments.get("--tf")
    _embed_file = _arguments.get("--tef")
    _model_file = _arguments.get("--mf")
    _outfile = _arguments.get("--out")
    _use_cuda = True if _arguments.get("--cuda").lower() == "true" else False
    _topic_arg = _arguments.get("--topic")
    _extract_method_str = _arguments.get("--em")

    _testEmbedNode = _arguments.get("--ten")
    embed_node = EmbedNodeHidden([_testEmbedNode])
    

    _topic_config = Topics.get_topic_config(_topic_arg)
    _extract_method = RelationExtraction.get_extract_method(_extract_method_str)

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    start_time = datetime.now()
    dt_string = start_time.strftime("%Y-%m-%d_%H-%M")
    _output_folder = create_and_get_path("checkpoints")
    log_params_str = dt_string+"_ds_" + "wec-eng" + "lexical_overlap_rate"
    create_logger_with_fh(_output_folder + "/depth_experiment_" + log_params_str)

    logger.info("loading model from-" + ntpath.basename(_model_file))
    # _event_topics = Topics()
    # _event_topics.create_from_file(_mentions_file, True)

    # if _topic_config == TopicConfig.Corpus and len(_event_topics.topics_dict) > 1:
    #     _event_topics.to_single_topic()

    _cluster_algo = None
    _model = None
    if _extract_method == RelationTypeEnum.PAIRWISE:
        _model = get_pairwise_model()
    elif _extract_method == RelationTypeEnum.SAME_HEAD_LEMMA:
        _model = HeadLemmaRelationExtractor()

    logger.info("Running agglomerative clustering with model:" + type(_model).__name__)
   
   
    files = ['../experiments/lexical_overlap/overlap_0_10.pickle','../experiments/lexical_overlap/overlap_10_30.pickle','../experiments/lexical_overlap/overlap_30_50.pickle','../experiments/lexical_overlap/overlap_50_100.pickle']
    for file in files:
        filename = file.split("/")[-1]
        logger.info(f"experiment of lexical overlap on {filename}")
        predict_and_save(file)
