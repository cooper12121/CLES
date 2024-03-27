"""

Usage:
    generate_pairs_predictions.py --tmf=<TestMentionsFile> --tef=<TestEmbedFile> --ten=<TestEmbedNode> --mf=<ModelFile> --out=<OurPredFile>
            [--cuda=<y>] [--topic=<type>] [--em=<ExtractMethod>]

Options:
    -h --help                   Show this screen.
    --cuda=<y>                  True/False - Whether to use cuda device or not [default: True]
    --topic=<type>              subtopic/topic/corpus -  take pairs only from the same sub-topic, topic or corpus wide [default: subtopic]
    --em=<ExtractMethod>        pairwize/head_lemma/exact_string - model type to run [default: pairwize]
"""

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

from src.preprocess_edu_embed import EmbedNodeHidden

logger = logging.getLogger(__name__)
MAX_ALLOWED_BATCH_SIZE = 200

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def generate_pred_matrix(inference_model, topic):
    all_pairs = list(product(topic.mentions, repeat=2))#两两组队
    pairs_chunks = [all_pairs]
    if len(all_pairs) > MAX_ALLOWED_BATCH_SIZE:
        pairs_chunks = [all_pairs[i:i + MAX_ALLOWED_BATCH_SIZE] for i in
                        range(0, len(all_pairs), MAX_ALLOWED_BATCH_SIZE)]#划分为子列表  
    predictions = np.empty(0)
    with torch.no_grad():
        for chunk in pairs_chunks:
            chunk_predictions, _ = inference_model.predict(chunk, bs=len(chunk))
            predictions = np.append(predictions, chunk_predictions.detach().cpu().numpy())
    predictions = 1 - predictions
    pred_matrix = predictions.reshape(len(topic.mentions), len(topic.mentions)) #两两组队的，这样操作是为什么[len_mention,len_mentions]#mention对预测矩阵
    return pred_matrix

    #得到gold label
    # gold_lable_dict=defaultdict(list)#每个coref_chain对应的index
    # index2coref =defaultdict(int)#注意原始数据的index是从1开始，而层次矩阵认为从0节点开始故减一
    # for mention in topic.mentions:
    #     gold_lable_dict[mention.coref_chain].append(mention.mention_index)
    #     index2coref[mention.mention_index-1]=mention.coref_chain
    # #再映射一个index2corefchain,便于去除单例样本
    
    # #放大3倍后划分社区
    # predictions=predictions*1000
    # for i in range(len(predictions)):
    #     if predictions[i]<500:
    #         predictions[i]=0#认为没有边连接
    # import louvain2
    # pred_matrix=predictions.reshape(len(topic.mentions), len(topic.mentions))
    # louvain2.test(pred_matrix,gold_lable_dict,index2coref)

   



def predict_and_save():
    all_predictions = list()
    for topic in _event_topics.topics_dict.values():
        logger.info("Evaluating Topic No:" + str(topic.topic_id))
        all_predictions.append((topic, generate_pred_matrix(_model, topic)))
    logger.info("Generating prediction file-" + _outfile)
    pickle.dump(all_predictions, open(_outfile, "wb"))


def get_pairwise_model():
    device = torch.device("cuda:0" if torch.cuda.is_available()else "cpu")


    # pairwize_model = torch.load(_model_file,map_location={'cuda:1':'cuda:0'})#默认存在哪个卡加载时就会加载到哪个卡，因此需要转
    pairwize_model = torch.load(_model_file,map_location=torch.device("cpu"))#默认存在哪个卡加载时就会加载到哪个卡，因此需要转
    # pairwize_model = torch.load(_model_file,map_location=lambda storage, loc: storage.cuda(0))#默认存在哪个卡加载时就会加载到哪个卡，因此需要转
    # pairwize_model = torch.load(_model_file,map_location=device)#要用torch.device包装

    if _use_cuda:
        # pairwize_model.cuda()
        pairwize_model.to(device)

    pairwize_model.set_embed_utils(EmbedFromFile([_embed_file]))

    pairwize_model.set_embed_node(embed_node)

   

    pairwize_model.eval()
    return pairwize_model


if __name__ == '__main__':
    os.chdir(sys.path[0])

    logging.basicConfig(level=logging.INFO)

    argv=['--tmf','../datasets/WEC-Eng/final_processed/Test_Event_gold_mentions_validated.json',
          '--tef','../datasets/WEC-Eng/final_processed/embed/Test_Event_gold_mentions_validated_roberta_large.pickle',
          '--ten','../datasets/WEC-Eng/final_processed/gat_embed/Test_Event_gold_mentions_validated_roberta_large.pickle',
          '--mf','../checkpoints/wec_pairwise_modeldev_ratio_-1_iter_10.pickle',
          '--out','../model/rhetorical/pair_pred_no_Elaboration.pickle',
          ]
    _arguments = docopt(__doc__, argv=argv, help=True, version=None, options_first=False)
    print(_arguments)
    _mentions_file = _arguments.get("--tmf")
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

    logger.info("loading model from-" + ntpath.basename(_model_file))
    _event_topics = Topics()
    _event_topics.create_from_file(_mentions_file, True)

    if _topic_config == TopicConfig.Corpus and len(_event_topics.topics_dict) > 1:
        _event_topics.to_single_topic()

    _cluster_algo = None
    _model = None
    if _extract_method == RelationTypeEnum.PAIRWISE:
        _model = get_pairwise_model()
    elif _extract_method == RelationTypeEnum.SAME_HEAD_LEMMA:
        _model = HeadLemmaRelationExtractor()

    logger.info("Running agglomerative clustering with model:" + type(_model).__name__)
    predict_and_save()
