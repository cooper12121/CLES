"""
Usage:
    inference.py --tpf=<TestPosFile> --tnf=<testNegFile> --te=<TestEmbed> --mf=<ModelFile> --ten=<TestEmbedNode> [--cuda=<b>]

Options:
    -h --help       Show this screen.
    --cuda=<y>      True/False - Whether to use cuda device or not [default: True]

"""

import logging
import ntpath
import os,sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2]) #windows 用\
sys.path.append(root_path)
import torch
from docopt import docopt

from src.train import accuracy_on_dataset
from src.utils.log_utils import create_logger_with_fh
from src.dataobjs.dataset import EcbDataSet
from src.utils.embed_utils import EmbedFromFile
from src.Metric.calculate_metric  import process_cluster

from src.preprocess_edu_embed import EmbedNodeHidden

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    os.chdir(sys.path[0])
    argv = ['--tpf','../datasets/WEC-Eng/final_processed/pair/Test_Event_gold_mentions_validated_PosPairs.pickle',
            '--tnf','../datasets/WEC-Eng/final_processed/pair/Test_Event_gold_mentions_validated_NegPairs.pickle',
            '--te','../datasets/WEC-Eng/final_processed/embed/Test_Event_gold_mentions_validated_roberta_large.pickle',
            '--ten','../datasets/WEC-Eng/final_processed/gat_embed/Test_Event_gold_mentions_validated_roberta_large.pickle',
            '--mf','../model/wec_model.pickle']
    _arguments = docopt(__doc__, argv=argv, help=True, version=None, options_first=False)
    print(_arguments)
    _dataset_arg = _arguments.get("--dataset")
    _model_in = _arguments.get("--mf")
    _event_test_file_pos = _arguments.get("--tpf")
    _event_test_file_neg = _arguments.get("--tnf")
    _embed_file = _arguments.get("--te")

    _testEmbedNode = _arguments.get("--ten")

    _use_cuda = True if _arguments.get("--cuda").lower() == "true" else False

    _dataset = EcbDataSet()

    log_param_str = os.path.dirname(_model_in) + "/inference_" + ntpath.basename(_model_in)
    create_logger_with_fh(log_param_str)

    logger.info("Loading the model from-" + _model_in)
    _pairwize_model = torch.load(_model_in)
    _embed_utils = EmbedFromFile([_embed_file])
    _pairwize_model.set_embed_utils(_embed_utils)

    # embed_node = EmbedNodeHidden([_testEmbedNode])
    # _pairwize_model.set_embed_node(embed_node)

    _pairwize_model.eval()

    positive_ = _dataset.load_pair_pickle(_event_test_file_pos)
    negative_ = _dataset.load_pair_pickle(_event_test_file_neg)
    split_feat = _dataset.create_features_from_pos_neg(positive_, negative_)

    _, _, _, dev_f1,all_labels, all_predictions =accuracy_on_dataset("Test", 0, _pairwize_model, split_feat)
    # best_predict,best_label=all_predictions,all_labels
    # best_predict,best_label=best_predict.tolist(),best_label.tolist()
    # with open(f"../model/predict/test/{_dataset_arg}_test_best_predict_label.json",'w',encoding='utf-8')as f:
    #     json.dump({"best_predict":best_predict,"best_label":best_label},f,ensure_ascii=False)
    #     f.close()
    # process_cluster(best_predict,best_label,split_feat,"test","ratio:30")
    result = []
    for i in range(len(all_labels)):
        if all_labels[i]!=all_predictions[i]:
            context_0 = " ".join(i for i in split_feat[i][0].mention_context)
            context_1 = " ".join(i for i in split_feat[i][1].mention_context)
            event= {
                "event_0":split_feat[i][0].tokens_str+"____"+context_0,
                "event_1":split_feat[i][1].tokens_str+"____"+context_1,
                "gold":all_labels[i].tolist(),
                "pre":all_predictions[i].tolist()
            }
            result.append(event)
    import json
    with open("../datasets/error_samples.json",'w',encoding='utf-8')as f:
        json.dump(result,f,ensure_ascii=False)
