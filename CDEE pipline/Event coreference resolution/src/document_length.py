"""
Usage:
    document_length.py --tpf=<TestPosFile> --tnf=<TestNegFile> [--dataset_arg=<d>] [--dataset=<d>]
                   
Options:
    -h --help       Show this screen.
    --dataset_arg=<d>   wec - which dataset to generate for [default: wec]
    --dataset=<d>   wec-eng/wec-zh - which dataset to load [default: wec-eng]
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




import logging
import pickle

logger = logging.getLogger(__name__)
def load_data():
    test_dataset = DataSet.get_dataset(_dataset_arg, split=Split.Dev)
    test_feat = test_dataset.load_pos_neg_pickle(_test_pos_file, _test_neg_file)
    calculate(test_feat)

def calculate(test_feat,batch_size=10000):
    docu_length_0_512 = list()
    docu_length_512_1024 = list()
    docu_length_1024_upper = list()
    for mention_pair in tqdm(test_feat,desc="test pairs"):
        #，这里长度是考虑平均长度呢还是单个长度？平均长度更合理一些
        average_length = (len(mention_pair[0].mention_context)+len(mention_pair[1].mention_context))/2
        if average_length < 512:
            docu_length_0_512.append(mention_pair)
        elif average_length<1024:
            docu_length_512_1024.append(mention_pair)
        else:
            docu_length_1024_upper.append(mention_pair)
    logger.info(f"document length 0-512 pairs:{len(docu_length_0_512)}, document length 512-1024 pairs:{len(docu_length_512_1024)}, document length >1024 pairs:{len(docu_length_1024_upper)}")
    pickle.dump(docu_length_0_512,open("../experiments/document_length/document_length_0_512.pickle","w+b"))
    pickle.dump(docu_length_512_1024,open("../experiments/document_length/document_length_512_1024.pickle","w+b"))
    pickle.dump(docu_length_1024_upper,open("../experiments/document_length/document_length_1024_upper.pickle","w+b"))





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