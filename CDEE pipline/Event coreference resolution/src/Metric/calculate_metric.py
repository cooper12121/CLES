import json
import pickle
import os,sys
os.chdir(sys.path[0])
sys.path.append('..')
from collections import defaultdict
from dataobjs.dataset import DataSet, Split
from metric4coref import muc, ceaf, b_cubed, conll_coref_f1
import logging
logger = logging.getLogger(__name__)

def calculate_cluster(data,pred_labels,gold_labels):
    """ calculate the cluster according the predict_label and gold lable 
    data;list[mention_pair]
    """
    cluster_pred=defaultdict(set)#共值链
    cluster_pred_id=0
    pred_index2cluster_dict = defaultdict(int)#映射实体对应的cluster

    cluster_gold=defaultdict(set)
    cluster_gold_id=0
    gold_index2cluster_dict=defaultdict(int)
    
    for index,mention_pair in enumerate(data):
        mention1,mention2=mention_pair[0],mention_pair[1]
        mention1_index,mention2_index=mention1.mention_index,mention2.mention_index

        mention1_pred_cluster_id = gold_index2cluster_dict.get(mention1_index)
        mention2_pred_cluster_id = gold_index2cluster_dict.get(mention2_index)

        mention1_gold_cluster_id = gold_index2cluster_dict.get(mention1_index)
        mention2_gold_cluster_id = gold_index2cluster_dict.get(mention2_index)

        #处理pred
        if pred_labels[index]==1:
            if mention1_pred_cluster_id!=None:#聚入当前类
                if mention2_pred_cluster_id==None:
                    cluster_pred[mention1_pred_cluster_id].add(mention2_index)#吧to_id加入from_id的聚类
                else:#不管她两相不相等求并集删除2即可
                    cluster_pred[mention1_pred_cluster_id]=cluster_pred[mention1_pred_cluster_id].union(cluster_pred[mention2_pred_cluster_id])
                    cluster_pred[mention2_pred_cluster_id].clear()
                pred_index2cluster_dict[mention2_index]=mention1_pred_cluster_id#修改映射关系，2并入1对应的聚类
            else:#from还没有聚类
                if mention2_pred_cluster_id==None:
                    cluster_pred[cluster_pred_id].add(mention1_index)
                    cluster_pred[cluster_pred_id].add(mention2_index)
                    pred_index2cluster_dict[mention1_index]=cluster_pred_id
                    pred_index2cluster_dict[mention2_index]=cluster_pred_id
                    cluster_pred_id+=1
                else:#1不存在而2存在，1并入2
                    cluster_pred[mention2_pred_cluster_id].add(mention1_index)
                    pred_index2cluster_dict[mention1_index]=mention2_pred_cluster_id
        else:#不共指分别聚类
            if mention1_pred_cluster_id!=None:#聚入当前类
                cluster_pred[mention1_pred_cluster_id].add(mention1_index)#吧to_id加入from_id的聚类
            else:#不管她两相不相等求并集删除2即可
                cluster_pred[cluster_pred_id].add(mention1_index)
                pred_index2cluster_dict[mention1_index]=cluster_pred_id#修改映射关系，2并入1对应的聚类
                cluster_pred_id+=1
            if mention2_pred_cluster_id!=None:#聚入当前类
                cluster_pred[mention2_pred_cluster_id].add(mention2_index)#吧to_id加入from_id的聚类
            else:#不管她两相不相等求并集删除2即可
                cluster_pred[cluster_pred_id].add(mention2_index)
                pred_index2cluster_dict[mention2_index]=cluster_pred_id#修改映射关系，2并入1对应的聚类
                cluster_pred_id+=1
            
    
    #gold
        if(gold_labels[index]==1):
            if mention1_gold_cluster_id!=None:#聚入当前类
                if mention2_gold_cluster_id==None:
                    cluster_gold[mention1_gold_cluster_id].add(mention2_index)#吧to_id加入from_id的聚类
                else:#不管她两相不相等求并集删除2即可
                    cluster_gold[mention1_gold_cluster_id]=cluster_gold[mention1_gold_cluster_id].union(cluster_gold[mention2_gold_cluster_id])
                    cluster_gold[mention2_gold_cluster_id].clear()
                gold_index2cluster_dict[mention2_index]=mention1_gold_cluster_id#修改映射关系，2并入1对应的聚类
            else:#from还没有聚类
                if mention2_gold_cluster_id==None:
                    cluster_gold[cluster_gold_id].add(mention1_index)
                    cluster_gold[cluster_gold_id].add(mention2_index)
                    gold_index2cluster_dict[mention1_index]=cluster_gold_id
                    gold_index2cluster_dict[mention2_index]=cluster_gold_id
                    cluster_gold_id+=1
                else:#1不存在而2存在，1并入2
                    cluster_gold[mention2_gold_cluster_id].add(mention1_index)
                    gold_index2cluster_dict[mention1_index]=mention2_gold_cluster_id
        else:
            if mention1_gold_cluster_id!=None:#聚入当前类
                cluster_gold[mention1_gold_cluster_id].add(mention1_index)#吧to_id加入from_id的聚类
            else:#不管她两相不相等求并集删除2即可
                cluster_gold[cluster_gold_id].add(mention1_index)
                gold_index2cluster_dict[mention1_index]=cluster_gold_id#修改映射关系，2并入1对应的聚类
                cluster_gold_id+=1
            if mention2_gold_cluster_id!=None:#聚入当前类
                cluster_gold[mention2_gold_cluster_id].add(mention2_index)#吧to_id加入from_id的聚类
            else:#不管她两相不相等求并集删除2即可
                cluster_gold[cluster_gold_id].add(mention2_index)
                gold_index2cluster_dict[mention2_index]=cluster_gold_id#修改映射关系，2并入1对应的聚类
                cluster_gold_id+=1
    return cluster_pred,cluster_gold
def process_cluster(predict_label,gold_label,data,split,ratio):
    cluster_data=read_validation_dataset(data)
    predict_labels,gold_labels=read_prediction_gold_label(predict_label,gold_label)
    assert len(cluster_data)==len(predict_labels)
    cluster_pred,cluster_gold=calculate_cluster(cluster_data,predict_labels,gold_labels)
    cluster_pred=[list(i)for i in cluster_pred.values() if len(i)>1]
    cluster_gold=[list(i)for i in cluster_gold.values() if len(i)>1]#抛弃单例聚类试一下
    print(f"len_clusterr_pred:{len(cluster_pred)},len_cluster_gold:{len(cluster_gold)}")
    metrics_result(cluster_pred,cluster_gold,split,ratio)
def metrics_result(cluster_pred,cluster_gold,split,ratio):
    with open(f"./{split}_metric.txt","a")as f:
        f.write(f"ratio:{ratio}\n")
        muc_metric=muc(cluster_pred,cluster_gold)
        f.write(f"muc_metric:{muc_metric}\n")
        b3_metric=b_cubed(cluster_pred,cluster_gold)
        f.write(f"b3_metric:{b3_metric}\n")
        ceaf_metric=ceaf(cluster_pred,cluster_gold)
        f.write(f"ceaf_metric:{ceaf_metric}\n")
        conll_coref_f1_metric=conll_coref_f1(cluster_pred,cluster_gold)
        f.write(f"conll_metric:{conll_coref_f1_metric}\n")
        f.close()
        logger.info(f"muc:{muc_metric},b3:{b3_metric},ceaf:{ceaf_metric},conll_f1:{conll_coref_f1_metric}")
def in_list(clu_list,offset):
    for clu_ls in clu_list:
        if clu_ls==offset:
            return True
    return False

def read_prediction_gold_label(predict_labels=None,gold_labels=None):
    """ 
    read the best predict_label and gold label from file
    label的顺序与test/dev pair的顺序一致，直接读入pair判断其真正的共指链即可
    """
    if predict_labels!=None:
        return predict_labels,gold_labels
    else:
        with open('../../model/wec_dev_best_predict_dev_label.json','r',encoding='utf-8')as f:
            data=json.load(f)
            f.close()
        predict_labels=data['best_predict']
        gold_labels=data['best_label']
        assert len(predict_labels)==len(gold_labels)
        return predict_labels,gold_labels
def read_validation_dataset(data=None):
    """ 加载预测时用到的数据集（由于训练时加载会打乱顺序，故每次保存的都不同）"""
    if data!=None:
        return data
    else:
        data = pickle.load(open('../../model/wec_dev_best_validation_dataset.pickle','rb'))
        # print(data)
        return data
def calculate_metric():
    "report some kinds of metrics result"
    pass

def data_process():
    """ 读入相应数据，并处理 """
    pass

# if __name__ =="__main__":
# #     # read_prediction_gold_label()
# #     # data_process()
#     os.chdir(sys.path[0])
#     process_cluster(None,None,None,'test',10)
