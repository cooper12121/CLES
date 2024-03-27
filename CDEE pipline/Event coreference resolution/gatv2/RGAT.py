'''
关系图注意力网络
'''
from dgl.nn.pytorch import HeteroGraphConv
from dgl.nn.pytorch import GATv2Conv
from graph import Graph
import dgl
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import BertTokenizer,AutoTokenizer
class RGAT(nn.Module):
   def __init__(
       self,
       in_feats, # 输入的特征维度 （边和节点一样）
       hid_feats, # 隐藏层维度
       out_feats,  # 输出的维度
       num_heads, # 注意力头数
       rel_names, # 关系的名称（用于异构图卷积）
   ):
       super().__init__()
       self.conv1 = HeteroGraphConv({rel: GATv2Conv(in_feats, hid_feats // num_heads, in_feats, num_heads) for rel in rel_names},aggregate='sum')
       self.conv2 = HeteroGraphConv({rel: GATv2Conv(hid_feats, out_feats, in_feats, num_heads) for rel in rel_names},aggregate='sum')
       self.hid_feats = hid_feats

   def forward(self,graph,inputs):
       # graph 输入的异构图
       # inputs 输入节点的特征
       h = self.conv1(graph, inputs) # 第一层异构卷积
       h = {k: F.relu(v).view(-1, self.hid_feats) for k, v in h.items()} # 经过激活函数，将注意力头数拉平
       h = self.conv2(graph, h)  # 第二层异构卷积
       return h

def test():
    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1, 2], [2, 3, 2]),
        ('user', 'plays', 'game'): ([0, 0], [1, 0]),
        ('store', 'sells', 'game'): ([0], [2])})
    print(g.number_of_nodes('user'),g.number_of_nodes('game'),g.number_of_nodes('store'))
    print(g.number_of_edges('follows'),g.number_of_edges('plays'),g.number_of_edges('sells'))

def test01():
    s = "Explanation"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.encode(s)
    print(tokens)

if __name__ == "__main__":

    # wec文件时对graph_info中的id减1，dgl默认从0开始，而我处理时是从1开始的
    #节点类型只有一个，边的类型有多个，注意
    #边特征的初始化
    test01()
