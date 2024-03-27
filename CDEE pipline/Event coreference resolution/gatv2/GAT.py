import dgl
import torch
from dgl.nn.pytorch import GATv2Conv
from gatv2.graph import Graph
from torch import nn
import torch.nn.functional as F



class GAT(nn.Module):
    def __init__(self,
                 in_feats=1024,
                 hid_feats=512,
                 out_feats=512,
                 num_heads=2,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=True,
                 bias=True,
                 share_weights=False,
                 ):
        super().__init__()
        self.conv1 = GATv2Conv(in_feats, hid_feats // num_heads, num_heads,
                feat_drop=0.5,
                 attn_drop=0.5,
                 negative_slope=0.2,
                 residual=True,
                 activation=True,
                 allow_zero_in_degree=True,
                 bias=True,
                 share_weights=False,)
        self.conv2 = GATv2Conv(hid_feats, hid_feats, num_heads,
                               feat_drop=0.5,
                 attn_drop=0.5,
                 negative_slope=0.2,
                 residual=True,
                 activation=True,
                 allow_zero_in_degree=True,
                 bias=True,
                 share_weights=False,)  # hid*number模型自己会做
        self.projection = nn.Linear(hid_feats, out_feats)
        self.hid_feats = hid_feats
        self.num_heads=num_heads

    def forward(self, graph, inputs):
        # graph 输入的异构图
        # inputs 输入节点的特征
        h = self.conv1(graph, inputs)  # 第一层异构卷积
        h = F.relu(h).view(-1, self.hid_feats)  # 经过激活函数，将注意力头数拉平[34,3,2]->[2*3,1]
        h = self.conv2(graph, h)  # 第二层异构卷积[node,heads,out_feats]#将多个注意力头的结果相加作为最终结果
        # h = torch.sum(F.relu(h), dim=1)/self.num_heads
        h = torch.sum(F.relu(h), dim=1) #不用求平均吧
        h=self.projection(F.relu(h))
        return h


# def test():
#     graph = Graph()
#     in_feats = graph.node_features
#     model = GAT(in_feats, 128, 1, 2)
#     h = model(graph.G, graph.G.ndata['feat'])
#     print(h, h.shape)


# if __name__ == "__main__":
#     test()
