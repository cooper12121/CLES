import math
import torch
from torch import nn
import os,sys
import traceback
root_path = os.path.abspath(__file__)

root_path = '/'.join(root_path.split('/')[:-3]) #windows 用\
sys.path.append(root_path)
import dgl
from gatv2.GAT import GAT
from gatv2.graph import Graph

#gat_featrues
class PairWiseModelKenton(nn.Module):
    def __init__(self, f_in_dim, f_hid_dim, f_out_dim, embed_utils,embed_node, use_cuda,use_arguments=False,RST=True,Lexical_chain=True):
        super(PairWiseModelKenton, self).__init__()

        
        self.attend = self.get_sequential(embed_utils.get_embed_size(), f_hid_dim)
        self.w_alpha = nn.Linear(f_hid_dim, 1)
        self.embed_utils = embed_utils
        self.embed_node = embed_node
        self.use_cuda = use_cuda
        self.use_arguments=use_arguments
        self.W = nn.Linear(f_hid_dim, f_out_dim)
        # if self.use_arguments:
        #     self.pairwize = self.get_sequential(9 * f_in_dim+4, f_hid_dim)
        # else:
        #     self.pairwize = self.get_sequential(12* f_in_dim, f_hid_dim)

        self.pairwize = self.get_sequential(12*f_in_dim,f_hid_dim)
        
        # self.arguments_pairwize=self.get_sequential(3*f_in_dim,f_hid_dim)
        self.gat = GAT(in_feats=embed_node.get_embed_size(),out_feats=embed_node.get_embed_size())#输出的特征数与pairwise的hidden特征数相同,因为需要要做拼接，而拼接时mention的hidden为embed_size而不是hid-dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available()else "cpu")

        self.RST = RST#消融实验时使用
        self.lexical_chain = Lexical_chain

    @staticmethod
    def get_sequential(ind, hidd):
        return nn.Sequential(
            nn.Linear(ind, hidd),
            nn.ReLU(),
            nn.Linear(hidd, hidd),
            nn.ReLU(),
            nn.Linear(hidd, hidd),
            nn.ReLU(),
        )

    def forward(self, batch_features, bs):
        embeded_features, gold_labels = self.get_bert_rep(batch_features, bs)
        
        prediction = self.W(self.pairwize(embeded_features))  #[batch,12,hidden]
        return prediction, gold_labels

    def predict(self, batch_features, bs):
        output, gold_labels = self.__call__(batch_features, bs)
        prediction = torch.sigmoid(output)  #[batch,1]
        return prediction, gold_labels
    
    def get_node_rep(self,batch_features,batch_size=32):
        mentions1, mentions2 = zip(*batch_features)
        #处理gat信息，有mention1、2都可以,由于只取单个向量，故不用填充
        hiddens_node1,mentions1_node_index=zip(*self.embed_node.get_hidden_rep(mentions1))#(batch,node_num,1024) ([[tensor1,tenfor2,]])
        hiddens_node2,mentions2_node_index=zip(*self.embed_node.get_hidden_rep(mentions2))
       
        #建图,为每个图赋初始值
        batched_graph1,graphs1 = self.construct_graph(mentions1,hiddens_node1)
        output1 = self.gat(batched_graph1,batched_graph1.ndata['feat'])
        # graphs = dgl.unbatch(batched_graph1)
        node_features1=[]
        start=0 #把output中的编码对应到原始编号：因为batch后对所有图重新进行了编号
        for i,mention in enumerate(mentions1):
            node_index =mention.mention_node_index
            node_hidden1 = output1[start+node_index].tolist()
            start+=graphs1[i].num_nodes()
            node_features1.append(node_hidden1)
        hidden_node_features1 = torch.tensor(node_features1,requires_grad=True).view(batch_size,-1).to(self.device)#[batch,hidden]

        batched_graph2,graphs2 = self.construct_graph(mentions2,hiddens_node2)#这里建图出错了，500个mention，建了499个图
        output2 = self.gat(batched_graph2,batched_graph2.ndata['feat'])
        # graphs = dgl.unbatch(batched_graph1)
        node_features2=[]
        start=0 #把output中的编码对应到原始编号：因为batch后对所有图重新进行了编号
        for i,mention in enumerate(mentions2):
            node_index =mention.mention_node_index
            node_hidden2 = output2[start+node_index].tolist()
            start+=graphs2[i].num_nodes()
            node_features2.append(node_hidden2)
        hidden_node_features2 = torch.tensor(node_features2,requires_grad=True).view(batch_size,-1).to(self.device)


        return hidden_node_features1,hidden_node_features2
        


        #最终的返回值应该是[batch,hidden]#每个样本只取一个节点的信息
    def construct_graph(self,mentions,hidden_nodes):
        graphs = []
        for i,mention in enumerate(mentions):
            try:
                graph = Graph(mention.graph_info_list,mention.edu_list,mention.mention_node_index,hidden_nodes[i]).G
                g = dgl.add_self_loop(graph).to(self.device)
                graphs.append(g)
            except Exception as e:
                traceback.print_exc()
                print(e,i,len(graphs))
                #第220个样本出错
        batched_graph = dgl.batch(graphs).to(self.device)
        return batched_graph,graphs

    def get_bert_rep(self, batch_features, batch_size=32):
        mentions1, mentions2 = zip(*batch_features)
        # (batch_size, embed_utils.get_embed_size())
    #     if self.use_arguments:

    #         hiddens1, first1_tok, last1_tok, ment1_size ,arg0_1_hidden,arg0_1_size,arg1_1_hidden,arg1_1_size,loc_1_hidden,loc_1_size,time_1_hidden,time_1_size= zip(*self.embed_utils.get_mentions_rep(mentions1))
    #    #first_tok:[batch,hidden]但每个batch是一个元组 (0,...batch_size-1)
    #         hiddens2, first2_tok, last2_tok, ment2_size,arg0_2_hidden,arg0_2_size,arg1_2_hidden,arg1_2_size,loc_2_hidden,loc_2_size,time_2_hidden,time_2_size = zip(*self.embed_utils.get_mentions_rep(mentions2))
    #         #[batch,seqlen,hidden]
    #     else:
        hiddens1, first1_tok, last1_tok, ment1_size= zip(*self.embed_utils.get_mentions_rep(mentions1))
        hiddens2, first2_tok, last2_tok, ment2_size= zip(*self.embed_utils.get_mentions_rep(mentions2))
        max_ment_span = max([max(ment1_size), max(ment2_size)])#hid:[seq,hidden] #这里是一个batch
        hiddens1_pad = [torch.nn.functional.pad(hid, [0, 0, 0, max_ment_span - hid.shape[0]]) for hid in hiddens1]#操作的维度是从后往前，也就是先最后一维左侧填充0维，右侧0维，倒数第二个维度，左侧0维度，右侧max-shape[0维度，shape[0]=seq_len   
        hiddens2_pad = [torch.nn.functional.pad(hid, [0, 0, 0, max_ment_span - hid.shape[0]]) for hid in hiddens2]#[batch,maxlen,hidd]
        #pad  [seq+max-shape[0],hidden]也就是将长度统一填充为max_span  hidden_span =[tensor()]


        hiddens1_pad = torch.cat(hiddens1_pad)#取掉第一维度的list，即为tensor[batch*max_span,hidden]
        hiddens2_pad = torch.cat(hiddens2_pad)
        first1_tok = torch.cat(first1_tok).reshape(batch_size, -1)#去掉外层的元组
        first2_tok = torch.cat(first2_tok).reshape(batch_size, -1)
        last1_tok = torch.cat(last1_tok).reshape(batch_size, -1)
        last2_tok = torch.cat(last2_tok).reshape(batch_size, -1)


        if self.use_cuda:
            # hiddens1_pad = hiddens1_pad.cuda()
            # hiddens2_pad = hiddens2_pad.cuda()
            # first1_tok = first1_tok.cuda()
            # first2_tok = first2_tok.cuda()
            # last1_tok = last1_tok.cuda()
            # last2_tok = last2_tok.cuda()

            hiddens1_pad = hiddens1_pad.to(self.device)
            hiddens2_pad = hiddens2_pad.to(self.device)
            first1_tok = first1_tok.to(self.device)
            first2_tok = first2_tok.to(self.device)
            last1_tok = last1_tok.to(self.device)
            last2_tok = last2_tok.to(self.device)

        attend1 = self.attend(hiddens1_pad)
        attend2 = self.attend(hiddens2_pad)

        att1_w = self.w_alpha(attend1)
        att2_w = self.w_alpha(attend2)#[batch,max_len]

        # Clean attention on padded tokens
        att1_w = att1_w.reshape(batch_size, max_ment_span)
        att2_w = att2_w.reshape(batch_size, max_ment_span)
        self.clean_attnd_on_zero(att1_w, ment1_size, att2_w, ment2_size, max_ment_span)

        att1_soft = torch.softmax(att1_w, dim=1) #[batch_size,max_sent_span]
        att2_soft = torch.softmax(att2_w, dim=1)
        hidden1_reshape = hiddens1_pad.reshape(batch_size, max_ment_span, -1)#[batch,max_sent,hidden]
        hidden2_reshape = hiddens2_pad.reshape(batch_size, max_ment_span, -1)
        att1_head = hidden1_reshape * att1_soft.reshape(batch_size, max_ment_span, 1)#torch.mul,broadcast调整维度后对应元素相乘
        att2_head = hidden2_reshape * att2_soft.reshape(batch_size, max_ment_span, 1)  #[batch,max_span,hidden]
        
        # if not self.RST:
        #     g1 = torch.cat((first1_tok, last1_tok, torch.sum(att1_head, dim=1)), dim=1)#[batch,3*hidden],first_token:[batch,hidden]
        #     g2 = torch.cat((first2_tok, last2_tok, torch.sum(att2_head, dim=1)), dim=1)
        #     span1_span2 = g1 * g2  #g1,g2，就是mention的第一个，最后一个以及mention拼接的结果
        #     concat_result = torch.cat((g1, g2, span1_span2), dim=1) #最后把所有的参数信息都在这里拼接[batch,9*hidden]

        # else:
        #拼接gat的信息
        hidden_node_features1,hidden_node_features2 = self.get_node_rep(batch_features,batch_size)#[batch,hidden]
        g1 = torch.cat((first1_tok,last1_tok,torch.sum(att1_head,dim=1),hidden_node_features1),dim=1)#[batch,4*hidden]
        g2 = torch.cat((first2_tok,last2_tok,torch.sum(att2_head,dim=1),hidden_node_features2),dim=1)#[batch,4*hidden]
        span1_span2 = g1 * g2  #g1,g2，就是mention的第一个，最后一个以及mention拼接的结果[batch,4*hidden]
        concat_result = torch.cat((g1, g2, span1_span2), dim=1) #最后把所有的参数信息都在这里拼接[batch,12*hidden]



        

        
        # hiddens_mention1_node=[]
        # for i in range(len(mentions1_node_index)):
        #     hiddens_mention1_node.append(hiddens_node1[i][mentions1_node_index[i]])
        
        # hiddens_mention2_node=[]
        # for i in range(len(mentions2_node_index)):
        #     hiddens_mention2_node.append(hiddens_node2[i][mentions2_node_index[i]])#[batch,1,hidden]
        
        # hiddens_mention1_node = torch.tensor(hiddens_mention1_node).cuda()
        # hiddens_mention2_node = torch.tensor(hiddens_mention2_node).cuda()


        #处理参数信息,一个batch,要处理掉外面的元组,且长度不一为避免拼接，先在循环中sum成相同维度
        # hidden_size=hidden1_reshape.shape[-1]

        # arg0_1=torch.stack([torch.sum(i,dim=0) for i in arg0_1_hidden],dim=0).cuda()#[batch,hidden]   ,sum:[hidden]
        # arg0_2=torch.stack([torch.sum(i,dim=0) for i in arg0_2_hidden],dim=0).cuda()
        # # arg0= #计算一个arg0的结果：可以是拼接后投影的一维向量即可[batch,1]
        # arg0=self.w_alpha(self.arguments_pairwize ( torch.cat((arg0_1,arg0_2,arg0_1*arg0_2),dim=1)))#cat[batch,3*hidden],arg0[batch,1]

        # arg1_1=torch.stack([torch.sum(i,dim=0) for i in arg1_1_hidden],dim=0).cuda()#[batch,hidden]   ,sum:[hidden]
        # arg1_2=torch.stack([torch.sum(i,dim=0) for i in arg1_2_hidden],dim=0).cuda()
        # arg1=self.w_alpha(self.arguments_pairwize (torch.cat((arg1_1,arg1_2,arg1_1*arg1_2),dim=1)) )

        # loc_1=torch.stack([torch.sum(i,dim=0) for i in loc_1_hidden],dim=0).cuda()#[batch,hidden]   ,sum:[hidden]
        # loc_2=torch.stack([torch.sum(i,dim=0) for i in loc_2_hidden],dim=0).cuda()
        # loc = self.w_alpha(self.arguments_pairwize (torch.cat((loc_1,loc_2,loc_1*loc_2),dim=1)) )

        # time_1=torch.stack([torch.sum(i,dim=0) for i in time_1_hidden],dim=0).cuda()#[batch,hidden]   ,sum:[hidden]
        # time_2=torch.stack([torch.sum(i,dim=0) for i in time_2_hidden],dim=0).cuda()
        # time = self.w_alpha(self.arguments_pairwize (torch.cat((time_1,time_2,time_1*time_2),dim=1)) )
        

        # concat = torch.cat((concat_result,arg0,arg1,loc,time),dim=1)#[batch,9*hidden+4]
        
        ret_golds = torch.tensor(self.get_gold_labels(batch_features))

        if self.use_cuda:
            # concat_result = concat_result.cuda()
            # ret_golds = ret_golds.cuda()

            concat_result = concat_result.to(self.device)
            ret_golds = ret_golds.to(self.device)

            # concat=concat.cuda()
        # if self.use_arguments:
        #     return concat, ret_golds
        return concat_result,ret_golds

    @staticmethod
    def clean_attnd_on_zero(attend1, ment_size1, attend2, ment_size2, max_mention_span):
        for i, vals in enumerate(list(zip(ment_size1, ment_size2))):
            val1, val2 = vals
            if val1 > max_mention_span or val2 > max_mention_span:
                raise Exception("Mention size exceed maximum!")
            #将一个句子编码的mention位置保留，其余全部填充0
            attend1_fx = attend1[i:i + 1, 0:val1]#i应该指的是一个 batch中的每一个
            attend1_fx = torch.nn.functional.pad(attend1_fx, [0, max_mention_span - val1, 0, 0], value=-math.inf)
            attend1[i:i + 1] = attend1_fx

            attend2_fx = attend2[i:i + 1, 0:val2]
            attend2_fx = torch.nn.functional.pad(attend2_fx, [0, max_mention_span - val2, 0, 0], value=-math.inf)#在第二维度填充，也即val
            attend2[i:i + 1] = attend2_fx

    @staticmethod
    def get_gold_labels(batch_features):
        batch_labels = list()
        for mentions1, mentions2 in batch_features:
            gold_label = 1 if mentions1.coref_chain == mentions2.coref_chain else 0
            batch_labels.append(gold_label)
        return batch_labels

    def set_embed_utils(self, embed_utils):
        self.embed_utils = embed_utils
    
    def set_embed_node(self, embed_node):
        self.embed_node = embed_node
