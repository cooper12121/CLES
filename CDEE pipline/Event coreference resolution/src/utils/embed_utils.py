import logging
import pickle
from typing import List

import torch
from transformers import RobertaTokenizer, RobertaModel
from dataobjs.mention_data import MentionData

logger = logging.getLogger(__name__)


class EmbedTransformersGenerics(object):
	def __init__(self, max_surrounding_contx,
				 finetune=False, use_cuda=True):

		model = RobertaModel.from_pretrained("roberta-large")
		# self.model = BertModel.from_pretrained("bert-large-cased")
		self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
		# self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
		self.max_surrounding_contx = max_surrounding_contx
		self.use_cuda = use_cuda
		self.finetune = finetune
		self.embed_size = 1024

		device_ids=[0,1]
		
		if self.use_cuda:
			device = torch.device('cuda:{}'.format(device_ids[0]))#主gpu,输出时汇总梯度
			self.device=device
			# self.model=torch.nn.DataParallel(model,device_ids=device_ids,output_device=device_ids[0])
			self.model.to(device)
			# self.model.cuda()

	def get_mention_full_rep(self, mention):
		sent_ids, ment1_inx_start, ment1_inx_end = self.mention_feat_to_vec(mention)
		# arg0_start_idx,arg0_end_idx,arg1_start_idx,arg1_end_idx,loc_start_idx,loc_end_idx,time_start_idx,time_end_idx=self.get_arguments(mention,sent_ids[0])

		if self.use_cuda:
			# sent_ids = sent_ids.cuda()
			sent_ids=sent_ids.to(self.device)

		if not self.finetune:
			with torch.no_grad():#不更新encoder参数
				last_hidden_span = self.model(sent_ids).last_hidden_state #编码器编码得到last_hidden_state
		else:
			last_hidden_span = self.model(sent_ids).last_hidden_state #[batch,sequence,hidden_state],单个句子，batch=1
												#[sequence,hidden_state]         [mention，hidden_statte]
		mention_hidden_span = last_hidden_span.view(last_hidden_span.shape[1], -1)[ment1_inx_start:ment1_inx_end]#取得mention的hidden

		# hidden_state = mention_hidden_span.shape[1]
		# if arg0_start_idx==-1:
		# 	arg0_hidden_span=torch.zeros((1,hidden_state)).cuda()   #[1,hidden_state]的零向量即可
		# else:
		# 	arg0_hidden_span=last_hidden_span.view(last_hidden_span.shape[1], -1)[arg0_start_idx:arg0_end_idx]
		
		# if arg1_start_idx==-1:
		# 	arg1_hidden_span=torch.zeros((1,hidden_state)).cuda()   #[1,hidden_state]的零向量即可
		# else:
		# 	arg1_hidden_span=last_hidden_span.view(last_hidden_span.shape[1], -1)[arg1_start_idx:arg1_end_idx]
		
		# if loc_start_idx==-1:
		# 	loc_hidden_span=torch.zeros((1,hidden_state)).cuda()   #[1,hidden_state]的零向量即可
		# else:
		# 	loc_hidden_span=last_hidden_span.view(last_hidden_span.shape[1], -1)[loc_start_idx:loc_end_idx]
		
		# if time_start_idx==-1:
		# 	time_hidden_span=torch.zeros((1,hidden_state)).cuda()   #[1,hidden_state]的零向量即可
		# else:
		# 	time_hidden_span=last_hidden_span.view(last_hidden_span.shape[1], -1)[time_start_idx:time_end_idx]

		# return mention_hidden_span, mention_hidden_span[0], mention_hidden_span[-1], mention_hidden_span.shape[0],arg0_hidden_span,arg0_hidden_span.shape[0],arg1_hidden_span,arg1_hidden_span.shape[0],loc_hidden_span,loc_hidden_span.shape[0],time_hidden_span,time_hidden_span.shape[0]
		return mention_hidden_span, mention_hidden_span[0], mention_hidden_span[-1], mention_hidden_span.shape[0]

	@staticmethod
	def extract_mention_surrounding_context(mention):
		tokens_inds = mention.tokens_number
		context = mention.mention_context #是列表
		start_mention_index = tokens_inds[0]
		end_mention_index = tokens_inds[-1] + 1
		assert len(tokens_inds) == len(mention.tokens_str.split(" "))

		ret_context_before = context[0:start_mention_index]
		ret_mention = context[start_mention_index:end_mention_index]
		ret_context_after = context[end_mention_index:]

		assert ret_mention == mention.tokens_str.split(" ")
		assert ret_context_before + ret_mention + ret_context_after == mention.mention_context

		return ret_context_before, ret_mention, ret_context_after

	def mention_feat_to_vec(self, mention):
		cntx_before_str, ment_span_str, cntx_after_str = EmbedTransformersGenerics.\
			extract_mention_surrounding_context(mention) #text中，mention前面的句子，mention,mention后面的句子，是列表

		cntx_before, cntx_after = cntx_before_str, cntx_after_str
		if len(cntx_before_str) != 0:
			cntx_before = self.tokenizer.encode(" ".join(cntx_before_str), add_special_tokens=False)#变列表为句子进行tokenizer
		if len(cntx_after_str) != 0:
			cntx_after = self.tokenizer.encode(" ".join(cntx_after_str), add_special_tokens=False)

		if self.max_surrounding_contx != -1:
			if len(cntx_before) > self.max_surrounding_contx:
				cntx_before = cntx_before[-self.max_surrounding_contx+1:]
			if len(cntx_after) > self.max_surrounding_contx:
				cntx_after = cntx_after[:self.max_surrounding_contx-1]

		ment_span = self.tokenizer.encode(" ".join(ment_span_str), add_special_tokens=False)

		if isinstance(ment_span, torch.Tensor):
			ment_span = ment_span.tolist()#转为列表是因为要存储到文件里，后面训练时加载后在存到torch
		if isinstance(cntx_before, torch.Tensor):
			cntx_before = cntx_before.tolist()
		if isinstance(cntx_after, torch.Tensor):
			cntx_after = cntx_after.tolist()

		all_sent_toks = [[0] + cntx_before + ment_span + cntx_after + [2]]#[0]和[2]是roberta的cls,sep(roberta用的是<s>,</s>),这里是二维列表[[]]
		sent_tokens = torch.tensor(all_sent_toks)   #[1,seqlen]#1是组成batch,便于送入编码器
		mention_start_idx = len(cntx_before) + 1  #加1是因为前面加入了cls
		mention_end_idx = len(cntx_before) + len(ment_span) + 1
		assert all_sent_toks[0][mention_start_idx:mention_end_idx] == ment_span
		return sent_tokens, mention_start_idx, mention_end_idx
	def get_arguments(self,mention,sent_tokens):
		arguments=mention.arguments
		arg0_start,arg0_end,arg0_span=arguments[0]#数据集中分词的位置
		arg0_start_idx,arg0_end_idx=self.wordpairID_2_tokenpairID(arg0_span,arg0_start,arg0_end,sent_tokens)#tokenizer后分词的位置

		arg1_start,arg1_end,arg1_span=arguments[1]
		arg1_start_idx,arg1_end_idx=self.wordpairID_2_tokenpairID(arg1_span,arg1_start,arg1_end,sent_tokens)

		loc_start,loc_end,loc_span=arguments[2]
		loc_start_idx,loc_end_idx=self.wordpairID_2_tokenpairID(loc_span,loc_start,loc_end,sent_tokens)

		time_start,time_end,time_span=arguments[3]
		time_start_idx,time_end_idx=self.wordpairID_2_tokenpairID(time_span,time_start,time_end,sent_tokens)

		return arg0_start_idx,arg0_end_idx,arg1_start_idx,arg1_end_idx,loc_start_idx,loc_end_idx,time_start_idx,time_end_idx
		
	def wordpairID_2_tokenpairID(self,span, wordindex_left, wordindex_right, sent_tokens):
		'''pls note that the input indices pair include the b in (a,b), but the output doesn't'''
		'''first find the position of [2,2]'''
		if wordindex_left==-1 or wordindex_right==-1:return -1,-1
		span = " ".join(i for i in span)
		if wordindex_left!=0:
			'''this span is the begining of the sent'''#不是开头
			span=' '+span#加一个空格，和句子中的词的分词相同（cls是开头）

		span_token_list = self.tokenizer.tokenize(span)
		span_id_list = self.tokenizer.convert_tokens_to_ids(span_token_list)#list,要转为tensor
		# print('span:', span, 'span_id_list:', span_id_list)
		sent_tokens=sent_tokens.tolist()
		for i in range(wordindex_left,len(sent_tokens)):
			if sent_tokens[i:i+len(span_id_list)] == span_id_list:
				return i, i+len(span_id_list)

		return -1, -1#对于None的直接用0向量表示即可
	def get_embed_size(self):
		return self.embed_size


class EmbedFromFile(object):
	def __init__(self, files_to_load: List[str]):
		"""
		:param files_to_load: list of pre-generated embedding file names
		"""
		self.embed_size = 1024
		bert_dict = dict()

		if files_to_load is not None and len(files_to_load) > 0:
			for file_ in files_to_load:
				bert_dict.update(pickle.load(open(file_, "rb")))
				logger.info("roberta representation loaded-" + file_)

		self.embeddings = list(bert_dict.values())
		self.embed_key = {k: i for i, k in enumerate(bert_dict.keys())}

	def get_mention_full_rep(self, mention):
		return self.embeddings[self.embed_key[mention.mention_id]]

	def get_mentions_rep(self, mentions_list):
		embed_list = [self.embeddings[self.embed_key[mention.mention_id]] for mention in mentions_list]
		return embed_list

	def get_embed_size(self):
		return self.embed_size



class EmbedInMemory(object):
	def __init__(self, mentions: List[MentionData], max_surrounding_contx, use_cuda):
		self.embed_size = 1024
		bert_dict = dict()
		embed_model = EmbedTransformersGenerics(max_surrounding_contx=max_surrounding_contx, use_cuda=use_cuda)

		for ment in mentions:
			hidden, first_tok, last_tok, ment_size = embed_model.get_mention_full_rep(ment)
			bert_dict[ment.mention_id] = (hidden.cpu(), first_tok.cpu(), last_tok.cpu(), ment_size)

		self.embeddings = list(bert_dict.values())
		self.embed_key = {k: i for i, k in enumerate(bert_dict.keys())}

	def get_mention_full_rep(self, mention):
		return self.embeddings[self.embed_key[mention.mention_id]]

	def get_mentions_rep(self, mentions_list):
		embed_list = [self.embeddings[self.embed_key[mention.mention_id]] for mention in mentions_list]
		return embed_list

	def get_embed_size(self):
		return self.embed_size
