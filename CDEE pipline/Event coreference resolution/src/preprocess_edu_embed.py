"""
Usage:
	preprocess_edu_embed.py <File> [<File2>] [<File3>]
	preprocess_edu_embed.py <File> [<File2>] [<File3>] [--max=<x>]
	preprocess_edu_embed.py <File> [<File2>] [<File3>] [--cuda=<y>]
	preprocess_edu_embed.py <File> [<File2>] [<File3>] [--max=<x>] [--cuda=<y>]

Options:
	-h --help	Show this screen.
	--max=<x>	Maximum context [default: 511]
	--cuda=<y>  True/False - Whether to use cuda device or not [default: True].

"""
#用于生成edu的编码，初始化gat的节点

""" 
1.23修改记录：进行修辞关系的消融。
"""



import multiprocessing
import pickle
import random
import time

import os,sys
from os import path
import torch
from docopt import docopt

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2]) #windows 用\
sys.path.append(root_path)

import logging
from typing import List

import torch
from transformers import RobertaTokenizer, RobertaModel
from dataobjs.mention_data import MentionData

from dataobjs.topics import Topics
from utils.embed_utils import EmbedTransformersGenerics
logger = logging.getLogger(__name__)

def extract_feature_dict(topics: Topics, embed_model):
	result_train = dict()
	topic_count = len(topics.topics_dict)
	for topic in topics.topics_dict.values():
		mention_count = len(topic.mentions)  #全部mention信息
		for mention in topic.mentions:
			start = time.time()
			edu_node_hidden,mention_node_index= embed_model.get_mention_full_rep(mention)#编码器编码#,h_mention,h_mention[0],h_mention[-1],mention.shape[0]也即mention的长度
			#hidden:sentence_ids,
			end = time.time()  

			result_train[mention.mention_id] = (edu_node_hidden,mention_node_index) #每一个mention其text对应的编码
			# print("To Go: Topics" + str(topic_count) + ", Mentions" + str(mention_count) + ", took-" + str((end - start)))
			mention_count -= 1
		topic_count -= 1

	return result_train


def worker(resource_file, max_surrounding_contx, use_cuda,finetune):
	embed_model = EduEmbedModel(max_surrounding_contx=max_surrounding_contx, use_cuda=use_cuda,finetune=finetune)
	name = multiprocessing.current_process().name
	print(name, "Starting")

	basename = path.basename(path.splitext(resource_file)[0]) #Dev_Event_gold_mentions #file name has no suffix
	dirname = os.path.dirname(resource_file)
	save_to = dirname + "/gat_embed/" + basename + "_roberta_large.pickle"

	topics = Topics()
	topics.create_from_file(resource_file, keep_order=True) #对于wec，没有topic,topics={None:[所有的mention]}#此处是Mention类
	train_feat = extract_feature_dict(topics, embed_model)#字典{mentionid:（h_mention,h_mention[0],h_mention[-1], )}
	pickle.dump(train_feat, open(save_to, "w+b"))
	print("Done with -" + basename)

class EduEmbedModel():
	def __init__(self, max_surrounding_contx,
				 finetune=False, use_cuda=True):

		self.model = RobertaModel.from_pretrained("roberta-large")
		# self.model = BertModel.from_pretrained("bert-large-cased")
		self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
		# self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
		self.max_surrounding_contx = max_surrounding_contx
		self.use_cuda = use_cuda
		self.finetune = finetune
		self.embed_size = 1024

		device_ids=[0,1]
		
		if self.use_cuda:
			device = torch.device('cuda:{}'.format(device_ids[1]))#主gpu,输出时汇总梯度
			self.device=device
			# self.model=torch.nn.DataParallel(model,device_ids=device_ids,output_device=device_ids[0])
			self.model.to(device)
			# self.model.cuda()
	def get_mention_full_rep(self, mention):
		try:
			encode_ids,encode_edu_index=self.get_edu_encode(mention)
			mention_node_index=mention.mention_node_index

			if self.use_cuda:
				# encode_ids = encode_ids.cuda()
				encode_ids=encode_ids.to(self.device)

			if not self.finetune:
				with torch.no_grad():#不更新encoder参数
					last_hidden_span = self.model(encode_ids).last_hidden_state #编码器编码得到last_hidden_state  [batch=1,seq_len.1024]
			else:
				last_hidden_span = self.model(encode_ids).last_hidden_state #[batch,sequence,hidden_state],单个句子，batch=1
													#[sequence,hidden_state]         [mention，hidden_statte]
			hidden_span = last_hidden_span.view(last_hidden_span.shape[1], -1)

			edu_node_hidden = []
			for start,end in encode_edu_index:
				# h=hidden_span[start:end+1]
				node_hidden_span=torch.sum(hidden_span[start:end+1],dim=0)
				edu_node_hidden.append(node_hidden_span.cpu().tolist())#列表中的每个都是tensor用来初始化
			edu_node_hidden = torch.tensor(edu_node_hidden)

			return edu_node_hidden,mention_node_index
		except Exception as e:
			print(e)
	
	def get_edu_encode(self,mention):
		try:

			edu_list = mention.edu_list
			encode_ids_list = [self.tokenizer.cls_token_id]
			encode_edu_index = []
			for edu in edu_list:#错误的edu
				if isinstance(edu,list):
					edu=edu[0]
				if edu!=None and edu!=0:
					encode=self.tokenizer.encode(edu,add_special_tokens=False)
					start=len(encode_ids_list)
					if len(encode_ids_list)+len(encode)>self.max_surrounding_contx:
						break
					encode_ids_list.extend(encode)
					end = len(encode_ids_list)-1
					encode_edu_index.append([start,end])#注意取索引时end要包括在内
			encode_ids_list.append(self.tokenizer.sep_token_id)
			encode_ids = torch.tensor([encode_ids_list])

			return encode_ids,encode_edu_index
		except Exception as e:
			print(e)


	
	def get_embed_size(self):
		return self.embed_size


class EmbedNodeHidden(object):
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

	def get_hidden_rep(self, mentions_list):
		embed_list = [self.embeddings[self.embed_key[mention.mention_id]] for mention in mentions_list]
		return embed_list

	def get_embed_size(self):
		return self.embed_size

if __name__ == '__main__':
	os.chdir(sys.path[0])
	multiprocessing.set_start_method("spawn")
	argv = ['../datasets/WEC-Eng/final_processed/Dev_Event_gold_mentions_validated.json','../datasets/WEC-Eng/final_processed/Test_Event_gold_mentions_validated.json','../datasets/WEC-Eng/final_processed/Train_Event_gold_mentions.json','--max',511,'--cuda',True]
	arguments = docopt(__doc__, argv=argv, help=True, version=None, options_first=False)
	print(arguments)
	_file1 = arguments.get("<File>")
	_file2 = arguments.get("<File2>")
	_file3 = arguments.get("<File3>")
	_use_cuda = True if arguments.get("--cuda")== True else False
	_max_surrounding_contx = int(arguments.get("--max"))

	_all_files = list()
	if _file1:
		_all_files.append(_file1)
	if _file2:
		_all_files.append(_file2)
	if _file3:
		_all_files.append(_file3)

	torch.manual_seed(0)
	random.seed(0)
	if _use_cuda:
		torch.cuda.manual_seed(0)

	print("Processing files-" + str(_all_files))

	jobs = list()
	
	for _resource_file in _all_files:
		finetune=False
		# if(_resource_file=='../datasets/WEC-Eng/arguments/Train_Event_gold_mentions.json'):
		#     finetune=True
		job = multiprocessing.Process(target=worker, args=(_resource_file, _max_surrounding_contx, _use_cuda,finetune))
		jobs.append(job)
		job.start()

	for job in jobs:
		job.join()

	print("DONE!")
