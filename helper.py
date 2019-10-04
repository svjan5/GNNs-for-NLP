from pymongo import MongoClient
import os, sys, pdb, numpy as np, random, argparse, codecs, pickle, time, json, queue, re
from pprint import pprint
from collections import defaultdict as ddict
from joblib import Parallel, delayed
import numpy as np, sys, unicodedata, requests, os, random, pdb, requests, json, itertools, argparse, pickle
from random import randint
import networkx as nx
import scipy.sparse as sp
from pprint import pprint
import logging, logging.config, itertools, pathlib
from sklearn.metrics import precision_recall_fscore_support
import gzip, queue, threading
from threading import Thread
from scipy.stats import describe as des

np.set_printoptions(precision=4)

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))

def checkFile(filename):
	return pathlib.Path(filename).is_file()

def getWord2vec(wrd_list):
	dim = 300
	embeds = np.zeros((len(wrd_list), dim), np.float32)
	embed_map = {}

	res = db_word2vec.find({"_id": {"$in": wrd_list}})
	for ele in res:
		embed_map[ele['_id']] = ele['vec']

	count = 0
	for wrd in wrd_list:
		if wrd in embed_map: 	embeds[count, :] = np.float32(embed_map[wrd])
		else: 			embeds[count, :] = np.random.randn(dim)
		count += 1

	return embeds

def getGlove(wrd_list, embed_type, c_dosa=None):
	if c_dosa == None: c_dosa = MongoClient('mongodb://10.24.28.104:27017/')
	dim = int(embed_type.split('_')[1])
	db_glove = c_dosa['glove'][embed_type]

	embeds = np.zeros((len(wrd_list), dim), np.float32)
	embed_map = {}

	res = db_glove.find({"_id": {"$in": wrd_list}})
	for ele in res:
		embed_map[ele['_id']] = ele['vec']

	count = 0
	for wrd in wrd_list:
		if wrd in embed_map: 	embeds[count, :] = np.float32(embed_map[wrd])
		else: 			embeds[count, :] = np.random.randn(dim)
		count += 1

	return embeds

def getPhr2vec(phr_list, embed_type, c_dosa=None):
	if c_dosa == None: c_dosa = MongoClient('mongodb://10.24.28.104:27017/')
	dim 	 = int(embed_type.split('_')[1])
	db_glove = c_dosa['glove'][embed_type]
	
	wrd_list = []

	embeds    = np.zeros((len(phr_list), dim), np.float32)
	embed_map = {}

	for phr in phr_list:
		wrd_list += phr.split()

	wrd_list = list(set(wrd_list))

	res = db_glove.find({"_id": {"$in": wrd_list}})
	for ele in res:
		embed_map[ele['_id']] = ele['vec']
	
	for i, phr in enumerate(phr_list):
		wrds = phr.split()
		vec  = np.zeros((dim,), np.float32)
		
		for wrd in wrds:
			if wrd in embed_map: 	vec += np.float32(embed_map[wrd])
			else: 			vec += np.random.normal(size=dim, loc=0, scale=0.05)

		vec = vec / len(wrds)
		embeds[i, :] = vec
	return embeds

def getEmbeddings(embed_loc, wrd_list, embed_dims):
	embed_list = []

	wrd2embed = {}
	for line in open(embed_loc):
		data = line.strip().split(' ')
		wrd, embed = data[0], data[1:]
		embed = list(map(float, embed))
		wrd2embed[wrd] = embed

	for wrd in wrd_list:
		if wrd in wrd2embed: 	embed_list.append(wrd2embed[wrd])
		else: 	
			print('Word not in embeddings dump')
			embed_list.append(np.random.randn(embed_dims))

	return np.array(embed_list, dtype=np.float32)

def signal(message):
	requests.post( 'http://10.24.28.210:9999/jobComplete', data=message)

def len_key(tp):
	return len(tp[1])



def shape(tensor):
	s = tensor.get_shape()
	return tuple([s[i].value for i in range(0, len(s))])


coreNLP_url = [ 'http://10.24.28.106:9006/', 'http://10.24.28.106:9007/', 'http://10.24.28.106:9008/', 'http://10.24.28.106:9009/', 'http://10.24.28.106:9010/', 'http://10.24.28.106:9011/', 
		'http://10.24.28.106:9012/', 'http://10.24.28.106:9013/', 'http://10.24.28.106:9014/', 'http://10.24.28.106:9015/', 'http://10.24.28.106:9016/']

def callnlpServer(text):
        params = {
        	'properties': 	'{"annotators":"tokenize"}',
        	'outputFormat': 'json'
        }

        res = requests.post(	coreNLP_url[randint(0, len(coreNLP_url)-1)],
        			params=params, data=text, 
        			headers={'Content-type': 'text/plain'})

        if res.status_code == 200: 	return res.json()
        else: 				print("CoreNLP Error, status code:{}".format(res.status_codet))




def stanford_tokenize(text):
	res = callnlpServer(text)
	toks = [ele['word'] for ele in res['tokens']]
	return toks


def is_number(wrd):
	isNumber = re.compile(r'\d+.*')
	return isNumber.search(wrd)

isNumber = re.compile(r'\d+.*')
def norm_word(word):
	# if isNumber.search(word.lower()):
	# 	return '---num---'
	# if re.sub(r'\W+', '', word) == '':
	# 	return '---punc---'
	# else:
	return word.lower()
		

def is_int(s):
	try:
		int(s)
		return True
	except ValueError:
		return False



def partition(lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def getChunks(inp_list, chunk_size):
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))


pdb_multi  = '!import code; code.interact(local=vars())'
pdb_global = 'globals().update(locals())'