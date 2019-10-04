import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa


from utils import *
import tensorflow as tf


class Main(object):

	def load_data(self):
		"""
		Reads the data from pickle file

		Parameters
		----------
		self.p.dataset: The path of the dataset to be loaded

		Returns
		-------
		self.X: 	Input Node features
		self.A: 	Adjacency matrix
		self.num_nodes: Total nodes in the graph
		self.input_dim: 
		"""

		print("loading data")
		path		= osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
		dataset		= Planetoid(path, dataset, T.NormalizeFeatures())
		self.data	= dataset[0]


	def add_model(self, model_name):

		if   model_name == 'kipf_gcn': 	model = KipfGCN()
		elif model_name == 'syn_gcn':	model = SynGCN()
		else: raise NotImplementedError

		model.to(self.device)
		return model


	def add_optimizer(self, parameters):
		"""
		Add optimizer for training variables

		Parameters
		----------
		parameters:	Model parameters to be learned

		Returns
		-------
		train_op:	Training optimizer
		"""
		if self.p.opt == 'adam' : return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)
		else                    : return torch.optim.SGD(parameters,  lr=self.p.lr, weight_decay=self.p.l2)

	def __init__(self, params):
		"""
		Constructor for the main function. Loads data and creates computation graph. 

		Parameters
		----------
		params:		Hyperparameters of the model

		Returns
		-------
		"""
		self.p  = params

		self.p.save_dir = '{}/{}'.format(self.p.model_dir, self.p.name)
		if not os.path.exists(self.p.log_dir): 	os.system('mkdir -p {}'.format(self.p.log_dir))		# Create log directory if doesn't exist
		if not os.path.exists(self.p.save_dir): os.system('mkdir -p {}'.format(self.p.model_dir))	# Create model directory if doesn't exist

		# Get Logger
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
		self.logger.info(vars(self.p)); pprint(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')
	

		self.load_data()
		self.model        = self.add_model(self.p.model)
		self.optimizer    = self.add_optimizer(self.model.parameters())


	def evaluate(self, sess, split='valid'):
		"""
		Evaluate model on valid/test data

		Parameters
		----------
		sess:		Session of tensorflow
		split:		Data split to evaluate on

		Returns
		-------
		loss:		Loss over the entire data
		acc:		Overall Accuracy
		"""

		feed_dict 	= self.create_feed_dict(split=split)
		loss, acc 	= sess.run([self.loss, self.accuracy], feed_dict=feed_dict)

		return loss, acc

	def run_epoch(self, sess, epoch, shuffle=True):
		"""
		Runs one epoch of training and evaluation on validation set

		Parameters
		----------
		sess:		Session of tensorflow
		data:		Data to train on
		epoch:		Epoch number
		shuffle:	Shuffle data while before creates batches

		Returns
		-------
		loss:		Loss over the entire data
		Accuracy:	Overall accuracy

		"""

		self.model.train()
		self.optimizer.zero_grad()

		logits	= self.model(self.data.x, self.data.edge_index)[self.data.train_mask], self.data.y[self.data.train_mask]
		import pdb; pdb.set_trace()
		loss	= F.nll_loss().backward()
		self.optimizer.step()

		feed_dict = self.create_feed_dict(split='train')
		feed_dict.update({self.dropout: self.p.dropout})

		# Training step
		_, train_loss, train_acc = sess.run([self.train_op, self.loss, self.accuracy], feed_dict=feed_dict)

		# Validation
		val_loss, val_acc, _ = self.evaluate(sess, split='valid')

		if acc > self.best_val: 
			self.best_val		= acc
			_, self.best_test, _	= self.evaluate(sess, split='test')

		print(	"Epoch:", 	'%04d' % (epoch + 1), 
			"train_loss=", 	"{:.5f}".format(outs[1]),
			"train_acc=",	"{:.5f}".format(outs[2]), 
			"val_loss=", 	"{:.5f}".format(cost),
			"val_acc=", 	"{:.5f}".format(acc), 
			"time=", 	"{:.5f}".format(time.time() - t))


	def fit(self, sess):
		"""
		Trains the model and finally evaluates on test

		Parameters
		----------
		sess:		Tensorflow session object

		Returns
		-------
		"""
		self.saver		= tf.train.Saver()
		self.save_path		= os.path.join(self.p.save_dir, 'best_int_avg')

		self.best_val, self.best_test = 0.0, 0.0

		if self.p.restore:
			self.saver.restore(sess, self.save_path)

		for epoch in range(self.p.max_epochs):
			train_loss = self.run_epoch(sess, epoch)

		print('Best Valid: {}, Best Test: {}'.format(self.best_val, self.best_test))



if __name__== "__main__":

	parser = argparse.ArgumentParser(description='GNN for NLP tutorial - Kipf GCN')

	parser.add_argument('--data',     	dest="data",    	default='cora', 		help='Dataset to use')
	parser.add_argument('--gpu',      	dest="gpu",            	default='0',                	help='GPU to use')
	parser.add_argument('--name',     	dest="name",           	default='test',             	help='Name of the run')

	parser.add_argument('--lr',       	dest="lr",             	default=0.01,   type=float,     help='Learning rate')
	parser.add_argument('--epoch',    	dest="max_epochs",     	default=200,    type=int,       help='Max epochs')
	parser.add_argument('--l2',       	dest="l2",             	default=5e-4,   type=float,     help='L2 regularization')
	parser.add_argument('--seed',     	dest="seed",           	default=1234,   type=int,       help='Seed for randomization')
	parser.add_argument('--opt',      	dest="opt",            	default='adam',             	help='Optimizer to use for training')

	# GCN-related params
	parser.add_argument('--gcn_dim',  	dest="gcn_dim",     	default=16,     type=int,       help='GCN hidden dimension')
	parser.add_argument('--drop',     	dest="dropout",        	default=0.5,    type=float,     help='Dropout for full connected layer')
	parser.add_argument('--gcn_layer',	dest="gcn_layer",     	default=1,      type=int,       help='Number of layers in GCN over dependency tree')

	parser.add_argument('--restore',  	dest="restore",        	action='store_true',        	help='Restore from the previous best saved model')
	parser.add_argument('--log_dir',   	dest="log_dir",		default='./log/',   	   	help='Log directory')
	parser.add_argument('--model_dir',   	dest="config_dir",	default='./config/',        	help='Config directory')
	parser.add_argument('--config_dir',   	dest="model_dir",	default='./models/',        	help='Model directory')

	args = parser.parse_args()

	if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

	# Set seed
	tf.set_random_seed(args.seed)
	np.random.seed(args.seed)

	# Set GPU to use
	set_gpu(args.gpu)

	# Create Model
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	model = Main(args)
	model.fit()
	print('Model Trained Successfully!!')


"""
dataset = 'Cora'








def train():
	


def test():
	model.eval()
	logits, accs = model(), []
	for _, mask in data('train_mask', 'val_mask', 'test_mask'):
		pred = logits[mask].max(1)[1]
		acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
		accs.append(acc)
	return accs


best_val_acc = test_acc = 0
for epoch in range(1, 201):
	train()
	train_acc, val_acc, tmp_test_acc = test()
	if val_acc > best_val_acc:
		best_val_acc = val_acc
		test_acc = tmp_test_acc
	log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
	print(log.format(epoch, train_acc, best_val_acc, test_acc))

"""