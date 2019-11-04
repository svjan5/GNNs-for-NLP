import os.path as osp, argparse, time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv


class KipfGCN(torch.nn.Module):
	def __init__(self, data, num_class, params):
		super(KipfGCN, self).__init__()
		self.p     = params
		self.data  = data
		self.conv1 = GCNConv(self.data.num_features, self.p.gcn_dim, cached=True)
		self.conv2 = GCNConv(self.p.gcn_dim, num_class,  cached=True)

	def forward(self, x, edge_index):
		x		= F.relu(self.conv1(x, edge_index))
		x		= F.dropout(x, p=self.p.dropout, training=self.training)
		x		= self.conv2(x, edge_index)
		return F.log_softmax(x, dim=1)


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
		path		= osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', self.p.data)
		dataset		= Planetoid(path, self.p.data, T.NormalizeFeatures())
		self.num_class  = dataset.num_classes
		self.data	= dataset[0]


	def add_model(self):
		model = KipfGCN(self.data, self.num_class, self.p)
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
		self.data.to(self.device)
		self.model        = self.add_model()
		self.optimizer    = self.add_optimizer(self.model.parameters())


	def get_acc(self, logits, y_actual, mask):
		"""
		Calculates accuracy

		Parameters
		----------
		logits:		Output of the model
		y_actual: 	Ground truth label of nodes
		mask: 		Indicates the nodes to be considered for evaluation

		Returns
		-------
		accuracy:	Classification accuracy for labeled nodes
		"""

		y_pred = torch.max(logits, dim=1)[1]
		return y_pred.eq(y_actual[mask]).sum().item() / mask.sum().item()

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

	def run_epoch(self, epoch, shuffle=True):
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

		t = time.time()

		self.model.train()
		self.model.train()
		self.optimizer.zero_grad()

		logits		= self.model(self.data.x, self.data.edge_index)[self.data.train_mask]
		train_loss	= F.nll_loss(logits, self.data.y[self.data.train_mask])
		train_loss.backward()
		self.optimizer.step()

		self.model.eval()
		logits		= self.model(self.data.x, self.data.edge_index)
		train_acc	= self.get_acc(logits[self.data.train_mask], self.data.y, self.data.train_mask)
		val_acc		= self.get_acc(logits[self.data.val_mask], self.data.y, self.data.val_mask)

		if val_acc > self.best_val: 
			self.best_val	= val_acc
			self.best_test  = self.get_acc(logits[self.data.test_mask], self.data.y, self.data.test_mask)

		print(	"Epoch:", 	'%04d' % (epoch + 1), 
			"train_loss=", 	"{:.5f}".format(train_loss),
			"train_acc=",	"{:.5f}".format(train_acc), 
			"val_acc=", 	"{:.5f}".format(val_acc), 
			"time=", 	"{:.5f}".format(time.time() - t))


	def fit(self):
		"""
		Trains the model and finally evaluates on test

		Parameters
		----------
		sess:		Tensorflow session object

		Returns
		-------
		"""
		self.save_path = os.path.join(self.p.save_dir, 'best_int_avg')

		self.best_val, self.best_test = 0.0, 0.0

		if self.p.restore:
			self.saver.restore(self.save_path)

		for epoch in range(self.p.max_epochs):
			train_loss = self.run_epoch(epoch)

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
