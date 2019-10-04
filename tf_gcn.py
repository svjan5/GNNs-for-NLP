from utils import *
import tensorflow as tf

class KipfGCN(object):

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
		self.data = {}
		self.A, self.X, self.data['y_train'], self.data['y_valid'], self.data['y_test'], \
		self.data['mask_train'], self.data['mask_valid'], self.data['mask_test'] = load_network(self.p.data)

		self.num_nodes    = self.X.shape[0]
		self.input_dim    = self.X.shape[1]
		self.X 	  	  = preprocess_features(self.X)
		self.A  	  = preprocess_adj(self.A)
		self.num_labels   = self.data['y_train'].shape[1]


	def add_placehoders(self):
		"""
		Defines the placeholder required for the model
		"""
		self.features 	 = tf.sparse_placeholder(tf.float32, 	shape=[self.num_nodes, self.input_dim], name='features')
		self.adj_mat 	 = tf.sparse_placeholder(tf.float32, 	shape=[self.num_nodes, self.num_nodes], name='support')
		self.labels 	 = tf.placeholder(tf.float32, 	  	shape=[None, self.num_labels], 		name='labels')
		self.labels_mask = tf.placeholder(tf.int32, 							name='labels_mask')
		self.dropout 	 = tf.placeholder_with_default(0.,   	shape=(), 				name='dropout')
		self.num_nonzero = tf.placeholder(tf.int32, 							name='num_nonzero')

	def create_feed_dict(self, split='train'):
		"""
		Creates the feed_dict for training the given step.
		A feed_dict takes the form of:
		feed_dict = {
			<placeholder>: <tensor of values to be passed for placeholder>,
			....
		}
	
		If label_batch is None, then no labels are added to feed_dict.
		Hint: The keys for the feed_dict should be a subset of the placeholder tensors created in add_placeholders.
		
		Parameters
		----------
		input_batch: 	A batch of input data.
		label_batch: 	A batch of label data.

		Returns
		-------
		feed_dict: 	The feed dictionary mapping from placeholders to values.
		"""
		feed = {}
		
		feed[self.features]	= self.X
		feed[self.adj_mat]	= self.A
		feed[self.num_nonzero]	= self.X[1].shape

		feed[self.labels]	= self.data['y_{}'.format(split)]
		feed[self.labels_mask]	= self.data['mask_{}'.format(split)]

		return feed

	def sparse_dropout(self, x, keep_prob, noise_shape):
		"""
		Dropout for sparse tensors.
		"""
		random_tensor  = keep_prob
		random_tensor += tf.random_uniform(noise_shape)
		dropout_mask   = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
		pre_out        = tf.sparse_retain(x, dropout_mask)
		return pre_out * (1./keep_prob)


	def GCNLayer(self, gcn_in, adj_mat, input_dim, output_dim, act, dropout, num_nonzero, input_sparse=False, name='GCN'):
		"""
		GCN Layer Implementation

		Parameters
		----------
		gcn_in:		Input to GCN Layer
		adj_mat:	Adjacency matrix
		input_dim:	Dimension of input to GCN Layer 
		output_dim:	Dimension of output of GCN Layer
		act:		Activation function used
		droptout:	Dropout probability
		num_numzero: 	Number of non-zero elements in input features (used when input_sparse=True)
		input_sparse:	Whether input features are sparse or not
		name; 		Name of the Layer

		Returns
		-------
		Output of GCN Layer
		
		"""

		with tf.name_scope(name):
			with tf.variable_scope('{}_vars'.format(name)) as scope:
				wts  = tf.get_variable('weights', [input_dim, output_dim], 	 initializer=tf.initializers.glorot_normal())
				bias = tf.get_variable('bias', 	  [output_dim], 	   	 initializer=tf.initializers.glorot_normal())
				self.l2_vars.extend([wts, bias])
			
			if input_sparse:
				gcn_in  = self.sparse_dropout(gcn_in, 1 - dropout, num_nonzero)
				pre_sup = tf.sparse_tensor_dense_matmul(gcn_in,  wts)
			else:
				gcn_in = tf.nn.dropout(gcn_in, 1-dropout)
				pre_sup = tf.matmul(gcn_in,  wts)

			support = tf.sparse_tensor_dense_matmul(adj_mat, pre_sup)

		return act(support)

	def add_model(self):

		gcn1_out = self.GCNLayer(		
				gcn_in 			= self.features,
				adj_mat 		= self.adj_mat,
				input_dim		= self.input_dim,
				output_dim		= self.p.gcn_dim,
				act			= tf.nn.relu,
				dropout			= self.dropout,
				num_nonzero 		= self.num_nonzero,
				input_sparse 		= True,
				name 			= 'GCN_1'
			)
		
		gcn2_out = self.GCNLayer(		
				gcn_in 			= gcn1_out,
				adj_mat 		= self.adj_mat,
				input_dim		= self.p.gcn_dim,
				output_dim		= self.num_labels,
				act			= lambda x: x,
				dropout			= self.dropout,
				num_nonzero 		= self.num_nonzero,
				input_sparse 		= False,
				name 			= 'GCN_2'
			)

		nn_out = gcn2_out
		return nn_out

	def get_accuracy(self, nn_out):
		"""
		Calculates accuracy

		Parameters
		----------
		nn_out:		Logits for each bag in the batch

		Returns
		-------
		accuracy:	Classification accuracy for labeled nodes
		"""

		correct_prediction 	 = tf.equal(tf.argmax(nn_out, 1), tf.argmax(self.labels, 1))	# Identity position where prediction matches labels
		accuracy_all 		 = tf.cast(correct_prediction, tf.float32)			# Cast result to float
		mask 			 = tf.cast(self.labels_mask, dtype=tf.float32)			# Cast mask to float
		mask 			/= tf.reduce_mean(mask)						# Compute mean of mask
		accuracy_all 		*= mask 							# Apply mask on computed accuracy

		return tf.reduce_mean(accuracy_all)


	def add_loss_op(self, nn_out):
		"""
		Computes loss based on logits and actual labels

		Parameters
		----------
		nn_out:		Logits for each bag in the batch

		Returns
		-------
		loss:		Computes loss based on prediction and actual labels of the bags
		"""

		loss  = tf.nn.softmax_cross_entropy_with_logits(logits=nn_out, labels=self.labels) 	# Compute cross entropy loss
		mask  = tf.cast(self.labels_mask, dtype=tf.float32)					# Cast masking from boolean to float
		mask /= tf.reduce_mean(mask)								# Compute mean for mask
		loss *= mask 										# Mask the output of cross entropy loss
		loss  = tf.reduce_mean(loss)

		for var in self.l2_vars:
			loss += self.p.l2 * tf.nn.l2_loss(var)

		return loss

	def add_optimizer(self, loss, isAdam=True):
		"""
		Add optimizer for training variables

		Parameters
		----------
		loss:		Computed loss

		Returns
		-------
		train_op:	Training optimizer
		"""
		with tf.name_scope('Optimizer'):
			if isAdam:  optimizer = tf.train.AdamOptimizer(self.p.lr)
			else:       optimizer = tf.train.GradientDescentOptimizer(self.p.lr)
			train_op  = optimizer.minimize(loss)

		return train_op

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

		# Vairable for storing variables which needs to be regularized
		self.l2_vars = []

		self.load_data()			# Load Dataset
		self.add_placehoders()			# Define Placeholders

		nn_out    	= self.add_model()		# Construct Computational Graph
		self.loss 	= self.add_loss_op(nn_out)
		
		self.accuracy 	= self.get_accuracy(nn_out)
		self.train_op 	= self.add_optimizer(self.loss)
		self.cost_val 	= []

		self.merged_summ = tf.summary.merge_all()


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

		t = time.time()
		feed_dict = self.create_feed_dict(split='train')
		feed_dict.update({self.dropout: self.p.dropout})

		# Training step
		_, train_loss, train_acc = sess.run([self.train_op, self.loss, self.accuracy], feed_dict=feed_dict)

		# Validation
		val_loss, val_acc = self.evaluate(sess, split='valid')

		if val_acc > self.best_val: 
			self.best_val		= val_acc
			_, self.best_test	= self.evaluate(sess, split='test')

		print(	"Epoch:", 	'%04d' % (epoch + 1), 
			"train_loss=", 	"{:.5f}".format(train_loss),
			"train_acc=",	"{:.5f}".format(train_acc), 
			"val_loss=", 	"{:.5f}".format(val_loss),
			"val_acc=", 	"{:.5f}".format(val_acc), 
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
	model = KipfGCN(args)

	# Start training and evaluation
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		model.fit(sess)

	print('Model Trained Successfully!!')