import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 

from scipy.stats import bernoulli, uniform, norm, truncnorm
import networkx as nx
import xml.etree.ElementTree as XMLParser


import warnings
warnings.simplefilter('ignore', FutureWarning)

class Data():

	def __init__(self, seed):
		self.mediators = []
		self.G = nx.DiGraph()
		self.indices = {}
		self.seed = seed
		self.X_train = np.array([])
		self.X_test = np.array([])
		self.y_train = np.array([])
		self.y_test = np.array([])
		self.true_train = np.array([])
		self.true_test = np.array([])
		self.true_y_train = np.array([])
		self.true_y_test = np.array([])
		np.random.seed(self.seed)

	def get_mediators(self):
		return self.mediators
	
	def get_predecessors(self,v):
		return [self.indices[i] for i in self.G.predecessors(v)]

	def set_graph(self, G):
		self.G = G

	def read_data(self, train_file, test_file):
		a = np.loadtxt(train_file, delimiter=',', dtype='float64',skiprows=1)
		self.X_train = a[:,0:-1]
		self.y_train = a[:,-1]

		a = np.loadtxt(test_file, delimiter=',', dtype='float64',skiprows=1)
		self.X_test = a[:,0:-1]
		self.y_test = a[:,-1]

class Data_Adult_Wu(Data):

	def gen_data(self,graph_file, train_file, test_file):
		self.read_graph_from_xml(graph_file)
		super().read_data(train_file, test_file)

	def read_graph_from_xml(self, file):
		tree = XMLParser.parse(file)
		root = tree.getroot()
		# self.G = nx.DiGraph()

		for i, node in enumerate(root.findall(".//discreteVariable")):
			node_id = node.get("name")
			self.indices[node_id] = i
			self.G.add_node(node_id)
			# self.unfair_path.add_node(node_id)
			
		for edges in root.findall(".//parentsFor"):
			source = edges.get("name")
			if source == "high_income":
				continue
			for edge in edges.findall("parent"):
				target = edge.get("name")
				self.G.add_edge(target,source)	
				
		self.mediators = [node for node in self.G.nodes() if self.G.in_degree(node) != 0]

class Data_Synth_Wu(Data):

	def gen_data(self, train_file, test_file, features, target):
		self.create_graph()
		self.read_data(train_file, test_file, features, target)

	def read_data(self, train_file, test_file, features, target):
		data = pd.read_csv(train_file, delimiter=',' ,dtype = float)
		self.X_train = np.array(data[features])
		self.y_train = np.array(data[target])
		# print(np.array(self.X_train))

		data = pd.read_csv(test_file, delimiter=',' ,dtype = float)
		self.X_test = np.array(data[features])
		self.y_test = np.array(data[target])

	def create_graph(self):
		node_data = "ASNM"
		self.indices = {v:i for i, v in enumerate(node_data)}
		self.G.add_nodes_from(node_data)
		self.G.add_edges_from([("A","S"),("A","M"),("A","N"),("S","N"),("S","M")])
		self.mediators = [node for node in self.G.nodes() if self.G.in_degree(node) != 0]
		# print(self.G.nodes)

class Data_Adult_Chikahara(Data_Adult_Wu):

	def gen_data(self,graph_file, train_file, test_file, features, target):
		self.read_graph_from_xml(graph_file)
		self.read_data(train_file, test_file, features, target)
		# print("debug:node", self.G)
		# print("debug:index", self.indices)
	
	def read_data(self, train_file, test_file, features, target):
		data = pd.read_csv(train_file)
		self.X_train = np.array(data[features])
		self.y_train = np.array(data[target])
		# print(np.array(self.X_train))

		data = pd.read_csv(test_file)
		self.X_test = np.array(data[features])
		self.y_test = np.array(data[target])

class Data_chikahara(Data):
	# def __init__(self, num, seed):
	# 	super().__init__(seed)
	# 	self.num = num

	def gen_data(self, num):
		self.create_graph()

		
		Ua = bernoulli.rvs(0.6, size=num)
		Uq = uniform.rvs(0.1, 1.0, size=num)
		# Uq = bernoulli.rvs(0.6, size=self.num)
		Ud = truncnorm.rvs(0.1, 3.0, 2, 1**2, size=num)
		Um = truncnorm.rvs(0.1, 3.0, 3, 2**2, size=num)
		# Um = bernoulli.rvs(0.7, size=self.num)

		A = Ua
		Q = Uq
		D = A + np.floor(0.5 * Q * Ud)
		M = A + 2 * Q * Um
		mu = self.sigmoid(- 10 + 5 * A + Q + D + M)
		X = np.stack([A, Q, D, M], axis=1)
		Y = bernoulli.rvs(mu)
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=1/6, random_state=42)

		# A = np.array([1 if a == 0 else 0 for a in A])
		A = np.ones(num)
		D = A + np.floor(0.5 * Q * Ud)
		M = A + 2 * Q * Um
		mu = self.sigmoid(- 10 + 5 * A + Q + D + M)
		X = np.stack([A, Q, D, M], axis=1)
		Y = bernoulli.rvs(mu)
		self.true_train, self.true_test, self.true_y_train, self.true_y_test = train_test_split(X, Y, test_size=1/6, random_state=42)
		# print(self.true_ref)


	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def create_graph(self):
		node_data = "AQDM"
		self.indices = {v:i for i, v in enumerate(node_data)}
		self.G.add_nodes_from(node_data)
		self.G.add_edges_from([("A","D"),("A","M"),("Q","D"),("Q","M")])
		self.mediators = [node for node in self.G.nodes() if self.G.in_degree(node) != 0]

class Data_Synthetic(Data):
	def __init__(self, seed, G, X_train, X_test, y_train, y_test):
		super().__init__(seed)
		self.G = G
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test
		self.mediators = [node for node in G.nodes() if G.in_degree(node) != 0]
		self.indices = {v:i for i, v in enumerate(G.nodes())}

class Data_German(Data):
	# def __init__(self,seed):

	def read_data(self, train_file, test_file, features, target):
		data = pd.read_csv(train_file, delimiter=',', dtype='float64')
		# print(data)
		self.X_train = np.array(data[features])
		self.y_train = np.array(data[target])
		# print(self.X_train,self.y_train)

		data = pd.read_csv(test_file, delimiter=',', dtype='float64')
		self.X_test = np.array(data[features])
		self.y_test = np.array(data[target])

	def create_graph(self):
		node_data = ["Sex","Age","Job","Housing","Saving accounts","Checking account","Credit amount","Duration","Purpose"]
		self.indices = {v:i for i, v in enumerate(node_data)}
		self.G.add_nodes_from(node_data)
		self.G.add_edges_from([("Sex","Saving accounts"),("Sex","Checking account"),("Sex","Housing"),("Sex","Job"),("Sex","Credit amount"),("Sex","Duration"),("Age","Saving accounts"),("Age","Checking account"),("Age","Housing"),("Age","Checking account"),("Age","Duration"),("Age","Job"),("Purpose","Saving accounts"),("Purpose","Checking account"),("Purpose","Housing"),("Purpose","Checking account"),("Purpose","Job"),("Purpose","Duration")])
		self.mediators = [node for node in self.G.nodes() if self.G.in_degree(node) != 0]