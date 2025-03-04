import numpy as np
import time

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx

from data import *

class Alg_dc:
	def __init__(self, data, unfair, attr_target, num_itr=1, var=0):
		self.data = data
		self.med_model = {}
		self.result = []
		self.attr_target = attr_target
		self.unfair_graph = unfair
		self.unfair_set = []
		self.var = var
		self.num_itr = num_itr

		self.unfair_med = list(nx.descendants(unfair, attr_target))
		self.med = list(nx.descendants(self.data.G, attr_target))
		
		if attr_target in self.unfair_med:
			self.unfair_med.remove(attr_target)
	
	def run(self, clf, has_exogenous=False, num_res = 20):
		self.init_mediator(has_exogenous)
		# self.init_mediator_with_noise()
		clf.fit(self.data.X_train, self.data.y_train)

		ref_unf = self.comp_ref_unfair(self.data.X_test,has_exogenous)
		ref = self.comp_ref(self.data.X_test,has_exogenous)

		dividend = self.comp_total_dividend(clf, ref, self.data.X_test)
		# dividend = self.comp_total_dividend(clf, self.data.X_test, ref)

		prob = clf.predict_proba(self.data.X_test)[:,1]
		true = clf.predict_proba(ref_unf)[:,1]
		self.mod_y = []

		epsilons = np.linspace(0, 1, num=num_res)
		acc_pce_ave = {}
		time_ave = {}
		for epsilon in epsilons:
			start_time = time.time()
			mod_prob = prob + dividend * epsilon
			pce = true - mod_prob
			self.mod_y = np.where(mod_prob >= 0.5, 1, 0)
			ave = np.average(np.abs(np.array(pce)))
			acc = accuracy_score(self.data.y_test, self.mod_y)
			acc_pce_ave[ave] = acc
			time_ave[ave] = time.time() - start_time
	
		return acc_pce_ave,time_ave

	def comp_unfair_set(self):
		# P_unfair
		Puf = list(nx.dfs_postorder_nodes(self.unfair_graph, self.attr_target))
		return Puf

	def comp_total_dividend(self, clf, x, ref):
		td = np.zeros(len(x))
		Puf = self.comp_unfair_set()

		Sp = [self.data.indices[i] for i in Puf]
		temp = ref.copy()
		temp[:,Sp] = x[:,Sp]
		td = td + clf.predict_proba(temp)[:,1] #\Delta(P_{unfair})
		td = td - clf.predict_proba(ref)[:,1] #\Delta(\emptyset)

		return  td

	def comp_ref_unfair(self, x, has_exogenous=False):  #compute reference value considering only unfair pathways 
		# mean_noise = 0
		# std_noise = 1 
		
		ref = x.copy()
		sense_att = self.data.indices[self.attr_target]
		ref[:,sense_att] = 1.0 # swap the value of sensitive attribute 0 -> 1 only
		# ref[:,sense_att] = 1 - ref[:,sense_att] # swap the value of sensitive attribute 0 -> 1 and 1 -> 0
		for v in self.unfair_med: # compute a reference value of mediators
			
			pred_id = [self.data.indices[i] for i in self.data.G.predecessors(v)]
			tmp = x.copy()
			unfair_id = [self.data.indices[i] for i in self.unfair_graph.predecessors(v) if i in self.data.G.predecessors(v)]
			tmp[:,unfair_id] = ref[:,unfair_id]
			tmp = tmp[:,pred_id]

			if has_exogenous:
				exo_est = self.est_exogenous(self.data.X_test[:,pred_id], self.data.X_test[:,self.data.indices[v]], v)
				tmp = np.hstack((tmp,np.array([exo_est]).T))

			ref[:,self.data.indices[v]] = self.med_model[v].predict(tmp)

		return ref
	
	def est_exogenous(self, X, y, v):
		# for v in self.med:
		selected_column = self.data.get_predecessors(v)
		selected_target = self.data.indices[v]
		p = len(selected_column)
		n = len(self.data.X_train)
		X = X[:,selected_column]

		beta = np.zeros(p)  
		gamma = 1.0  
		Z_est = np.zeros(n)  
		num_iterations = 100
		lambda_reg = 0  
	
		for i in range(num_iterations):
			Z_est = (y - X @ beta) / gamma

			beta = np.linalg.lstsq(X, y - gamma * Z_est, rcond=None)[0]
			gamma = (np.sum((y - X @ beta) * Z_est) / np.sum(Z_est ** 2)) / (1 + lambda_reg)
		return Z_est
	
	def comp_ref(self, x, has_exogenous = False): #compute reference value 
		mean_noise = 0
		std_noise = 1 
		ref = x.copy()
		sense_att = self.data.indices[self.attr_target]
		ref[:,sense_att] = 1.0 # swap the value of sensitive attribute 0 -> 1 only
		# ref[:,sense_att] = 1 - ref[:,sense_att] # swap the value of sensitive attribute 0 -> 1 and 1 -> 0
		for v in self.med: # compute a reference value of mediators
			pred_id = [self.data.indices[i] for i in self.data.G.predecessors(v)]
			tmp = x.copy()
			tmp[:, pred_id] = ref[:, pred_id]
			tmp = tmp[:, pred_id]			

			if has_exogenous:
				exo_est = self.est_exogenous(self.data.X_test[:,pred_id], self.data.X_test[:,self.data.indices[v]], v)
				tmp = np.hstack((tmp,np.array([exo_est]).T))

			ref[:, self.data.indices[v]] = self.med_model[v].predict(tmp)
		return ref

	def init_mediator(self, has_exogenous = False):
		for v in self.med:
			exo_est = self.est_exogenous(self.data.X_train, self.data.y_train, v) if has_exogenous else None
			self.med_model[v] = LinearRegression()
			# self.med_model[v] = MLPRegressor(hidden_layer_sizes=(100, ), max_iter=500, random_state=42)
			selected_column = self.data.get_predecessors(v)
			selected_target = self.data.indices[v]
			if has_exogenous:
				X = np.hstack((np.array(self.data.X_train[:,selected_column]), np.array([exo_est]).T))
			else:
				X = self.data.X_train[:,selected_column]
			self.med_model[v].fit(X, self.data.X_train[:,selected_target])
	
	
class Alg_dc_mis(Alg_dc): #algorithm for misspecification of mediator's model
	
	def run(self, clf, has_exogenous=False, num_res = 20, ):

		self.init_mediator(has_exogenous)
		clf.fit(self.data.X_train, self.data.y_train)

		ref = self.comp_ref(self.data.X_test, has_exogenous)
		dividend = self.comp_total_dividend(clf, ref, self.data.X_test)

		prob = clf.predict_proba(self.data.X_test)[:,1]
		true = clf.predict_proba(self.data.true_test)[:,1]
		self.mod_y = []
		
		epsilons = np.linspace(0, 1, num=num_res)
		acc_pce_ave = {}
		for epsilon in epsilons:
			mod_prob = prob + dividend * epsilon
			pce = true - mod_prob
			self.mod_y = np.where(mod_prob >= 0.5, 1, 0)
			ave = np.average(np.abs(np.array(pce)))
			acc = accuracy_score(self.data.y_test, self.mod_y)
			acc_pce_ave[ave] = acc
		return acc_pce_ave
	
class Alg_remove(Alg_dc):# remove sensitive attribute and features on unfair pathways
	
	def run(self,clf, target):
		time_ave = time.time()
		self.init_mediator()
		Puf = self.comp_unfair_set()
		Puf.append(target)
		Pf = list(self.data.G.nodes)
		_ = [Pf.remove(i) for i in Puf if i in Pf]
		Sp = [self.data.indices[i] for i in Pf]
		clf.fit(self.data.X_train[:,Sp], self.data.y_train)
		ref = self.comp_ref(self.data.X_test, False)
		X = ref[:, Sp]
		y = self.data.y_test
		
		yp = clf.predict(X)
		time_ave = time.time() - time_ave
		ac_te = accuracy_score(y, yp)
		return ac_te, time_ave