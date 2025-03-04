import matplotlib.pyplot as plt
import numpy as np
# import tikzplotlib

def plot_result(acc_pce, write_tikz = False, output="test.tex"):
	ave_acc_pce = {}
	std_dev = {}
	for name in acc_pce.keys():
		ave_key = np.array([[key for key in a.keys()] for a in acc_pce[name]])
		ave_val = np.array([[val for val in a.values()] for a in acc_pce[name]])
		ave_key = np.average(ave_key,axis=0)
		std_dev[name] = np.std(ave_val,axis=0)
		ave_val = np.average(ave_val,axis=0)
		ave_acc_pce[name] = dict(zip(ave_key, ave_val))
	
	for k, d in ave_acc_pce.items():
		plt.plot(d.keys(), d.values(), label = k)
		plt.fill_between(d.keys(), list(d.values()) +std_dev[k], list(d.values()) - std_dev[k], alpha=0.15)

	plt.title ("Test accuracy v.s. Path-specific Counterfactual Effect")
	plt.xlabel("PCE")
	plt.ylabel("Test accuracy")
	
	# plt.ylim(0, 1)
	
	plt.grid()
	# if write_tikz: tikzplotlib.save(output)
	plt.legend()
	
	plt.show()
	
def plot_table(acc_pce):
	ave_acc_pce = {}
	std_dev = {}
	for name in acc_pce.keys():
		print(name)
		ave_key = np.array([[key  for key in a.keys()] for a in acc_pce[name]])
		ave_val = np.array([[val for val in a.values()] for a in acc_pce[name]])
		ave_key = np.average(ave_key,axis=0)
		std_dev[name] = np.std(ave_val,axis=0)
		ave_val = np.average(ave_val,axis=0)
		ave_acc_pce[name] = dict(zip(ave_key, ave_val))
		for k,v,s in zip(ave_key, ave_val, std_dev[name]):
			print(f"{k} {v} {s}")

def plot_table_md(acc_pce):
	ave_acc_pce = {}
	std_dev = {}
	for name in acc_pce.keys():
		print(name)
		ave_key = np.array([[key  for key in a.keys()] for a in acc_pce[name]])
		ave_val = np.array([[val for val in a.values()] for a in acc_pce[name]])
		ave_key = np.average(ave_key,axis=0)
		std_dev[name] = np.std(ave_val,axis=0)
		ave_val = np.average(ave_val,axis=0)
		ave_acc_pce[name] = dict(zip(ave_key, ave_val))

		sorted_list = sorted(zip(ave_key, ave_val, std_dev[name]))
		print([f"{sorted_list[x][0]:.3f}" for x in range(len(sorted_list)) if x % 2 == 0])
		print([f"{sorted_list[x][1]:.3f} $\pm$ {sorted_list[x][2]:.3f}" for x in range(len(sorted_list)) if x % 2==0])
		# print([f"{x[1]:.3f} $\pm$ {x[2]:.3f}" for x in sorted_list])