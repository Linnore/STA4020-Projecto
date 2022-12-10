import numpy as np
from scipy.optimize import minimize

"""
This file constructs a strategic ERP portfolio and compare with an equal weighted portfolio.
x: portfolio weights
p_cov: covariance matrix
"""

# risk budgeting approach optimisation object function
# minimize variance
def obj_fun(x, p_cov):
	return np.sum((x*np.dot(p_cov, x)/np.dot(x.transpose(), np.dot(p_cov, x)))**2)

# maximize mean
def obj_fun2(x,p_mean):
	return -np.sum(x*p_mean)

# maximize  mean-variance
def obj_fun3(x,p_mean,p_cov):
	return -np.sum(x*p_mean)+np.sum((x*np.dot(p_cov, x)/np.dot(x.transpose(), np.dot(p_cov, x)))**2)

# maximize sharp ratio
def obj_fun4(x,p_mean,p_cov):
	return -np.sum(x*p_mean)/(np.sum((x*np.dot(p_cov, x)/np.dot(x.transpose(), np.dot(p_cov, x))))**2)

# constraint on sum of weights equal to one
def cons_sum_weight(x,tw):
	return -np.sum(x)+tw

# constraint on weight larger than zero
def cons_long_only_weight(x):
	return x

# constraint about the marignal risk w.r.t. each stocks(upper bound)
def cons_large_risk_constraint(x,p_cov):
	return -x*np.dot(p_cov, x)/np.dot(x.transpose(), np.dot(p_cov, x))+0.05

# constraint about the marignal risk w.r.t. each stocks(lower bound)
def cons_small_risk_constraint(x,p_cov):
	return x*np.dot(p_cov, x)/np.dot(x.transpose(), np.dot(p_cov, x))-0.02

# calculate risk budgeting portfolio weight given risk budget
def rb_p_weights(asset_rets, constrain="1"):
	"""_summary_

	Args:
		asset_rets (_type_): _description_
		constrain (str): In the format of "#+#+#". E.g., "1+3" specifies the first and third type of constraints.

	Returns:
		_type_: _description_
	"""
	# number of ARP series
	num_arp = asset_rets.shape[1]
	# covariance matrix of asset returns
	p_cov = asset_rets.cov()
	# mean of asset returns
	p_mean = asset_rets.mean()
	# initial weights
	w0 = 1.0 * np.ones((num_arp, 1)) / num_arp
	# constraints
	cons1 = {'type': 'ineq', 'fun': cons_sum_weight,'args':[2]}
	cons2 = {'type': 'ineq', 'fun': cons_long_only_weight}
	cons3 = {'type':'ineq','fun':cons_large_risk_constraint,'args':[p_cov]}
	cons4 = {'type':'ineq','fun':cons_small_risk_constraint,'args':[p_cov]}
	cons = [cons1, cons2, cons3, cons4]

	cons_strlist = constrain.split(sep='+')
	cons_list = []
	for strnum in cons_strlist:
		cons_list.append(cons[int(strnum)-1])
	# Portfolio optimisation 
	return minimize(obj_fun3, w0, args=(p_mean,p_cov), constraints=cons_list)