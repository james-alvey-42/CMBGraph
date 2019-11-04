import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def correlation_fn(i, j, bJ):
	r"""
	Returns the correlation function in the Ising model between two sites i and j,

	.. math:: \langle \sigma_i \sigma_j \rangle = \exp(|i - j|\log\tanh \beta J)

	Parameters
	----------
	i, j : int
		site labels
	bJ : :math:`\beta J`
		combination of temperature and interaction parameters

	Returns
	-------
	corr : float
		correlation function
	"""
	return np.exp(np.abs(i - j)*np.log(np.tanh(bJ)))

def correlation_fn_periodic(i, j, N, bJ):
	r"""
	Returns the correlation function in the Ising model between two sites i and j,

	.. math:: \langle \sigma_i \sigma_j \rangle = \exp(|i - j|\log\tanh \beta J)

	Parameters
	----------
	i, j : int
		site labels
	N : int
		total number of sites
	bJ : :math:`\beta J`
		combination of temperature and interaction parameters

	Returns
	-------
	corr : float
		correlation function
	"""
	return np.exp(np.min([np.abs(i - j), N - np.abs(i - j)])*np.log(np.tanh(bJ)))

if __name__ == '__main__':
	N       = 20
	bJ      = 1
	j_arr   = np.arange(1, N, 1)
	corr_np = np.empty(len(j_arr))
	corr_p  = np.empty(len(j_arr))
	for j in j_arr:
		corr_np[j - 1] = correlation_fn(0, j, bJ)
		corr_p[j - 1]  = correlation_fn_periodic(0, j, N, bJ)
	plt.figure(figsize=(10, 5))
	ax = plt.subplot(1, 2, 1)
	ax.set_xlabel(r'$|i - j|$')
	ax.set_ylabel(r'$\langle \sigma_i \sigma_j \rangle$')
	ax.set_ylim(0.0, 1.0)
	ax.scatter(j_arr, corr_np, 
		color='#D1495B', 
		linewidths=1.0,
		s=20.0,
		marker='o', 
		alpha=0.9, 
		label='Edge Boundary Conditions')

	ax.text(0.7, 0.05, r'$\beta J = 1$')
	ax.legend(fontsize=12, frameon=True)

	ax = plt.subplot(1, 2, 2)
	ax.set_xlabel(r'$|i - j|$')
	ax.set_ylim(0.0, 1.0)
	ax.scatter(j_arr, corr_p, 
		color='#01295F', 
		linewidths=1.0,
		s=20.0,
		marker='o', 
		alpha=0.9, 
		label='Periodic Conditions')
	ax.legend(fontsize=12, frameon=True)
	ax.text(0.7, 0.05, r'$\beta J = 1$')
	plt.savefig('figures/ising_correlation.pdf')
