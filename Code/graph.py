import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ising import correlation_fn, correlation_fn_periodic
seed = 10
cmap = plt.cm.viridis_r

def plot_periodic_graph(N, figsize=(7,6), savefile='figures/non_periodic.pdf'):
	r"""
	Plot the periodic graph on N nodes.

	Parameters
	----------
	N : int
		number of nodes
	figsize : (float, float)
		figure size as a tuple
	savefile : str
		save location of resulting figure
	"""
	np.random.seed(seed)
	G = nx.Graph()
	G.add_nodes_from(np.arange(0, N, 1))
	node_colors = []
	colors = []
	labels = {}
	count  = 1
	for node in G.nodes():
		labels[node] = count
		if np.random.uniform(0, 1, 1) > 0.5:
			node_colors.append('#D1495B')
		else:
			node_colors.append('#419D78')
		count += 1
	for i in np.arange(0, N, 1):
		for j in np.arange(i + 1, N, 1):
			G.add_edge(i, j)
			colors.append(correlation_fn_periodic(i, j, N, bJ=1.0))
	pos = nx.circular_layout(G)
	plt.figure(figsize=figsize)
	plt.scatter([], [],
		marker='o',
		color='#D1495B',
		linewidths=0.0,
		alpha=0.9,
		s=30.0,
		label=r'$\sigma_i = +1$')
	plt.scatter([], [],
		marker='o',
		color='#419D78',
		linewidths=0.0,
		alpha=0.9,
		s=30.0,
		label=r'$\sigma_i = -1$')
	nx.draw_networkx_nodes(G, pos,
						node_color=node_colors,
						node_size=500,
						alpha=0.9)                       
	edges = nx.draw_networkx_edges(G, pos, 
						width=1.5, 
						alpha=0.8, 
						edge_color=colors,
						edge_cmap=cmap)
	nx.draw_networkx_labels(G, pos, labels, font_size=16)
	cb = plt.colorbar(edges, 
		norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0),
		boundaries=np.linspace(0,1,100))
	cb.set_ticks(np.linspace(0, 1, 11))
	cb.set_label(label=r'$\langle \sigma_i \sigma_j \rangle$', 
		fontsize=20)
	cb.set_clim(0,1)
	plt.axis('off')
	plt.title('Periodic Ising Model')
	plt.legend(loc='lower left', fontsize=16)
	plt.savefig('figures/periodic.pdf')

def plot_non_periodic_graph(N, figsize=(7,6), savefile='figures/non_periodic.pdf'):
	r"""
	Plot the periodic graph on N nodes.

	Parameters
	----------
	N : int
		number of nodes
	figsize : (float, float)
		figure size as a tuple
	savefile : str
		save location of resulting figure
	"""
	np.random.seed(seed)
	G = nx.Graph()
	G.add_nodes_from(np.arange(0, N, 1))
	node_colors = []
	colors = []
	labels = {}
	count  = 1
	for node in G.nodes():
		labels[node] = count
		if np.random.uniform(0, 1, 1) > 0.5:
			node_colors.append('#D1495B')
		else:
			node_colors.append('#419D78')
		count += 1
	for i in np.arange(0, N, 1):
		for j in np.arange(i + 1, N, 1):
			G.add_edge(i, j)
			colors.append(correlation_fn(i, j, bJ=1.0))
	pos = nx.circular_layout(G)
	plt.figure(figsize=figsize)
	plt.scatter([], [],
		marker='o',
		color='#D1495B',
		linewidths=0.0,
		alpha=0.9,
		s=30.0,
		label=r'$\sigma_i = +1$')
	plt.scatter([], [],
		marker='o',
		color='#419D78',
		linewidths=0.0,
		alpha=0.9,
		s=30.0,
		label=r'$\sigma_i = -1$')
	nx.draw_networkx_nodes(G, pos,
						node_color=node_colors,
						node_size=500,
						alpha=0.9)                       
	edges = nx.draw_networkx_edges(G, pos, 
						width=1.5, 
						alpha=0.8, 
						edge_color=colors,
						edge_cmap=cmap)
	nx.draw_networkx_labels(G, pos, labels, font_size=16)
	cb = plt.colorbar(edges, 
		norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0),
		boundaries=np.linspace(0,1,100))
	cb.set_label(label=r'$\langle \sigma_i \sigma_j \rangle$', 
		fontsize=20)
	cb.set_ticks(np.linspace(0, 1, 11))
	cb.set_clim(0,1)
	plt.axis('off')
	plt.title('Non-periodic Ising Model')
	plt.legend(loc='lower left', fontsize=16)
	plt.savefig('figures/non_periodic.pdf')

if __name__ == '__main__':
	plot_periodic_graph(N=12)
	plot_non_periodic_graph(N=12)