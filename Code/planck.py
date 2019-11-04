import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import numpy as np
from scipy.special import legendre

def get_Dl(filename):
	r"""
	Gets the :math:`\mathcal{D}_\ell` values in a datafile downloaded from the Planck legacy archive.
	
	Parameters
	----------
	filename : str
		file where the power spectrum data is stored

	Returns
	-------
	Dl : np.ndarray
		the power spectra angular co-efficients
	Dlerr : np.ndarray(2,)
		the error on the Dl coefficients
	"""
	arr = np.loadtxt(filename)
	Dl  = arr[:, 1]
	err = arr[:, -2:]
	return Dl, err

def get_l(filename):
	r"""
	Gets the :math:`\ell` values in a datafile downloaded from the Planck legacy archive.
	
	Parameters
	----------
	filename : str
		file where the power spectrum data is stored

	Returns
	-------
	l : np.ndarray
		the angular modes corresponding to the power spectrum
	"""
	arr = np.loadtxt(filename)
	l   = arr[:, 0]
	return l

def get_Cl(l, Dl):
	r"""
	Given the :math:`\mathcal{D}_\ell` from the Planck datasets, can compute the :math:`C_\ell` using,

	.. math:: C_\ell = \frac{2\pi \mathcal{D}_\ell}{\ell(\ell + 1)}
	
	Parameters
	----------
	l : np.ndarray
		array of the multipole values
	Dl : np.ndarray
		array of the angular correlation coefficients

	Returns
	-------
	Cl : np.ndarray
		array of rescaled angular correlation coefficients
	"""
	return 2*np.pi*Dl*np.power(l*(l + 1), -1)

def TT_corr(theta, l, Cl):
	r"""
	Computes the correlation function between two vectors :math:`\textbf{n}` and :math:`\textbf{n}^{\prime}`,

	.. math:: C(\theta) = \sum_{\ell}{\frac{2\ell + 1}{4\pi}C_{\ell} P_{\ell}(\textbf{n}\cdot\textbf{n}^{\prime})}
	
	Parameters
	----------
	theta : np.ndarray
		angular separation
	l : np.ndarray
		array of multipoles from Planck data file
	Cl : np.ndarray
		computed correlation coefficients from Planck data file

	Returns
	-------
	C : np.ndarray
		array of angular correlations
	"""
	Pl_arr = np.empty((len(l), len(theta)), dtype=float)
	for idx, multipole in enumerate(l):
		Pl_arr[idx, :] = legendre(int(multipole))(np.cos(theta))
	C = np.dot((2*l + 1)*Cl/(4*np.pi), Pl_arr)
	return C
	


if __name__ == '__main__':
	TT_file   = 'TTpower_spectra.txt'
	l         = get_l(TT_file)
	Dl, Dlerr = get_Dl(TT_file)
	Cl        = get_Cl(l, Dl)
	theta     = np.linspace(0.0, np.pi, 3000)
	mask 	  = (l < 1000)
	C         = TT_corr(theta, l[mask], Cl[mask])

	# plot C function
	plt.figure()
	plt.plot(theta, C, c='#D1495B', label=r'$C(\theta)$')
	plt.xlabel(r'$\theta$')
	plt.ylabel(r'$C(\theta)$')
	plt.yscale('symlog')
	plt.xlim([0.0, np.pi])
	plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
	plt.gca().set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
	plt.legend(fontsize=14)
	ax = plt.gca()
	ax.tick_params(which='minor', length=2)
	plt.savefig('figures/correlation.pdf')


	# plot Dl spectrum
	
	plt.figure()
	plt.scatter(l, Dl, 
		s=6.0, alpha=0.6, marker='o', c='#D1495B', linewidths=0.0)
	plt.fill_between(l, Dl + 2*Dlerr[:, 1], Dl - 2*Dlerr[:, 0], 
		linewidth=0.0, alpha=0.3, color='#419D78')
	plt.xlim([2.0, 2500.0])
	plt.ylim([0.0, 7500.0])
	plt.xlabel(r'$\ell$')
	plt.ylabel(r'$\mathcal{D}_\ell$')
	plt.legend([r'Planck Data', r'$2\sigma$ Error Band'], fontsize=14)
	#plt.title(r'$TT$ CMB Power Spectrum')
	plt.savefig('figures/TTspectrum.pdf')
	

	# plot Cl spectrum
	"""	
	plt.figure(figsize=(7, 5))
	plt.scatter(l, Cl, 
		s=6.0, alpha=0.6, marker='o', c='#D1495B', linewidths=0.0)
	plt.xlim([2.0, 2500.0])
	axes = plt.axis()
	plt.ylim([0.0, axes[3]])
	plt.xscale('log')
	plt.xlabel(r'$\ell$')
	plt.ylabel(r'$C_\ell$')
	plt.legend([r'Planck Data', r'$2\sigma$ Error Band'])
	plt.title(r'$TT$ CMB Power Spectrum')
	plt.savefig('figures/TTClspectrum.pdf')
	"""