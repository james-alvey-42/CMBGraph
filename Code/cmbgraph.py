import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

if __name__ == '__main__':
	fullsky100 = hp.fitsfunc.read_map('fullsky100.fits')
	mask = hp.fitsfunc.read_map('intensitymask.fits')
	plt.set_cmap('seismic')
	hp.visufunc.mollview(fullsky100)
	plt.show()