import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from math import pi,e
from scipy.integrate import romberg
from astropy.cosmology import LambdaCDM
import astropy.units as u
from astropy.io import fits
from scipy import constants
import warnings
import sys
from os.path import dirname
from os.path import join
import bz2
try:
    import cPickle as pickle
except:
    import pickle
import urllib.request

def load_lightcurve_set():
    """
    Return the set of light curves for testing pdtrend.
    Returns
    -------
    lcs : numpy.ndarray
        An array of light curves.
    """

    # The light curves are bzipped and pickled.
    #file_path = 'data/lc.pbz2'
    file_path = 'http://astro.uchicago.edu/~andrey/classes/a211/data/lc.pbz2'
    # For Python 3.
    if sys.version_info.major >= 3:
        #lcs = pickle.load(bz2.BZ2File(file_path, 'r'), encoding='bytes')
        lcs = pickle.load(bz2.BZ2File(urllib.request.urlopen(file_path)), encoding='bytes')

    # For Python 2.
    else:
        lcs = pickle.load(bz2.BZ2File(file_path, 'r'))

    return lcs

lcs = load_lightcurve_set()

# %% codecell
def plot_pretty(dpi=200,fontsize=10):
    plt.rcParams['figure.dpi']= dpi
    plt.rc("savefig", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in')
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5)
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [2., 2.])
    plt.rc('legend',fontsize=5)
    plt.rcParams['figure.figsize'] = [5, 3]

plot_pretty()

def poly_fit(xtr, ftr,  method='polynomial', porder=None, s=0.):
    assert method in ['polynomial','splint','splreg']
    if method == 'polynomial':
        assert porder != None
        if type(porder) is str:
            assert porder == 'interp'
            porder  = xtr.size - 1
        else:
            assert 0 <= porder and porder < xtr.size
        if porder == xtr.size - 1:
            coefficients = np.transpose(np.array([xtr**i for i in range(0,xtr.size)]))
            solved = np.linalg.solve(coefficients,ftr)
            return np.poly1d(solved[::-1])
        else:
            return np.poly1d(np.polyfit(xtr,ftr,porder))

    elif method == 'splint':
        assert porder in [0,1,2,3]
        return scipy.interpolate.interp1d(xtr,ftr,kind = ['zero','slinear','quadratic','cubic'][porder])
    elif method == 'splreg':
        if porder == None:
            porder = 3
        assert 1 <= porder and porder <= 5
        return scipy.interpolate.UnivariateSpline(xtr,ftr,k=porder,s=s)
# %% codecell
times = np.ones(lcs.shape) * np.arange(lcs.shape[1])

ilc = 1
x,y = times[ilc], lcs[ilc]

poly_regression = poly_fit(x,y,method="polynomial",porder=3)
spline_regression = poly_fit(x,y,method="splreg",s=1000)

# %% codecell
plt.figure(figsize=(3,3))
plt.xlabel('time index')
plt.ylabel('brightness (arbitrary units)')
plt.scatter(x,y,s=0.2,zorder=0)
plt.plot(x,poly_regression(x),color="red",zorder=1)
plt.show()
# %% codecell
plt.figure(figsize=(3,3))
plt.xlabel('time index')
plt.ylabel('brightness (arbitrary units)')
plt.scatter(x,y-poly_regression(x),s=0.2,zorder=0)
plt.show()
# %% codecell
spline_regression = poly_fit(x,y,method="splreg",s=1)
plt.figure(figsize=(3,3))
plt.xlabel('time index')
plt.ylabel('brightness (arbitrary units)')
plt.scatter(x,y,s=0.2,zorder=0)
plt.plot(x,spline_regression(x),color="red",zorder=1)
plt.show()
# %% codecell
plt.figure(figsize=(3,3))
plt.xlabel('time index')
plt.ylabel('brightness (arbitrary units)')
plt.scatter(x,y-spline_regression(x),s=0.2,zorder=0)
plt.show()
# %% markdown
