# -*- coding: utf-8 -*-
"""
Auxiliary plotting functions that will be used in the notebooks

Andrey Kravtsov (Spring 2022)
"""

# import relevant packages
import matplotlib.pyplot as plt
import numpy as np 

# the following commands make plots look better
def plot_prettier(dpi=200, fontsize=10, usetex=False): 
    '''
    Make plots look nicer compared to Matplotlib defaults
    Parameters: 
        dpi - int, "dots per inch" - controls resolution of PNG images that are produced
                by Matplotlib
        fontsize - int, font size to use overall
        usetex - bool, whether to use LaTeX to render fonds of axes labels 
                use False if you don't have LaTeX installed on your system
    '''
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
    # if you don't have LaTeX installed on your laptop and this statement 
    # generates error, comment it out
    plt.rc('text', usetex=usetex)

    
from math import factorial
def exp_taylor(x0, N, x):
    '''
    Taylor expansion up to order N for exp(x)
    '''
    dummy = np.zeros_like(x)
    for n in range(N+1):
        dummy += np.exp(x0)*(x-x0)**n/factorial(n)
    return dummy


def taylor_exp_illustration(figsize=3.0):
    '''
    Function to produce an illustration for the Taylor expansion approximation 
    for the exp(x) function
    
    Parameters
    ----------
    figsize : float, Matplotlib figure size 

    Returns
    -------
    None.

    '''
    N = 4; x0 = 1.0

    plt.figure(figsize=(figsize,figsize))
    #plt.title('Taylor expansion of $e^x$ at $x_0=%.1f$'%x0, fontsize=9)
    plt.xlabel('$x$'); plt.ylabel(r'$e^x, f_{\rm Taylor}(x)$')

    xmin = x0 - 1.0; xmax = x0 + 1.0
    x = np.linspace(xmin, xmax, 100)
    plt.xlim([xmin,xmax]); plt.ylim(0.,8.)

    exptrue = np.exp(x)
    plt.plot(x, exptrue, linewidth=1.5, c='m', label='$e^x$')
    colors = ['darkslateblue', 'mediumslateblue', 'slateblue', 'lavender']
    lstyles = [':','--','-.','-','-.']
    for n in range(N):
        expT = exp_taylor(x0, n, x)
        plt.plot(x, expT, linewidth=1.5, c=colors[n], ls=lstyles[n], label='%d term(s)'%(n+1))

    plt.legend(loc='upper left', frameon=False, fontsize=7)
    plt.show()
    return

