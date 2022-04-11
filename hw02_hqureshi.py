# %% codecell
import numpy as np
from math import e
import matplotlib.pyplot as plt
from scipy.integrate import romberg

# use jupyter "magic" command to tell it to embed plot into the notebook
#%matplotlib inline

# the following commands make plots look better
def plot_prettier(dpi=200, fontsize=10):
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
    #plt.rc('text', usetex=True)

#plot_prettier()
# %% codecell
def r1(func, a, b, nsteps):
    return (4*trapzd(func, a, b, 2*nsteps) - trapzd(func, a, b, nsteps)) / 3

def trapzd(func,a,b,nsteps,*args):
    h = (b-a)/nsteps

    if nsteps == 1:
        return 0.5*(func(a, *args) + func(b, *args)) * h
    else:
        xd = a + np.arange(1,nsteps) * h
        return (0.5*(func(a, *args) + func(b, *args)) + np.sum(func(xd, *args))) * h

def integrate(func,a,b,*args,rtol=1e-6):
    start = 1
    err = np.inf
    last = trapzd(func,a,b,start)
    tries = 1
    while err > rtol:
        tries += 1
        start = start*2
        j = trapzd(func,a,b,start)
        err = np.abs(1. - j/r1(func,a,b,start))
        #err = np.abs(j/last - 1.)
        if err > last:
            return j,err
        last = err
    return j,err

# %% codecell

a, b = 0, 1
func = np.exp
exact = func(b) - func(a)
rtols = [1*10**-i for i in range(2,15,2)]
for rtol in rtols:
    my_val,est_err = integrate(func,a,b,rtol=rtol)
    my_err = np.abs(my_val/exact -1.)
    sci_val = romberg(func,a,b,tol=rtol)
    sci_err = np.abs(sci_val/exact -1.)
    print(f'for rtol = {rtol:.2e}, my frac error = {my_err:.5e}, scipy error = {sci_err:.5e}')

# %% codecell
