# %% codecell
import numpy as np
from math import e
import matplotlib.pyplot as plt
from scipy.integrate import romberg
from astropy.cosmology import LambdaCDM
import astropy.units as u

# use jupyter "magic" command to tell it to embed plot into the notebook
#%matplotlib inline

# %% markdown
### Exercise 1
# %% codecell
def r1(func, a, b, nsteps,*args):
    return (4*trapzd(func, a, b, 2*nsteps,*args) - trapzd(func, a, b, nsteps,*args)) / 3

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
    last = trapzd(func,a,b,start,*args)
    while err > rtol:
        start = start*2
        j = trapzd(func,a,b,start)
        err = np.abs(1. - j/r1(func,a,b,start,*args))
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

# %% markdown
### Exercise 2
# %% codecell

int_vector = np.vectorize(integrate)

def d_l_astropy(z, H0=70.0, Om0=0.3, OmL=0.7):
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=OmL)
    return cosmo.luminosity_distance(z=z) / u.Mpc

def d_l(z,rtol=1.e-6,H0=70.0,Om0=0.3,OmL=0.7,clight=2.99792458e5):
    if OmL == 0:
        assert(Om0 >= 0)
        q0 = 0.5 * Om0
        q0z = q0 * z
        return clight * z/H0 * (1. + (z-q0z) / (1. + q0z + np.sqrt(1. + 2.*q0z)))
    else:
        OmK = 1 - Om0 - OmL
        def f(x):
            E = np.sqrt(Om0*((1+x)**3) + OmK*((1+x)**2) + OmL)
            return 1/E
        dc,err = int_vector(f,0,z,rtol=rtol)

        if OmK > 0:
            dc = (1/np.sqrt(OmK)) * np.sinh(np.sqrt(OmK) * dc)

        return dc * (clight/H0) * (1+z)



z = np.arange(0,3,0.1)
print(d_l(z))
print(d_l_astropy(z))
