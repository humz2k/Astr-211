# %% codecell
import numpy as np
from math import e
import matplotlib.pyplot as plt
from scipy.integrate import romberg
from astropy.cosmology import LambdaCDM
import astropy.units as u
from scipy import constants

plt.rcParams['figure.figsize'] = [8, 8]

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

def integrate(func,a,b,*args,rtol=1e-6,mul=2):
    start = 1
    err = np.inf
    next = trapzd(func,a,b,start*mul,*args)
    last_err = np.inf
    while err > rtol:
        start = start*mul
        j = next
        next = trapzd(func,a,b,start*mul,*args)
        r1_val = (4*next - j) / 3
        if r1_val == 0:
            err = 0
        else:
            err = np.abs(1. - j/r1_val)
        if err > last_err:
            return r1_val,err
        last_err = err
    return r1_val,err
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
    print(f'for rtol = {rtol:.0e}, my frac error = {my_err:.5e}, scipy error = {sci_err:.5e}')

# %% markdown
### Exercise 2
# %% codecell

int_vector = np.vectorize(integrate)

def d_l_astropy(z, H0=70.0, Om0=0.3, OmL=0.7):
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=OmL)
    return cosmo.luminosity_distance(z=z)

def d_l(z,rtol=1.e-6,H0=70.0,Om0=0.3,OmL=0.7,clight=None):
    units = False
    if clight == None:
        try:
            clight = (constants.speed_of_light * (u.meter/u.second)).to(u.km/u.second)
            H0 = H0 * ((u.km/u.second)/u.Mpc)
        except:
            clight = 2.99792458e5


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
        elif OmK < 0:
            dc = (1/np.sqrt(np.abs(OmK))) * np.sin(np.abs(OmK) * dc)

        return dc * (clight/H0) * (1+z)

# %% codecell

H0 = 70.
m_s = u.meter/u.second
km_s = u.km/u.second
clight = constants.speed_of_light * (u.meter/u.second)
clight = clight.to(km_s)
zmin, zmax, nz = 0,3,20
z = np.linspace(zmin,zmax,nz)

vals = [(0.3,0.7),(0.1,0.9),(0.8,0.1)]

for Om0,OmL in vals:
    OmK = 1 - Om0 - OmL
    print("#############################")
    print(f'Om0 = {Om0:.1f}, OmL = {OmL:.1f}, OmK = {OmK:.1f}')
    astropy_vals = d_l_astropy(z,H0=H0,Om0=Om0,OmL=OmL)
    my_vals = d_l(z,H0=H0,Om0=Om0,OmL=OmL,rtol=1.00e-8)
    for zd,my_val,astropy_val in zip(z,my_vals,astropy_vals):
        #print(zd,my_val,astropy_val)
        print(f'   z={zd:.3f};  astropy value = {astropy_val:>11.5f};  my value = {my_val:>11.5f}')


# %% codecell
def read_jla_data(sn_list_name = None):
    """
    read in table with the JLA supernova type Ia sample

    Parameters
    ----------
    sn_list_name: str
        path/file name containing the JLA data table in ASCII format

    Returns
    -------
    zsn, msn, emsn - numpy float vectors containing
                       zsn: SNIa redshifts in the CMB frame
                       msn, emsn: apparent B-magnitude and its errors
    """
    zsn, msn, emsn = np.loadtxt(sn_list_name, usecols=(1, 4, 5),  unpack=True)

    return zsn, msn, emsn

zsn, msn, emsn = read_jla_data(sn_list_name = 'https://astro.uchicago.edu/~andrey/classes/a211/data/jla_lcparams.txt')
nsn = np.size(zsn)
print("read sample of %d supernovae..."%(nsn))

# %% codecell

def calculate_pred(z,M,H0=70.0,Om0=0.3,OmL=0.7,rtol=1.e-8):
    dl = d_l(z,H0=H0,Om0=Om0,OmL=OmL,rtol=rtol)
    try:
        dl = dl/u.Mpc
    except:
        pass
    return M + 5*np.log10(dl) + 25

def get_mean_bins(zsn,msn,nbins=20):
    zmin,zmax = np.min(zsn),np.max(zsn)

    bin_size = (zmax-zmin)/nbins

    bins = np.arange(bin_size,zmax,bin_size)

    means = []
    to_check = []
    for i in bins:
        smaller = zsn < i+bin_size/2
        bigger = zsn[smaller] >= i-bin_size/2
        if len(msn[smaller][bigger]) != 0:
            to_check.append(i)
            means.append(np.mean(msn[smaller][bigger]))

    return np.array(to_check),np.array(means)

def find_best_values(zsn,msn,pred_func,
                    start_m=0,end_m=-25,step_m=-1,
                    start_Om0=0,end_Om0=1,nsteps_Om0=10,
                    start_OmL=0,end_OmL=1,nsteps_OmL=10,
                    nsteps=None):
    if nsteps != None:
        nsteps_Om0 = nsteps
        nsteps_OmL = nsteps
    zs,means = get_mean_bins(zsn,msn)
    best_diff = np.inf
    best_M = 0
    best_Om0 = 0
    best_OmL = 0
    for OmL in np.linspace(start_OmL,end_OmL,nsteps_OmL):
        for Om0 in np.linspace(start_Om0,end_Om0,nsteps_Om0):
            found_better = False
            for M in np.arange(best_M,end_m+step_m,step_m):
                pred = pred_func(zs,M,Om0=Om0,OmL=OmL)
                diff = np.sum((means-pred)**2)
                if diff < best_diff:
                    found_better = True
                    best_M = M
                    best_diff = diff
                    best_Om0 = Om0
                    best_OmL = OmL
                else:
                    break
            if not found_better:
                break

    return best_M, best_Om0, best_OmL

M, Om0, OmL = find_best_values(zsn,msn,calculate_pred,nsteps=30)

zs_plot = np.linspace(np.min(zsn),np.max(zsn),100)

fig, (plot) = plt.subplots(1,1)
plot.scatter(zsn,msn,s=0.2,label="JLA Data",color="green",alpha=0.8)
plot.plot(zs_plot,calculate_pred(zs_plot,M,Om0=Om0,OmL=OmL),label="Model",color="red",alpha=0.8)
plot.legend()
plot.set_xlabel("Z")
plot.set_ylabel("Magnitude")
plot.set_title(f'Model: M={M:.0f},Om0={Om0:.2f},OmL={OmL:.2f}')
fig.suptitle("Magnitude vs Redshift for Supernovas from Betoule et al.",fontsize=20)
fig.tight_layout()
plt.show()


#calculate_pred(zsn,25)

# %% codecell
