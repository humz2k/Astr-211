# %% codecell
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM
import astropy.units as u
import scipy.interpolate
from scipy import stats

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
# %% codecell
def d_l_tilde_astropy(z, H0, Om0, OmL, clight=2.99792e5):
    '''
    compute d_l_tilde using AstroPy d_L function

    Parameters:

    z - float, or numpy array, redshift
    H0 - float, Hubble constant in km/s/Mpc
    Om0, OmL - floats, dimensionless matter and dark energy densities

    Returns:

        d_L - float, or numpy array, rescaled by c/H0 in Mpc
    '''
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=OmL)

    return cosmo.luminosity_distance(z=z) / u.Mpc / (clight/H0)

def get_dl_train_test(ntrain=15, ntest=100, z=1.0, H0=70.,
                      om0min=0., om0max = 1., omlmin=0., omlmax=1., spacing=np.linspace):

    om0tr = spacing(om0min, om0max, ntrain)
    omltr = spacing(omlmin, omlmax, ntrain)

    dl_train = np.zeros((ntrain, ntrain)) # initialize 2D numpy array for 2D grid of d_L values
    # Now cycle through Om0 and OmL values, compute d_L and fill the dlgrid array with values
    for i, omd in enumerate(om0tr):
        for j, omld in enumerate(omltr):
                dl_train[i,j] = d_l_tilde_astropy(z, H0, omd, omld)

    # test points
    om0t = np.linspace(om0min, om0max, ntest)
    omlt = np.linspace(omlmin, omlmax, ntest)

    dl_test = np.zeros((ntest, ntest)) # initialize 2D numpy array for 2D grid of d_L values
    # Now cycle through Om0 and OmL values, compute d_L and fill the dlgrid array with values
    for i, omd in enumerate(om0t):
        for j, omld in enumerate(omlt):
                dl_test[i,j] = d_l_tilde_astropy(z, H0, omd, omld)

    return om0tr, omltr, om0t, omlt, dl_train, dl_test
# %% markdown
## Question 1a
# %% codecell
def chebyshev_nodes1(a, b, N):
    return (a + 0.5*(b-a)*(1. + np.cos((2.*np.arange(N)+1)*np.pi/(2.*(N+1)))))[::-1]

def chebyshev_nodes2(a, b, N):
    return (a + 0.5*(b-a)*(1. + np.cos(np.arange(N)*np.pi/N)))[::-1]

def polyfit2d(xtr, ytr, ftr, order=None):
    '''
    Parameters:
        xtr, ytr - 1d numpy vectors with training points of x and y
        ftr - function values at xtr, ytr values
        order - int, order of the polynomial

    Returns:
        coefficients of the 2D polynomial
    '''
    # generate 2d coordinates on a rectangular grid
    x, y = np.meshgrid(xtr, ytr)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((order+1, order+1))
    # array that will contain polynomial term values
    s = np.zeros((coeffs.size, x.size))

    # construct the 2D matrix of values for each polynomial term i, j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = x**i * y**j # coeffs[i, j] *
        s[index] = arr.flatten()

    # solve for the polynomial coefficients using least squares approximation of ftr values
    return np.linalg.lstsq(s.T, np.ravel(ftr), rcond=None)[0]

def poly2d(xtest, ytest, a):
    order1 = np.rint(a.size**0.5).astype(int)
    return np.polynomial.polynomial.polyval2d(xtest, ytest, a.reshape((order1,order1)))

class poly2dfunc:
    def __init__(self,a):
        self.a = a
        self.order1 = np.rint(a.size**0.5).astype(int)

    def __call__(self,xtest,ytest):
        return np.polynomial.polynomial.polyval2d(xtest, ytest, self.a.reshape((self.order1,self.order1)))

def train_model(ntrain=15, ntest=100, z=1.0, H0=70.,om0min=0., om0max = 1., omlmin=0., omlmax=1.,s=0,kx=3,ky=3,spacing=np.linspace,fit_type="spline",order=5):
    om0tr, omltr, om0t, omlt, dl_train, dl_test = get_dl_train_test(ntrain=ntrain, ntest=ntest, z=z, H0=H0, om0min=om0min, om0max=om0max, omlmin=omlmin, omlmax=omlmax, spacing=spacing)
    testing = {"om0t":om0t,"omlt":omlt,"dl_test":dl_test}
    if fit_type == "spline":
        return scipy.interpolate.RectBivariateSpline(om0tr, omltr, dl_train, s=s, kx=kx, ky=ky),testing
    elif fit_type == "poly":
        return poly2dfunc(polyfit2d(om0tr,omltr,dl_train,order=order)), testing

def get_frac_err(**kwargs):
    f,test_data = train_model(**kwargs)
    return f(test_data["om0t"],test_data["omlt"])/test_data["dl_test"] -1.

spacing = np.linspace
min_z = 0
max_z = 2
nzs = 10
ntrain_a = 37
ntrain_b = 100

frac_err_b = []
frac_err_a = []
zs = np.linspace(min_z,max_z,nzs+1)[1:]

for z in zs:
    frac_err_a.append(np.nanmax(get_frac_err(ntrain=ntrain_a,z=z,spacing=spacing,kx=5,ky=5,fit_type="spline")))

for z in zs:
    frac_err_b.append(np.nanmax(get_frac_err(ntrain=ntrain_b,z=z,spacing=spacing,kx=5,ky=5,fit_type="poly")))

# %% codecell
plt.plot(zs,[1e-4]*len(zs),label="Acceptable Error: $1e-4$",color="green",linewidth=1)
plt.scatter(zs,frac_err_a,s=4,label="Cal Frac Error")
plt.xlabel("z")
plt.ylabel("Frac Error")
plt.legend(loc = "upper left",framealpha=1)
plt.show()
# %% markdown
## Question 1b
# %% codecell
plt.plot(zs,[1e-4]*len(zs),label="Acceptable Error: $1e-4$",color="green",linewidth=1)
plt.scatter(zs,frac_err_b,s=4,label="Cal Frac Error")
plt.xlabel("z")
plt.ylabel("Frac Error")
plt.legend(loc = "upper left",framealpha=1)
plt.show()

# %% codecell
def rosenbrock(x):
    """The Rosenbrock "banana" function
    x is a vector of points in 2 or more dimensional space
    """
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def minimize_de(func=rosenbrock, x0=None, atol=1.e-6, s=0, smin=None, bounds=None):
    assert type(x0) is not None
    npop = x0.shape[0]
    nd = x0.shape[1]
    xnow = np.copy(x0)
    fnow = np.empty(npop)
    for i in range(npop):
        fnow[i] = func(xnow[i])
    xnext = np.zeros_like(xnow)
    has_bounds = True
    if isinstance(bounds, type(None)):
        has_bounds = False
    else:
        bounds = bounds.T
        bounds[1] -= 1

    while True:
        inds = np.ma.arange(npop)
        delta = 0
        for i in range(npop):
            inds[i] = np.ma.masked
            while True:
                ir1,ir2,ir3 = np.random.choice(inds.compressed(),3,replace=False)
                inds.mask = np.ma.nomask
                if smin != None:
                    s = np.random.uniform(smin,1)
                xtry = xnow[ir3] + s * (xnow[ir1] - xnow[ir2])
                if has_bounds:
                    if np.all(np.logical_xor(*((xtry-bounds) > 0))):
                        break
                else:
                    break
            f_xtry = func(xtry)
            if f_xtry <= fnow[i]:
                temp_delta = np.abs(fnow[i] - f_xtry)
                xnext[i] = xtry
                fnow[i] = f_xtry
                if temp_delta > delta:
                    delta = temp_delta
            else:
                xnext[i] = xnow[i]
        xnow = np.copy(xnext)
        if delta < atol:
            break
    out = []
    for i in range(xnow.shape[1]):
        out.append(stats.mode(xnow[:,i])[0])
    return np.array(out).T[0]

def get_error(de_func=minimize_de,s=0,smin=0,npop=10,nd=2,bounds=np.array([(20,-20)]),known_minimum=np.array([1.,1.]),repeats=20):
    if nd != bounds.shape[0]:
        bounds = np.repeat(np.reshape(bounds[0],(1,bounds[0].shape[0])),nd,axis=0)
    high,low = np.repeat(bounds.T,npop,axis=1)
    avg_error = np.zeros(repeats,dtype=np.float64)
    for i in range(repeats):
        x0 = np.reshape(np.random.uniform(high=high,low=low),(npop,nd))
        minimum = de_func(x0 = x0,bounds = bounds,smin=smin,s=s)
        avg_error[i]=np.max(np.abs(minimum-known_minimum))
    return avg_error

def get_data(npops,repeats,verbose=True,**args):
    npops = np.array(npops)
    errors = np.zeros((len(npops),repeats))
    for idx,npop in enumerate(npops):
        if verbose:
            print(str(round((idx/len(npops))*100,1)) + "% done")
        errors[idx] = get_error(npop = npop, repeats = repeats, **args)
    return npops,errors

npops = (10**(np.linspace(1,2,5))).astype(np.int)
repeats = 20
R5s = {}
for smin in np.linspace(0,1,4):
    npops,errors = get_data(npops,repeats,known_minimum=np.array([1,1,1,1,1]),nd=5)
    R5s[smin] = (npops,errors)

R2s = {}
for smin in np.linspace(0,1,4):
    npops,errors = get_data(npops,repeats,known_minimum=np.array([1,1]),nd=2)
    R2s[smin] = (npops,errors)

# %% codecell

#print()
for smin in R5s.keys():
    npops,errors = R5s[smin]
    plt.scatter(np.repeat(npops,errors.shape[1]),errors.flatten(),s=1,alpha=0.5)
    plt.plot(npops,np.mean(errors,axis=1),label="smin="+str(smin))

plt.xscale('log')
plt.yscale('log')
plt.title("R5")
plt.legend()
plt.show()

# %% codecell

#print()
for smin in R2s.keys():
    npops,errors = R2s[smin]
    plt.scatter(np.repeat(npops,errors.shape[1]),errors.flatten(),s=1,alpha=0.5)
    plt.plot(npops,np.mean(errors,axis=1),label="smin="+str(smin))

plt.xscale('log')
plt.yscale('log')
plt.title("R2")
plt.legend()
plt.show()


    #print(np.max(np.abs(minimum-compare)))
#print(x0)

# %% markdown
# ***???????? 2b (5 points).*** Test your implementation using Rosenbrock function implemented below in 2- and 5-dimensions. Try different number of population members and $s$ values and choices for how $s$ is chosen and examine how results change and for what number of population members the algorithm returns correct minimum value reliably ($[1,1]$ in 2D and $[1, 1, 1, 1, 1]$ in 5D).
#
# * Present a brief discussion of how large population should be in 2D and 5D to get correct minimum reliably.
#
# * Present a brief discussion of how choices of $s$ affect results
#
# * Demonstrate that your function returns values within the specified atol value reliably in 5D.
#
# * Compare results of your function to results of the <tt>scipy.optimize.differential_evolution</tt>
# %% codecell

# %% markdown
# ## <font color='blue'> Exercise 3: implementing cross-over stage of the DE algorithm (extra-credit 10 points).</font>
#
#
# Implement the modification of DE with the cross-over stage of the algorithm described below  and test your implementation using the Rosenbrock pdf in 2- and 5 dimensions.  (7 points)
#
# Discuss what difference you can notice in the performance of the algorithm with the crossover stage compared to the simplest form in exercise 2 (3 points).
#
#
# **Cross-over stage of the DE algorithm**. One of the modifications to this basic algorithm is introduction of the ***crossover stage*** so that the mutation and crossover stages together are as follows:
#
# * compute mutation vector $\mathbf{x}^\prime_i=\mathbf{x}_{{\rm now}, r_3} + s\,(\mathbf{x}_{{\rm now}, r_2}-\mathbf{x}_{{\rm now}, r_1})$, as before, where vector $\mathbf{x}^\prime_i$ has components $\mathbf{x}^\prime_i=[x^{\prime}_{0i}, x^{\prime}_{1i}, \ldots, x^{\prime}_{(D-1)i}]$, and $D$ is the number of parameters of the minimized function (i.e., dimensionality of the problem).
#
# * "***crossover stage***": form the trial vector $\mathbf{x}^{\prime\prime}_i=[x^{\prime\prime}_{0i}, x^{\prime\prime}_{1i}, \ldots, x^{\prime\prime}_{(D-1)i}]$, where
#
# \begin{equation}
# x^{\prime\prime}_{ji} =
# \begin{cases}
# x^{\prime}_{ji}, \ {\rm if\ }r_j\leq \mathrm{cr\ or\ } j= \mathrm{ir}_i,\\
# x_{{\rm now},ji}, \ {\rm otherwise\ }
# \end{cases}
# \end{equation}
#
# and $r_j$ is the random floating point number uniformly distributed in the interval $[0,1]$ that is generated for the index $j$, and $\mathrm{ir}_i$ is the random integer uniformly distributed in the range $[0, D-1]$ generated for index $i$, which ensures that $\mathbf{x}^{\prime\prime}_i$ gets at least one element from $\mathbf{x}^\prime_i$. The crossover parameter $\mathrm{cr}\in [0,1]$ is a constant set by user.
#
# * *Selection stage:* if $f(\mathbf{x}^{\prime\prime}_i)\leq f(\mathbf{x}_{{\rm now},i})$, then $\mathbf{x}_{{\rm next},i}=\mathbf{x}^{\prime\prime}_i$, else $\mathbf{x}_{{\rm next},i}=\mathbf{x}_{{\rm now},i}$ (no mutation).
#
