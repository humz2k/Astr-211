# %% codecell
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM
import astropy.units as u
import scipy.interpolate

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
# %% codecell
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
    '''
    Compute values of the 2D polynomial given the coefficients in 1d array a
    at points given by 2d arrays xtest and ytest (generated using meshgrid)
    '''
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
ntrain = 37

frac_err = []
zs = np.linspace(min_z,max_z,nzs+1)[1:]

for z in zs:
    frac_err.append(np.nanmax(get_frac_err(ntrain=ntrain,z=z,spacing=spacing,kx=5,ky=5,fit_type="spline")))

# %% codecell
plt.plot(zs,[1e-4]*len(zs),label="$1\times10^4$")
plt.scatter(zs,frac_err)
plt.legend()
plt.show()

# %% markdown
# **Task 1a. (4 points)**  Use functions above that generate 2D grids of d_L_tilde values. Try different numbers of training points along 1 dimension for the 2d spline approximation for $\tilde{d}_L$ for the ranges $\Omega_{\rm m0}=[0,1]$ and $\Omega_{\Lambda}=[0,1]$ and  find the minimal number of training points for which we can ensure the fractional error of $\tilde{d}_L$ $<10^{-4}$  for the entire range of $z\in [0,2]$.
#
# You should
#
# * Report the minimal number of training points that you find.
#
# * Present a calculation or a plot that demonstrates that fractional error is smaller than $10^{-4}$ for the entire range of  $z\in [0,2]$.
#
# **Note:** There are several different SciPy routines that can be used for this.  I recommend using function <tt>scipy.interpolate.RectBivariateSpline(x, y, z, s=0, kx=3, ky=3)</tt>, where $z$ is the array of function values tabulated at training points in vectors $x$ and $y$, $s=0$ indicates interpolation (no smoothing), parameters <tt>kx=3, ky=3</tt> specify that cubic splines should be used in $x$ and $y$ variables. Example of using this function is shown below (you can read about other available options and see examples of how they are used <a href="https://mmas.github.io/interpolation-scipy">here</a>). Examples of how this function is used to construct 2D spline approximation is available in [08_multid_optimization_class](https://drive.google.com/file/d/1-ptIvIvbRqtObk8x09ausJanXcqL8pJ0/view?usp=sharing) notebook.
#
# %% markdown
# **Task 1b (6 points).** Using functions <tt>polyfit2d</tt> and <tt>poly2d</tt> below construct 2D polynomial approximations of $\tilde{d}_L(z,\Omega_{\rm m0}, \Omega_\Lambda)$ for a given input single value of redshift $z$ and for ranges of the $\Omega_{\rm m0}$ and $\Omega_\Lambda)$ parameters of $\Omega_{\rm m0}\in [0,1]$ and $\Omega_\Lambda\in[0,1]$.
#
# Try different number of training points and polynomial order and try to find the minimal number and order that ensures the target fractional accuracy of $<10^{-4}$ for any $z$ in the interval $z\in [0,2]$.
#
# You should:
#
# * Report the minimal number of training points that you find.
#
# * Present a calculation or a plot that demonstrates that fractional error is smaller than $10^{-4}$ for the entire range of  $z\in [0,2]$.
#
#
# Based on the results of 1a and 1b, state your conclusions about the optimal method for approximating $\tilde{d}_L(z,\Omega_{\rm m0}, \Omega_\Lambda)$ with this target accuracy.
# %% codecell
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
    '''
    Compute values of the 2D polynomial given the coefficients in 1d array a
    at points given by 2d arrays xtest and ytest (generated using meshgrid)
    '''
    order1 = np.rint(a.size**0.5).astype(int)
    return np.polynomial.polynomial.polyval2d(xtest, ytest, a.reshape((order1,order1)))

# %% codecell
def polyfit2d(xtr, ytr, ftr, kx=3, ky=3, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resulting fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters:
    ----------
    xtr, ytr: array-like, 1d
        xtr and ytr coordinates.
    ftr: 2d numpy array
        f(xgtr, ygtr) values evaluated on meshgrid of xtr and ytr vectors to fit by polynomial
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns:
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    x, y = np.meshgrid(xtr, ytr)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    V = np.zeros((coeffs.size, x.size))

    # construct Vandermonde matrix: for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        V[index] = arr.flatten()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(V.T, np.ravel(ftr), rcond=None)[0]
# %% codecell

# %% markdown
# ## <font color='blue'>Exercise 2: implementing and testing  Differential Evolution algorithm for minimization (25 points + 5 extra-credit points)</font>
#
# **Background.** Minimization in many dimensions is generally a complicated task. However, a class of <a href="https://en.wikipedia.org/wiki/Differential_evolution">Differential Evolution</a> (DE) algorithms developed from the initial ideas of R. Storn and K. Price in 1997 (<a href="https://link.springer.com/article/10.1023%2FA%3A1008202821328">Storn & Price 1997</a>), are relatively simple to implement, work in arbitrary number of dimensions, do not require function derivatives, allow imposing bounds on the domain, and are quite efficient in minimizing multi-dimensional functions.
#
# ### <font color='blue'>What you are learning in this exercise:</font>
#
# * how to implement a general multi-dimensional minimization DE algorithm
# * how to find minimum of a function in practice.
# %% markdown
# The simplest version of the differential evolution algorithm described in detail in the notebook [08_optimization](https://drive.google.com/file/d/1oq838Jla7r6upwYf1uE7Oa6ctua0gIqU/view?usp=sharing),  can be presented as the following pseudo-code:
#
#     npop = np.size(x0)[0] # the number of population members
#     fnow = func(xnow)
#     xnow = np.copy(x0)
#     xnext = np.zeros_like(xnow)
#     ....
#     while some convergence criterion is not met:
#         # xnow is a vector of coordinate vectors of the current population
#         # xnext is a vector of coordinate vector of the next gen population
#         for i in range(npop):
#             # generate random unique indices  ir1, ir2, ir3
#             # where all indices are not equal to each other and not equal to i
#             # s can be a constant for large npop, but it's more safe to make it a
#             # random number drawn from uniform distribution in the range [smin,1]
#             xtry = xnow[ir3] + s * (xnow[ir1] - xnor[ir2])
#             if func(xtry) <= fnow[i]:
#                 xnext[i] = xtry
#             else:
#                 xnext[i] = xnow[i]
# %% markdown
#
#
# %% markdown
# **Task 2a. (20 points)** Use pseudo-code of the DE algorithm above to implement DE minimization function with the following interface (15 points):
#
#     def minimize_de(func, x0, atol=1.e-6, s=0.1, bounds=None):
#         """
#         Parameters:
#         ------------
#         func - Python function object
#                function to minimize, should expect x0 as a parameter vector
#         x0   - vector of real numbers of shape (npop, nd),
#                 where npop is population size and nd is the number of func parameters
#         atol - float
#                 absolute tolerance threshold for change of population member positions
#         s    - float
#                s parameter for scaling steps, the step size will be dwarf from uniform distribution between s and 1
#         bounds - array of tuples
#                 bounds for the minimization exploration; define the region in which to search for the minimum
#         """
#
#
# ***Note:*** guard against for the cases when the small number of population members is used when population does not move at a given mutation stage, so that this does not result in premature stopping of the algorithm.
#
# ***Note:*** Try to "vectorize" as much of the algorithm as possible. This code can be fully vectorized with only one loop for the mutations of the population.
#
# %% markdown
# ***Note:*** Assuming that we are searching for a minimum within some rectangular domain defined by the minimum and maximum values along each coordinate axis: $\mathbf{x}_{\rm min}$ and $\mathbf{x}_{\rm max}$, we can initialize the population members as
#
# $$\mathbf{x}_0 = \mathbf{x}_{\rm min} + (\mathbf{x}_{\rm max}-\mathbf{x}_{\rm min}) \cdot\mathrm{rand}(0,1),$$
#
# where $\mathrm{rand}(0,1)$ is a random number uniformly distributed from 0 to 1, generated using <tt>np.random.uniform</tt>.
#
# ***Note:*** the algorithm requires selection of 3 random indices of members, excluding the current member that is being mutated. As always, there are multiple ways of doing this in Python. Below is one example of how this can be done using NumPy functions
# %% codecell
npop = 10 # number of population members
inds = np.arange(npop) # create a list of indices from 0 to npop-1
inds = np.delete(inds,7) # remove specific index 7 from inds
np.random.shuffle(inds) # shuffle indices randomly
print(inds[0], inds[1], inds[2]) # print the first 3 of the shuffled indices
# %% markdown
# ***Таск 2b (5 points).*** Test your implementation using Rosenbrock function implemented below in 2- and 5-dimensions. Try different number of population members and $s$ values and choices for how $s$ is chosen and examine how results change and for what number of population members the algorithm returns correct minimum value reliably ($[1,1]$ in 2D and $[1, 1, 1, 1, 1]$ in 5D).
#
# * Present a brief discussion of how large population should be in 2D and 5D to get correct minimum reliably.
#
# * Present a brief discussion of how choices of $s$ affect results
#
# * Demonstrate that your function returns values within the specified atol value reliably in 5D.
#
# * Compare results of your function to results of the <tt>scipy.optimize.differential_evolution</tt>
# %% codecell
def rosenbrock(x):
    """The Rosenbrock "banana" function
    x is a vector of points in 2 or more dimensional space
    """
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

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
