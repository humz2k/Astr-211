# %% codecell
import numpy as np

# use jupyter "magic" command to tell it to embed plot into the notebook
import matplotlib.pyplot as plt
%matplotlib inline

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
    plt.rc('text', usetex=True)

plot_prettier()
# %% markdown
# **Undergraduate students:** You can choose to do extra-credit exercise 4 instead of 1 for 10 extra-points. That exercise, requires implementation of the Romberg integration scheme using recursion of function calls.
#
# **Graduate students:** You should do exercise 4 instead of 1, as a default.
#
# ---------------------------------------------------------------------------------------------------------
# ## <font color='darkblue'>Exercise 1 (15 points): implementing numerical integration with error control
#
# Use examples provided in the <tt><a href="https://drive.google.com/file/d/1GYZ-plSXdInEGL4q_aNClTlH07A3GxtH/view?usp=sharing">04_integration_class</a></tt> notebook to implement a function that can numerically estimate integral for an input function over an interval $[a,b]$. The estimate must employ one or a combination of the approaches to ensure that the estimate returned by the function has fractional error smaller than the specified ``threshold'' (aka tolerance). A possible format of the function is outlined below. (10 points)
#
# Test your function by computing $$\int_0^1 e^x dx$$ and computing fractional error of the numerical relative to exact value ($=e-1$) and show that the fractional error of the estimate is smaller than the specified threshold, for several values of the threshold (e.g., 1e-6, 1e-8, 1e-10). Compare results to SciPy's function <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.romberg.html"><tt>scipy.integrate.romberg</tt></a> for the same values of the tolerance thresholds, as shown in the example below.
#
# Consider using functions <tt>trapzd</tt> and the <tt>r1</tt>, <tt>r2</tt>, <tt>r3</tt> that can be found  in the <tt><a href="https://drive.google.com/file/d/1GYZ-plSXdInEGL4q_aNClTlH07A3GxtH/view?usp=sharing">04_integration_class</a></tt> notebook. The latter functions use <tt>trapzd</tt> as the base for the Richardson extrapolation to obtain the 4th, 6th, and 8th order integration schemes.
#
#
#     def integrate(func, a, b, *args, rtol=1e-6):
#         '''
#         function computing numerical estimate of the integral of function func over the interval [a,b]
#         the integral estimate is guaranteed to be in the range [2e-16, rtol]
#
#         Parameters:
#         -----------------------------------------------------------
#         func: python function object
#               function to integrate, must take numpy arrays as input
#         a, b: floats
#               limits of the integral
#         rtol: float - the fractional error tolerance threshold
#
#         args: pointer to a list of parameters to be passed for func, if any
#
#         Returns:
#         -------------------------------------------------------------
#         value of the estimated int^b_a f(x)dx, estimate of the fractional and absolute error
#         '''
# %% markdown
# #### <font color='darkblue'>Note: estimating absolute and relative errors of the integral estimate
#
# As outlined in <tt><a href="https://drive.google.com/file/d/1GYZ-plSXdInEGL4q_aNClTlH07A3GxtH/view?usp=sharing">04_integration_class</a></tt>, if we only use a given estimator, such as trapezoidal scheme, we can view difference between integral estimates using step sizes $h$ and $h/2$ as an approximation of the integral error. If we use 2 estimators of different order, we can consider the difference between estimates using schemes for a given $h$ value as an approximation for the error, because one of the estimates is expected to be much more accurate (so is an approximation for the true value of the integral).
#
# For example, the absolute error of a trapezoidal estimate with step size $h=(b-a)/N$ can be estimated as
# $\epsilon_{\rm abs}=\vert R_1(h)-T(h)\vert$
# and fractional error as $\epsilon_{\rm r}=\vert1- T(h)/R_1(h)\vert$, where $R_1(h)$ is an estimator obtained by the first iteration of Richardson extrapolation (implemented in the function <tt>r1</tt>). Note that for the fractional error, we should guard against the cases when $R_1(h)=0$.
# %% markdown
# #### <font color='darkblue'>Example of using scipy.integrate.romberg
#
# %% codecell
from scipy.integrate import romberg

a, b = 0, 1
for tol in [1.e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
    ei  = romberg(np.exp, a, b, tol=tol) # tol is fractional error threshold
    fracerr = ei/(np.exp(b)-np.exp(a)) - 1.0
    print(f'for tol = {tol:.2e}   frac. error = {fracerr:.5e}')
# %% markdown
# ## <font color='darkblue'>Exercise 2 (10 points): implementing and testing a function to compute luminosity distance $d_L$
# %% markdown
#
# **2a (7 points).** Use the function you implemented in exercise 1 to implement a function to compute cosmological distance $d_L$ for a range of redshift values $z$ (for example, $z\in[0,3]$. Expressions for $d_L$ for different values of $\Omega_{\rm m0}$ and $\Omega_\Lambda$ are provided below.
#
#
#     def d_L(z, rtol=1.e-6, H0=70.0, Om0=0.3, OmL=0.7):
#         '''
#         Estimate luminosity distance for an object with redshift z and values of cosmological parameters, H0, Om0, OmL
#
#         Parameters:
#         -----------
#         z - float(s), a single redshift or a list/numpy array redshift values
#         rtol - fractional error tolerance to be passed to integrate function to ensure that fractional error of the
#                estimate is smaller than rtol (for rtol>2e-16)
#
#         H0  - float, keyword parameter holding default value of the Hubble constant in km/s/Mpc
#         Om0 - float, keyword parameter holding default value of the dimensionless mean matter density in the universe
#               (density in units of the critical density value). Default value is 0.3, reasonable range is [0,2]
#         OmL - float, keyword parameter holding default value of the dimensionless dark energy density in the universe
#               (density in units of the critical density value). Default value is 0.7.
#
#         Returns:
#         --------
#         d_L - float(s), a single estimate of d_L in Megaparsecs for a single or a vector d_L for input vector z
#         '''
#
# ***Note*** that there are two parts here: implementation (2a) and testing (2b).
#
# ***Note:*** If you did not manage to get the integration function working in exercise 1, you can use SciPy's function <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.romberg.html"><tt>scipy.integrate.romberg</tt></a>, as shown above, here to compute $d_L$.
#
# %% markdown
# #### Mathematical expressions for $d_L$
#
# Below are mathematical expressions for the distance of an object observed with spectral redshift $z$ assuming cosmological parameters: $H_0$ (Hubble constant), $\Omega_{\rm 0}$ (dimensionless mean matter density), $\Omega_\Lambda$ (dimensionless mean density of dark energy).
#
#
# #### $d_L$ for  models with $\Omega_\Lambda= 0$
# For the models with $\Omega_\Lambda=0$ the integral above does have mathematical solution in "closed form" (an equation can be written out for the integral) which gives the following expression:
#
# $$d_L = \frac{c}{H_0}\, z\left[1 + \frac{(1-q_0)z}{1+q_0z+\sqrt{1+2q_0z}}\right],$$
#
# where $q_0=\Omega_{\rm m0}/2$ is the *deceleration parameter* and $\Omega_{\rm m0}$ is the mean density of *matter* in the universe.
#
# #### $d_L$ for general models with $\Omega_\Lambda\ne 0$
#
# Denoting the integral we need to estimate as
# $$d_c =\int\limits_0^z \frac{dx}{E(x)}.$$
#
# where $E(x)=\sqrt{\Omega_{\rm m0}(1+x)^3+\Omega_k(1+x)^2+\Omega_\Lambda}.$
#
# expression for $d_L$ for general models with $\Omega_\Lambda\ne 0$:
#
# $$
# d_L(z, H_0,\Omega_{\rm m0},\Omega_\Lambda) = \frac{c}{H_0}\,(1+z)\,\left\{
# \begin{array}{ll}
# \frac{1}{\sqrt{\Omega_k}}\,\sinh\left[\sqrt{\Omega_k}\,d_{\rm c}\right] & {\rm for}~\Omega_k>0 \\
# d_{\rm c} & {\rm for}~\Omega_k=0 \\
# \frac{1}{\sqrt{|\Omega_k|}}\,\sin\left[\sqrt{|\Omega_k|}\,d_{\rm c}\right] & {\rm for}~\Omega_k<0
# \end{array}
# \right.
# $$
#
# where $\Omega_k = 1-\Omega_{\rm m0} - \Omega_\Lambda$, $c=2.99792458\times 10^5$ km/s is speed of light in km/s, $H_0$ is the Hubble constant in km/s/Mpc (current observations indicate that $H_0$ is close to $70$ km/s although values between 65 and 74 are possible.
#
# %% markdown
# Here are numpy functions that can be used to compute mathematical functions involved: absolute value <tt><a href="https://numpy.org/doc/stable/reference/generated/numpy.absolute.html">np.abs</a></tt> (this is shorthand for np.absolute), hyperbolic sine <tt><a href="https://numpy.org/doc/stable/reference/generated/numpy.sinh.html">np.sinh</a></tt>, sine <tt><a href="https://numpy.org/doc/stable/reference/generated/numpy.sin.html">np.sin</a></tt>.
# %% codecell
def d_L_no_de(z, H0, Om0, clight = 2.99792458e5):
    '''
    function estimating d_L in Mpc, works only for models with OmL = 0

    Parameters:
    -----------
        z - float(s), a float or a numpy vector of floats containing redshift(s) for which to compute d_L
        H0 - Hubble constant in km/s/Mpc
        Om0 - dimensionless mean matter density in units of the critical density

    Returns:
    --------
        d_L - float(s), float or numpy vector of floats containing d_L in Mpc for inpute value(s) of z

    '''
    assert(Om0 >=0)
    q0 = 0.5 * Om0
    q0z = q0 * z
    return clight * z/H0 * (1. + (z-q0z) / (1. + q0z + np.sqrt(1. + 2.*q0z)))
# %% markdown
# **2b (3 points).** Test your function by comparing it to the estimates using AstroPy function for several pairs of $\Omega_{\rm m0}$ and $\Omega_\Lambda$: $[0.3, 0.7]$, $[0.1,0.9]$, $[0.8,0.1]$, as shown below.
# %% codecell
from astropy.cosmology import LambdaCDM
import astropy.units as u

def d_l_astropy(z, H0, Om0, OmL):
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=OmL)
    return cosmo.luminosity_distance(z=z) / u.Mpc

Om0 = 0.3; OmL = 0.7; H0 = 70.

zmin, zmax, nz = 0, 3, 20
z = np.linspace(zmin, zmax, nz) # evenly spaced grid of z values
# astropy allows to process a vector of z values in one call
d_la = d_l_astropy(z, H0, Om0, OmL)

for i, zd in enumerate(z):
    # output using f-string formatting
    # add outout of d_L estimate using your integration func
    print(f'z = {zd:.3f};  distance_astropy = {d_la[i]:>11.5f} Mpc')
# %% markdown
# ## <font color='darkblue'>Exercise 3 (5 points): using $d_L$ calculation to compute distance modulus of supernovae Type Ia
#
# The code snippet below reads data from the study of <a href="https://ui.adsabs.harvard.edu/abs/2014A%26A...568A..22B/abstract">Betoule et al. (2014)</a>, which was downloaded <a href="http://supernovae.in2p3.fr/sdss_snls_jla/ReadMe.html">here</a>. The function uses numpy loadtxt function, which allows to read well formatted columns of data in the ASCII format easily.
#
#     After reading numpy array <tt>zsn</tt> contains redshifts of 740 supernovae, while arrays <tt>msn</tt>, $m$, and <tt>emsn</tt> contain measured apparent magnitudes and their uncertainties
#
#
# Distance modulus is defined as
#
# $$\mu = m - M = 5\log_{10}d_L(z,H_0,\Omega_{\rm m0},\Omega_\Lambda) + 25.$$
#
# where $d_L$ is distance in Megaparsecs and $M$ is the absolute magnitude of the supernovae (this is the magnitude supernova would have at a fixed distance of 10 parsecs). For this exercise we will assume that supernovae are perfect standard candles, which means that $M$ has a single value for all of them. This means that we should be able to predict what apparent magnitudes of supernovae should be at different redshifts:
#
# $$m_{\rm pred} = M + 5\log_{10}d_L(z,H_0,\Omega_{\rm m0},\Omega_\Lambda) + 25.$$
#
# **Task** plot supernovae data as a scatter of points in the $m-z$ plane and plot $m_{\rm pred}(z)$ for a grid of $z$ values in the range $z\in [0,2]$ as a line. Add a legend to your plot that describes points and the line. You should write your own Matplotlib code and *not* use function <tt>plot_line_points</tt> in <tt>codes.plotting</tt>, although you are welcome to consult it for example of how to make such plots.
#
#  Make sure your plot is sufficiently large, axes are labeled and font size in the axis labels and legend is sufficiently large to be legible. You can use <tt>codes.plotting.plot_prettier</tt> function to set up Matplotlib environment for good plotting, but adjust <tt>figsize</tt> of your plot, as needed. You can find examples of plots that have good size and font sizes in the distributed notebooks.
#
#
# Try different values of $M$ in the range from 0 to -25, and values of $\Omega_{\rm m0}$ and $\Omega_\Lambda$ in the range $[0,1]$, while keeping $H_0=70$ and try to find a combination of $M$, $\Omega_{\rm m0}$ and $\Omega_\Lambda$ for which the line matches the supernova data best. Plot a plot of $m-z$ described above for this parameter combination.
#
# Summarize your conclusions and quote the best values of $M$, $\Omega_{\rm m0}$ and $\Omega_\Lambda$ that you found.
#
# %% markdown
# ***Note:*** Useful Matlotlib and numpy functions: <tt>np.linspace, plt.xlabel, plt.ylabel, plt.scatter, plt.plot, plt.legend</tt>
#
# ***Note:*** If you did not complete implementation of functions in the previous exercises you can use AstroPy function to compute $d_L$ to do this exercise.
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

# %% codecell
# read vectors of redshifts, apparent magnitude in B-band, mB, and its uncertainty emB:
zsn, msn, emsn = read_jla_data(sn_list_name = 'https://astro.uchicago.edu/~andrey/classes/a211/data/jla_lcparams.txt')
nsn = np.size(zsn)
print("read sample of %d supernovae..."%(nsn))
# %% markdown
# ## <font color='darkblue'> Exercise 4 (extra-credit): implementing Romberg integration scheme (25 points)
#
#
# **Task 4a. (20 points)** Implement a function that estimates integral $\int_a^b f(x)dx$ using Romberg integration method with error better than specified tolerance level, as in the exercise above, but using full Romberg scheme to estimate $R_m$ rather than explicit <tt>r1</tt>, <tt>r2</tt>, <tt>r3</tt> functions and that uses  $R_{m+1}$ and $R_{m}$ estimates to control the current estimate of error. (25 points)
#
# **Task 4b. (5 points)** Test your function by computing $$\int_0^1 e^x dx$$ and computing fractional error of the numerical relative to exact value ($=e-1$) similarly to how this was done in <tt>04_integration</tt> notebook for trapezoidal integration function. (1 point)
#
# Plot the fractional error you get for your integral, as a function of input rtol value to demonstrate that your result is as accurate or better than specified (4 points).
#
