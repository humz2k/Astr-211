import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM
import astropy.units as u
import scipy.interpolate
import scipy.optimize

# the following commands make plots look better
def plot_pretty(dpi=150,fontsize=15):
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
    plt.rcParams['figure.figsize'] = [5, 5]

plot_pretty()

# %% codecell
#zCMB, mB, emB are redshift of SNia, its apparent B-band magnitude, and emB is its error
# x1 and ex1 are stretch parameter measured for each SN and its uncertainty
# csn and ecsn are color parameter and its uncertainty
zCMB, mB, emB, x1, ex1, csn, ecsn = np.loadtxt('https://astro.uchicago.edu/~andrey/classes/a211/data/jla_lcparams.txt',
                                               usecols=(1, 4, 5, 6, 7, 8, 9), unpack=True)

print("read sample of %d supernovae..."%(np.size(zCMB)))
# %% codecell
def d_l_tilde_astropy(z, H0, Om0, OmL, clight=2.99792e5):
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=OmL)

    return cosmo.luminosity_distance(z=z) / u.Mpc / (clight/H0)

def get_dl_train_test(ntrain=15, z=1.0, H0=70., clight=2.99792e5,
                      om0min=0., om0max = 1., omlmin=0., omlmax=1., spacing=np.linspace):

    om0tr = spacing(om0min, om0max, ntrain)
    omltr = spacing(omlmin, omlmax, ntrain)

    dl_train = np.zeros((ntrain, ntrain))
    for i, omd in enumerate(om0tr):
        for j, omld in enumerate(omltr):
                dl_train[i,j] = d_l_tilde_astropy(z, H0, omd, omld, clight = clight)

    return om0tr, omltr, dl_train

def train_model(ntrain=15, z=1.0, H0=70.,clight=2.99792e5,om0min=0., om0max = 1., omlmin=0., omlmax=1., s=0, kx=3, ky=3,spacing=np.linspace):
    om0tr, omltr, dl_train = get_dl_train_test(ntrain=ntrain, z=z, H0=H0, clight=clight, om0min=om0min, om0max=om0max, omlmin=omlmin, omlmax=omlmax, spacing=spacing)
    return scipy.interpolate.RectBivariateSpline(om0tr, omltr, dl_train, s=s, kx=kx, ky=ky)
# %% markdown
#### Constants
# %% codecell
constants = {
"H0": 70.,
"clight": 2.99792e5
}
# %% markdown
#### Model Parameters:
# %% codecell
parameters = {
"ntrain": 5,
"om0min": 0.,
"om0max": 1.,
"omlmin": 0.,
"omlmax": 1.,
"s": 0,
"kx": 3,
"ky": 3,
"spacing": np.linspace
}
# %% codecell
print("Training Models")
dlz = {}
for idx,z in enumerate(zCMB):
    if idx % (len(zCMB)//10) == 0:
        if idx == 0:
            print("   ->   0% done...")
        else:
            print("   ->  " + str(round((idx/len(zCMB))*100)) + "% done...")
    dlz[z] = train_model(**parameters, **constants, z=z)
print("   -> 100% done...")
# %% codecell
n_samples = 100
print("Testing the models using",n_samples,"random samples...")
sampled_frac_err = np.empty(n_samples,dtype=np.float64)
for i in range(n_samples):
    z = np.random.choice(zCMB)
    model = dlz[z]
    om0 = np.random.uniform()
    omL = np.random.uniform()
    model_val = model(om0,omL)
    direct_val = d_l_tilde_astropy(z,constants["H0"],om0,omL)
    sampled_frac_err[i] = np.abs(model_val/direct_val -1.)

print("Max Frac Err:",np.max(sampled_frac_err))
print("Min Frac Err:",np.min(sampled_frac_err))
print("Mean Frac Err:",np.mean(sampled_frac_err))
# %% codecell
def likelihood(x, zCMB, mB, emB, x1, ex1, csn, ecsn, dlz, constants=constants):
    om0, omL, M0, a, b = x
    sig_mu_squared = emB**2 + (a**2) * (ex1**2) + (b**2) * (ecsn**2)
    dl = np.array([dlz[z](om0,omL) for z in zCMB]).flatten()
    mu = 5 * np.log10(dl) + 25
    mu_obs = mB - (M0 + 5*np.log10(constants["clight"]/constants["H0"]) + 25)
    del_mu_squared = (mu_obs - mu)**2
    first = del_mu_squared/sig_mu_squared
    second = np.log(2*np.pi*sig_mu_squared)
    return -0.5*np.sum(first) - 0.5*np.sum(second)
# %% markdown
#### Differential Evolution Arguments
# %% codecell
func_kwargs = {
"zCMB": zCMB,
"mB": mB,
"emB": emB,
"x1": x1,
"ex1": ex1,
"csn": csn,
"ecsn": ecsn,
"dlz": dlz,
"constants": constants
}
bounds = ((0,1),(0,1),(20,28),(0.05,0.3),(1,5))

kwargs = {
"popsize": 15,
"tol": 0.01,
"atol": 0
}
# %% codecell
result = scipy.optimize.differential_evolution(likelihood,bounds,args=func_kwargs.values(),**kwargs)
print(result)
best_om0, best_omL, best_M0, best_a, best_b = result.x
# %% markdown
$$$
\begin{align}
\int_{\frac{1}{a}}^{a}g(x)\,\mathrm{d}x &= 1 \\
\int_{\frac{1}{a}}^{a}\frac{A}{\sqrt{x}}\,\mathrm{d}x &= 1 \\
A\int_{\frac{1}{a}}^{a}x^{-\frac{1}{2}}\,\mathrm{d}x &= 1 \\
A\left[ 2x^{\frac{1}{2}} \right]_{\frac{1}{a}}^a &= 1 \\
A(2\sqrt{a}-2\sqrt{\frac{1}{a}}) &= 1 \\
A &= \frac{1}{(2\sqrt{a}-2\sqrt{\frac{1}{a}})} \\
A &= \frac{1}{2(\sqrt{a}-\sqrt{\frac{1}{a}})} \\
\end{align}
$$$
