# %% codecell
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

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
    plt.rcParams['figure.figsize'] = [5, 3]

plot_pretty()

# %% codecell
def ftrain(x, scale=0.5):
    return 1.5*x + np.sin(x) + np.random.normal(scale=scale, size=np.size(x))
# %% codecell
ntr = 30
xtr = np.linspace(1., 6., ntr)
ftr = ftrain(xtr, scale=1.)

plt.plot(xtr, ftr)
plt.scatter(xtr,ftr,s=10)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
# %% codecell
ntr = 30
xtr = np.linspace(1., 6., ntr)
ftr = ftrain(xtr, scale=1.)

def poly_fit(xtr, ftr,  method='polynomial', porder=None, s=0.):
    assert method in ['polynomial','splint','splreg']
    if method == 'polynomial':
        pass
    elif method == 'splint':
        pass
    elif method == 'splreg':
        pass

poly_fit(xtr,ftr)
