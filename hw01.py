# %% markdown
# <center>
#
# ## <font color='darkblue'>ASTR 21100/ASTR 31200
# <center>
#
# ### <font color='darkblue'>"Computational Techniques in Astrophysics"
#
# <center>
#
# ### <font color='darkblue'> Integrating particle orbits in $N$-body problems
#
# <center>
#
# ### <font color='darkblue'> 35 points + 6 possible extra-credit
#
#
# <center>
#
# ### <font color='darkblue'> Due Wednesday,  Apr 6, 10pm
#
#
# %% markdown
# ### <font color='darkblue'>1  (25 points).  Implementing code for $N$-body integration
#
# **1a (20 points).** Implement a function that carries out integrations of $N$-body dynamics of $N$ particles. The function should take as input  initial coordinates and velocities of $N$ particles and evolves them forward using a specified constant time step for some input number of steps. You can use cgs unit system and examples provided in the notebook <a href="https://drive.google.com/file/d/1hPl5wm_XX8yBOibjQ0CyOceWuK1oF9BH/view?usp=sharing"><tt>01_orbit_integration_class</tt>.</a>
#
# This function should also take as input the name of a function that computes accelerations for every particle from using coordinates. This function should be called inside the $N$-body code to compute accelerations at each step.
#
# The function
#
# For example, function I implemented for this has the following inputs, which you can use as a suggested format.
#
#     def nbody_integrate(x, v, mp, dt=None, nsteps=None, acc_func=None, scheme='KD'):
#         """
#         integrate equations of motions starting from the input vectors x, v, mp
#         for nsteps with constant time step dt
#
#         Parameters:
#         ------------
#
#         x, v- NumPy vectors of floats of shape (N,3) containing
#              coordinates and velocities of N particles
#         mp - NumPy vector of masses of particles of length N
#         dt - float
#             step size
#         acc_func  - python function
#             name of a user-supplied function to compute mutual accelerations of particles
#             and/or accelerations from an external potential
#         scheme: str, scheme to use to evolve particle positions and velocities
#                 possible choices 'Euler', 'KD'
#
#         Returns:
#         -----------------
#         tt - numpy float vector
#             recorded orbit times
#         xt, vt - numpy float vectors
#             coordinates, velocities at times tt
#
#
#     """
#
# **1b. (5 points)** Required tests and follow up questions:
# After you complete the function, test it by integrating a two body system with the Sun and the Earth with their true masses and with Earth on a circular orbit while the Sun is at rest initially. Run simulations using integration with the 1st order Euler and 2nd order leapfrog. An example of how this problem can be initialized  is provided in 01_orbit_integration. Follow evolution for several tens of orbits (up to ~100) and make sure that at least for the leapfrog scheme the orbit stays circular during integration.
#
# Specifically, plot distance of the Earth from the origin $x, y, z = 0, 0, 0$ as a function of time for the 1st order Euler and KD  schemes. How do results for these schemes compare for a given step size? How do results depend on step size? Try several different step sizes.  Do results change/improve if you decrease step size significantly? Which step size you would be confident to use for the actual calculations of planet orbits?
# %% markdown
# #### Additional info and hints:
# The evolution code should consist of a loop stepping in time.
# For each particle acceleration is computed by direct summation of forces from all other particles and use it to advance positions of particles. Thus, to compute acceleration of all particles one has to have two loops over particles (which is what makes this problem scale as $N^2$).
#
#
# **Note:** Use of Python classes is useful for this problem. If you feel comfortable with using classes I encourage you to try their use for this problem.  If you want to go this route, I'll be happy to provide guidance, if needed.
#
# **Note:** The second loop during step can be done via NumPy operation without an explicit loop. With some effort both loops in computation of accelerations for all $N$ particles can be replaced with NumPy operations. I encourage you to try to do try getting rid of the second or both loops using NumPy operations. If you will be able to avoid using inner loop using NumPy operation you will receive ***3 extra-credit points*** and if you will be able to avoid both you will get ***6 extra-credit points.*** Attempt this only after you get code working with a function computing accelerations using loops.
# %% markdown
# ### <font color='darkblue'>2. (10 points)  Integrating orbits of planets in the <a href="http://www.openexoplanetcatalogue.com/planet/Gliese%20876%20e/">exoplanet system GJ 876</a>
#
# Note that if for some reason you will not be able to make your code in the exercise 1 above to work properly. I can provide you with a function to carry out this exercise. You will receive a partial credit for exercise 1, depending on how much progress you've made towards complete implementation.
#
#
# #### Background info:
# GJ876 is a red dwarf star for which a series of precise radial motion measurements exists. Modelling of these motions indicates that it has at least two planets (quite likely three) orbiting around it. The two planets have periods of $\approx 60$ and $\approx 30$ days and are locked in a 2:1 mean motion resonance.
# Cartesian coordinates, velocities, and masses of the stars for one of the best fitting models are presented in Table 3 of <a href="http://adsabs.harvard.edu/abs/2005ApJ...622.1182L">Laughlin et al. 2005</a> and are available in this <a href="https://github.com/a-kravtsov/a330f17/blob/master/data/gj876.dat">file.</a> The code snippet below reads it in cgs units.
#
# #### Required tests and follow up questions:
# Initialize the three body system (star and two planets) with these coordinates and velocities as initial conditions using data read in the code snippet below. Integrate them forward for 3320 days with the $N$-body code you implemented in the exercise 1 using your conclusions about the best scheme and step size from that exercise.
#
# Record positions and velocities of the planets and the star. Make a scatter plot of $x$ and $y$ coordinates (converted to the astronomical unit AU) of the star and planets in half-day intervals. Plot the $z$-component of velocity of the star as a function of time. Compare to Figures 1 and 2 in <a href="https://ui.adsabs.harvard.edu/abs/2005ApJ...622.1182L/abstract">Laughlin et al. 2005</a> reproduced below. Can you reproduce the figures? We will discuss what these figures show in class.
# %% codecell
# reading initial planet positions from Table 3 of Laughlin et al. 2005
import numpy as np

data_file = 'https://astro.uchicago.edu/~andrey/classes/a211/data/gj876.dat'
name = np.loadtxt(data_file, usecols=[0], unpack=True, dtype=str)
mp, xp, yp, zp, vxp, vyp, vzp = np.loadtxt(data_file, usecols=(1,2,3,4,5,6,7), unpack=True)
#print(name, mp, xp, yp, zp, vxp, vyp, vzp)
# %% codecell
AU = 1.49598e+13 # 1 AU = average distance from the Earth to the Sun in centimeters (cm)
G = 6.67259e-08 # universal gravitational constant in cgs units

# %% codecell
x = np.column_stack((xp,yp,zp))
v = np.column_stack((vxp,vyp,vzp))
# %% codecell

def nbody_integrate(x, v, mp, dt=None, nsteps=None, acc_func=None, scheme='KD'):
    xs = np.stack([x for i in range(x.shape[0])])
    r = np.array([np.linalg.norm(xs[i]-x[i],axis=1) for i in range(x.shape[0])])
    #a = np.array([(xs[i]-x[i])*r[i][:,np.newaxis] for i in range(x.shape[0])])
    a = np.array([np.sum((xs[i]-x[i])*r[i][:,np.newaxis],axis=0) for i in range(x.shape[0])])
    print(a)
    a = np.array([np.sum((xs[i]-x[i])*np.where(mp[i]/r[i][:,np.newaxis] != np.inf, mp[i]/r[i][:,np.newaxis], 0),axis=0) for i in range(x.shape[0])])
    print(a)
    test = np.where(mp[0]/r[0][:,np.newaxis] != np.inf, mp[0]/r[0][:,np.newaxis], 0)
    print(test)
    #print(mp[0])
    #print(v)
    #print(v+a*dt)

# %% codecell

print(nbody_integrate(x,v,mp,dt=3000,nsteps=100000))
