import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline

AU = 1.49598e+13 # 1 AU = average distance from the Earth to the Sun in centimeters (cm)
G = 6.67259e-08 # universal gravitational constant in cgs units
yr =  3.15569e+07 # 1 year in seconds
msun = 1.9891e33 # mass of the Sun in grams
mearth = 5.9742e27 # mass of the Earth in grams
vcirc = (G*msun/AU)**0.5 # circular velocity = sqrt(G*Msun/AU)

x1, y1, z1 = 0., 0., 0. # coordinates of the Sun
x2, y2, z2 =  AU, 0., 0. # coordinates of the Earth

vx1, vy1, vz1 = 0., 0., 0. # initial velocity of the Sun
vx2, vy2, vz2 = 0., vcirc, 0. # initial velocity of the Earth

m1, m2 = msun, mearth # masses

x, y, z = [], [], [] # lists to record positions of the Earth during time steps

nsteps = 100000
dt = 10 * yr / nsteps
print(dt)

for n in range(nsteps):
    r12 = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5

    ax1 = G * m2 / r12**3 * (x2 - x1)
    ay1 = G * m2 / r12**3 * (y2 - y1)
    az1 = G * m2 / r12**3 * (z2 - z1)
    ax2 = G * m1 / r12**3 * (x1 - x2)
    ay2 = G * m1 / r12**3 * (y1 - y2)
    az2 = G * m1 / r12**3 * (z1 - z2)

    vx1 = vx1 + ax1 * dt
    vy1 = vy1 + ay1 * dt
    vz1 = vz1 + az1 * dt
    vx2 = vx2 + ax2 * dt
    vy2 = vy2 + ay2 * dt
    vz2 = vz2 + az2 * dt

    x1 = x1 + vx1 * dt
    y1 = y1 + vy1 * dt
    z1 = z1 + vz1 * dt
    x2 = x2 + vx2 * dt
    y2 = y2 + vy2 * dt
    z2 = z2 + vz2 * dt

    x.append(x2)
    y.append(y2)
    z.append(z2)

x, y, z = np.array(x)/AU, np.array(y)/AU, np.array(z)/AU

plt.figure(figsize=(3,3))

plt.plot(x,y)

plt.show()
