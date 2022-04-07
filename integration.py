import matplotlib.pyplot as plt
import numpy as np

#function to integrate
def f(x):
    return x**2

#actual integral value
def f_integral(x):
    return (1/3)*x**3

#integration function
def integrate(f,a,b,nsteps,middle=True):

    bin_size = (b-a)/nsteps #calculate bin size
    bins = np.arange(a,b+bin_size,bin_size) #bin size -> range
    f_bins = f(bins) #call f on this range

    bar_xs = bins[:-1]+bin_size/2 #So  ca

    if middle:
        averages = (f_bins[1:] + f_bins[:-1])/2
    else:
        averages = f_bins[:-1]
    return bin_size,bar_xs,averages

def get_error(f,f_integral,a,b,start=10,end=1000,step=10,middle=True):
    expected = f_integral(b)-f_integral(a)
    steps = np.arange(start,end,step)
    ferr = []
    bins = []
    for s in steps:
        bin_size,bar_xs,averages = integrate(f,a,b,s,middle=middle)
        bins.append(bin_size)
        #print(bin_size)
        calculated = np.sum(averages*bin_size)
        #print(calculated)
        #print(np.abs(calculated/expected - 1.))
        ferr.append(np.abs(calculated/expected - 1.))
    return steps,np.array(ferr),bins

steps,ferr,bins = get_error(f,f_integral,0,100)
#print(ferr)
plt.plot(bins,ferr)
plt.xscale('log')
#plt.x_label("y")
plt.gca().invert_xaxis()
'''
last_low = 1
plt.text(steps[0] * (1 + 0.01), ferr[0] * (1 + 0.01) , f'{bins[0]:.5f}', fontsize=5)
for i in range(len(steps)):
    if ferr[i] > last_low:
        plt.text(steps[i] * (1 + 0.01), ferr[i] * (1 + 0.01) , f'{bins[i]:.5f}', fontsize=5)
    else:
        last_low = ferr[i]
'''

plt.show()


'''
a,b = 0,100
steps = 500
xs = np.arange(a,b+1)
expected = f_integral(b)-f_integral(a)

bin_size,bar_xs,averages = integrate(f,a,b,steps,middle=True)
calculated = np.sum(averages*bin_size)
ferr = np.abs(calculated/expected - 1.)
print(ferr)
'''
'''
plt.bar(bar_xs,averages,width=bin_size,color="red",zorder=0)
plt.plot(xs,f(xs),label="curve",zorder=1)
plt.show()
'''
