# -*- coding: utf-8 -*-
"""
Auxiliary plotting functions that will be used in the notebooks

Andrey Kravtsov (Spring 2022)
"""

# import relevant packages
import matplotlib.pyplot as plt
import numpy as np

# the following commands make plots look better
def plot_prettier(dpi=200, fontsize=10, usetex=False):
    '''
    Make plots look nicer compared to Matplotlib defaults
    Parameters:
        dpi - int, "dots per inch" - controls resolution of PNG images that are produced
                by Matplotlib
        fontsize - int, font size to use overall
        usetex - bool, whether to use LaTeX to render fonds of axes labels
                use False if you don't have LaTeX installed on your system
    '''
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
    plt.rc('text', usetex=usetex)


from math import factorial
def exp_taylor(x0, N, x):
    '''
    Taylor expansion up to order N for exp(x)
    '''
    dummy = np.zeros_like(x)
    for n in range(N+1):
        dummy += np.exp(x0)*(x-x0)**n/factorial(n)
    return dummy

def plot_line_points(x, y, figsize=6, xlabel=' ', ylabel=' ', col= 'darkslateblue',
                     xp = None, yp = None, eyp=None, points = False, pmarker='.',
                     psize=1., pcol='slateblue',
                     legend=None, plegend = None, legendloc='lower right',
                     plot_title = None, grid=None, figsave = None):
    plt.figure(figsize=(figsize,figsize))
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    # Initialize minor ticks
    plt.minorticks_on()

    if legend:
        plt.plot(x, y, lw = 1., c=col, label = legend)
        if points:
            if plegend:
                plt.scatter(xp, yp, marker=pmarker, s=psize, c=pcol, label=plegend)
            else:
                plt.scatter(xp, yp, marker=pmarker, s=psize, c=pcol)
            if eyp is not None:
                plt.errorbar(xp, yp, eyp, linestyle='none', marker=pmarker, color=pcol, markersize=psize)
        plt.legend(frameon=False, loc=legendloc, fontsize=3.*figsize)
    else:
        plt.plot(x, y, lw = 1., c=col)
        if points:
            plt.scatter(xp, yp, marker=pmarker, s=psize, c=pcol)

    if plot_title:
        plt.title(plot_title, fontsize=3.*figsize)

    if grid:
        plt.grid(linestyle='dotted', lw=0.5, color='lightgray')

    if figsave:
        plt.savefig(figsave, bbox_inches='tight')

    plt.show()


from matplotlib import cm

def plot_color_map2(x, y, data, xlim=[0.,1], ylim=[0.,1.],
                   xlabel = ' ', ylabel = ' ', cmap='winter', colorbar=False,
                   figsize=3.0, cbar_label=None, figsave=None):
    '''
    Visualize f(x,y) as a 2D color map

    Parameters
    ----------
    x : 1d numpy array of floats with x variable values
    y : 1d numpy array of floats with x variable values

    data : 2d numpy array of shape (x.size, y.size) with f(x,y) values

    xlim : list with 2 elements, limits for x-axis, optional
        The default is [0.,1].
    ylim : list with 2 elements, limits for y-axis, optional
        The default is [0.,1].
    xlabel : string, optional
        x-axis label. The default is ' '.
    ylabel : string, optional
        y-axis label. The default is ' '.
    cmap : str, optional
        Matplotlib colormap. The default is 'winter'.
    colorbar : bool, optional
        whether to plot colorbar. The default is False.
    figsize : float, optional
        will determine figure size (figsize,figsize). The default is 3.0.
    cbar_label : str, optional
        colorbar label. The default is None.
    figsave : str, optional
        if str is provided figure will be saved in the file given by the str.
        for example 'plot.pdf' or 'plot.png'. The default is None.

    Returns
    -------
    None.

    '''
    fig, ax = plt.subplots(figsize=(figsize,figsize))
    ax.axis([xlim[0], xlim[1], ylim[0], ylim[1]])

    plt.xlabel(xlabel); plt.ylabel(ylabel)
    cmap = cm.get_cmap(cmap)
    im = ax.pcolormesh(x, y, data, cmap=cmap, rasterized=True)
    if colorbar:
        cbar = fig.colorbar(im, ax=ax)
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label)
    if figsave:
        plt.savefig(figsave, bbox_inches='tight')
    plt.show()



def taylor_exp_illustration(figsize=3.0):
    '''
    Function to produce an illustration for the Taylor expansion approximation
    for the exp(x) function

    Parameters
    ----------
    figsize : float, Matplotlib figure size

    Returns
    -------
    None.

    '''
    N = 4; x0 = 1.0

    plt.figure(figsize=(figsize,figsize))
    #plt.title('Taylor expansion of $e^x$ at $x_0=%.1f$'%x0, fontsize=9)
    plt.xlabel('$x$'); plt.ylabel(r'$e^x, f_{\rm Taylor}(x)$')

    xmin = x0 - 1.0; xmax = x0 + 1.0
    x = np.linspace(xmin, xmax, 100)
    plt.xlim([xmin,xmax]); plt.ylim(0.,8.)

    exptrue = np.exp(x)
    plt.plot(x, exptrue, linewidth=1.5, c='m', label='$e^x$')
    colors = ['darkslateblue', 'mediumslateblue', 'slateblue', 'lavender']
    lstyles = [':','--','-.','-','-.']
    for n in range(N):
        expT = exp_taylor(x0, n, x)
        plt.plot(x, expT, linewidth=1.5, c=colors[n], ls=lstyles[n], label='%d term(s)'%(n+1))

    plt.legend(loc='upper left', frameon=False, fontsize=7)
    plt.show()
    return

def plot_line_points(x, y, figsize=6, xlabel=' ', ylabel=' ', col= 'darkslateblue',
                     xp = None, yp = None, eyp=None, points = False, pmarker='.',
                     psize=1., pcol='slateblue',
                     legend=None, plegend = None, legendloc='lower right',
                     plot_title = None, grid=None, figsave = None):
    plt.figure(figsize=(figsize,figsize))
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    # Initialize minor ticks
    plt.minorticks_on()

    if legend:
        plt.plot(x, y, lw = 1., c=col, label = legend)
        if points:
            if plegend:
                plt.scatter(xp, yp, marker=pmarker, s=psize, c=pcol, label=plegend)
            else:
                plt.scatter(xp, yp, marker=pmarker, s=psize, c=pcol)
            if eyp is not None:
                plt.errorbar(xp, yp, eyp, linestyle='none', marker=pmarker, color=pcol, markersize=psize)
        plt.legend(frameon=False, loc=legendloc, fontsize=3.*figsize)
    else:
        plt.plot(x, y, lw = 1., c=col)
        if points:
            plt.scatter(xp, yp, marker=pmarker, s=psize, c=pcol)

    if plot_title:
        plt.title(plot_title, fontsize=3.*figsize)

    if grid:
        plt.grid(linestyle='dotted', lw=0.5, color='lightgray')

    if figsave:
        plt.savefig(figsave, bbox_inches='tight')

    plt.show()

def colorbar(mappable):
    """
    a hack to make colorbars look good by Joseph Long
    https://joseph-long.com/writing/colorbars/
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def plot_data_fit2d(x, y, z, model, model_name='model'):
    # Plot the data with the best-fit model
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax1.imshow(z, origin='lower', interpolation='nearest', vmin=-1e4, vmax=5e4)
    ax1.set_title("data")

    ax2.imshow(model, origin='lower', interpolation='nearest', vmin=-1e4,
               vmax=5e4)
    ax2.set_title(model_name)
    img3 = ax3.imshow(z - model, origin='lower', interpolation='nearest', vmin=-1e4,
               vmax=1e4)
    colorbar(img3)
    ax3.set_title("residual")
    plt.show()


def show_image(img, cmap='Greys', auto_aspect=False, figsize=4):
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(img,  cmap='Greys')
    if auto_aspect:
        ax.set_aspect('auto')
    plt.show()

def plot_color_map(x, y, data, xlim=[0.,1], ylim=[0.,1.],
                   xlabel = ' ', ylabel = ' ', cmap='winter', colorbar=None,
                   contours = False, levels = [], contcmap = 'winter', cbar_label=None,
                   plot_title=None, figsize=3.0, figsave=None):
    """
    Helper function to plot colormaps of a 2d array of values with or without contours

    Parameters:
    -----------
    x, y - 1d numpy vectors generated by numpy.meshgrid functions
    data - 2d numpy array, containing data values at the grid represented by x and y
    xlim, ylim - lists of size 2 containing limits for x and y axes
    cmap: string, Matplotlib colormap to use
    colormap: boolean, if True plot colorbar
    contours: boolean, if True plot contours corresponding to data levels specified by levels parameter
    levels: values of data for which to draw contour levels
    contcmap: string, Matplotlib colormap to use for contour lines
    plot_title: string, if not None will be used to prouce plot title at the top
    figsize: float, size of Matplotlib figure
    figsave: None or string, if string, the string will be used as a path/filename to save PDF of the plot

    Returns:
    --------
    Nothing

    """

    fig, ax = plt.subplots(figsize=(figsize,figsize))
    ax.axis([xlim[0], xlim[1], ylim[0], ylim[1]])

    plt.xlabel(xlabel); plt.ylabel(ylabel)
    cmap = cm.get_cmap(cmap)
    im = ax.pcolormesh(x, y, data, cmap=cmap, rasterized=False)
    if contours:
        ax.contour(x, y, data, levels=levels, cmap=contcmap)
    if colorbar:
       cbar = fig.colorbar(im, ax=ax)
       if cbar_label is not None:
           cbar.ax.set_ylabel(cbar_label)
    if plot_title:
        plt.title(plot_title, fontsize=3.*figsize)

    if figsave:
        plt.savefig(figsave, bbox_inches='tight')
    plt.show()


import scipy.optimize
from matplotlib.colors import LogNorm

def conf_interval(x, pdf, conf_level):
    return np.sum(pdf[pdf > x])-conf_level

def plot_2d_dist(x,y, xlim, ylim, nxbins, nybins, figsize=(5,5),
                cmin=1.e-4, cmax=1.0, smooth=None, xpmax=None, ypmax=None,
                log=False, weights=None, xlabel='x', ylabel='y',
                clevs=None, fig_setup=None, savefig=None):
    """
    construct and plot a binned, 2d distribution in the x-y plane
    using nxbins and nybins in x- and y- direction, respectively

    log = specifies whether logged quantities are passed to be plotted on log-scale outside this routine
    """
    if fig_setup is None:
        fig, ax = plt.subplots(figsize=figsize)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
    else:
        ax = fig_setup
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim); ax.set_ylim(ylim)

    if xlim[1] < 0.: ax.invert_xaxis()

    if weights is None: weights = np.ones_like(x)
    H, xbins, ybins = np.histogram2d(x, y, weights=weights, bins=(np.linspace(xlim[0], xlim[1], nxbins),np.linspace(ylim[0], ylim[1], nybins)))

    H = np.rot90(H); H = np.flipud(H);

    X,Y = np.meshgrid(xbins[:-1],ybins[:-1])

    if smooth != None:
        from scipy.signal import wiener
        H = wiener(H, mysize=smooth)

    H = H/np.sum(H)
    Hmask = np.ma.masked_where(H==0,H)

    if log:
        X = np.power(10.,X); Y = np.power(10.,Y)

    pcol = ax.pcolormesh(X, Y,(Hmask),  cmap=plt.cm.BuPu, norm = LogNorm(), linewidth=0., rasterized=True)
    pcol.set_edgecolor('face')

    # plot contours if contour levels are specified in clevs
    if clevs is not None:
        lvls = []
        for cld in clevs:
            sig = scipy.optimize.brentq( conf_interval, 0., 1., args=(H,cld) )
            lvls.append(sig)

        ax.contour(X, Y, H, linewidths=(1.0,0.75, 0.5, 0.25), colors='black', levels = sorted(lvls),
                norm = LogNorm(), extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]])
    if xpmax is not None:
        ax.scatter(xpmax, ypmax, marker='x', c='orangered', s=20)
    if savefig:
        plt.savefig(savefig,bbox_inches='tight')
    if fig_setup is None:
        plt.show()
    return
