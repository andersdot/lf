from __future__ import print_function

import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from scipy import integrate
import pdb
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

def schecter(mag, Mstar, phi, alpha):
        return 2./5*phi*np.log(10)*(10.**(2./5*(Mstar-mag)))**(alpha+1.)*np.exp(-10.**(2./5*(Mstar-mag)))

def lnprior(theta):
    Mstar, alpha = theta
    #if -4.0 <= alpha <= 0.0 and -40 <= Mstar <= -14: return 0.
    if (alpha > -4.) & (alpha < 0.):
        return -0.5/4.*((Mstar-Mstar_guess)/sigma)**2. - np.log(sigma*np.sqrt(2*np.pi))
    else: return -np.inf

def lnlike(theta, mag):
    Mstar, alpha = theta
    phi = 1.
    sum = integrate.quad(schecter, np.min(mag), np.max(mag), args=(Mstar, phi, alpha))[0]
    pdf = schecter(mag, Mstar, phi, alpha)/sum
    return np.sum(np.log(pdf))

def lnprob(theta, mag):
    lp = lnprior(theta)
    if not np.isfinite(lp + lnlike(theta, mag)):
        return -np.inf
    #if np.isnan(lp + lnlike(theta, mag)): pdb.set_trace()
    return lp + lnlike(theta, mag)

def plotMcmc(ax, z):
    mcmcz = {10:None, 8:'z8', 7:'z7', 6:'z6', 5:'z5', 4:'z4'}
    span = {4:[-17.5, -22.5], 
        5:[-17.5, -22.5], 
        6:[-17.5, -22], 
        7:[-18,   -22], 
        8:[-18.5, -21.5],
        10:[-18.5, -21.5]}

    xx = np.arange(span[z][1], span[z][0], 0.1)
    if z == 10:
        plus  = schecter(xx, Mstar_g[ind[step]], phi_g[ind[step]]+dphi[ind[step]], alpha_g[ind[step]])
        minus = schecter(xx, Mstar_g[ind[step]], phi_g[ind[step]]-dphi[ind[step]], alpha_g[ind[step]])
        ax.fill_between(xx, plus, minus, color='grey')
    else:
        mcmcdata = np.load('cosmo25/mcmc_chains.npy')
        par = mcmcdata[mcmcz[z]]
        Mstarmcmc = par[0:1000000-1]
        alphamcmc = par[1000000:2000000-1]
        phimcmc   = par[2000000:3000000-1]
        if z in [6, 8]: size=5000
        else: size=500
        pick = np.random.random_integers(0, high=1000000-1, size=size)
        for j in range(len(pick)): ax.plot(xx, schecter(xx, Mstarmcmc[pick[j]], phimcmc[pick[j]], alphamcmc[pick[j]]), linewidth=0.5, color='grey')
    #p = plt.Rectangle((-20,1), 0, 0, alpha=0.5, color=color[z], label='z~'+str(z))
    #ax.add_patch(p)
    
IMF = 'kroupa'
s = ['001818', '001380', '001093', '000891', '000744', '000547']
ind = {'001818':0, '001380':1, '001093':2, '000891':3, '000744':4, '000547':5}
alpha_g = [-1.56, -1.67, -2.02, -2.03, -2.36, -2.27]
phi_g = np.array([14.1, 8.95, 1.86, 1.57, 0.72, 0.08])*1e-4
Mstar_g = [-20.73, -20.81, -21.13, -21.03, -20.89, -20.92]

dalpha = [0.06, 0.05, 0.10, 0.21, 0.54, 0.54]
dphi   = np.array([2.05, 1.92, 0.94, 1.45, 2.52, 0.04])*1e-4
dMstar = [0.09, 0.13, 0.31, 0.50, 1.08, 1.08]
redshift = {'001818':4, '001380':5, '001093':6,'000891':7, '000744':8, '000547':10}
color = {4:'purple', 5:'blue', 6:'cyan', 7:'green', 8:'orange', 10:'red'}

alpha_meas = []
alpha_plus = []
alpha_minus = []

Mstar_meas = []
Mstar_plus = []
Mstar_minus = []

phi_meas = []
phi_plus= []
phi_minus = []


#----------------------------------------------
#----------  Universal Plot Settings ----------
#----------------------------------------------
fontsize=16
nwalkers = 10
fracDeviation = 0.1
nsteps = 1000
nsamples = 2000
markersize = 8

#----------------------------------------------
#--------- EVOLUTION FIGURE -------------------
#----------------------------------------------

left  = 0.2  # the left side of the subplots of the figure
right = 0.99    # the right side of the subplots of the figure
bottom = 0.075   # the bottom of the subplots of the figure
top = 0.99      # the top of the subplots of the figure
wspace = 0.001   # the amount of width reserved for blank space between subplots
hspace = 0.001   # the amount of height reserved for white space between subplots

fig_evolution, axes_evolution = pl.subplots(3, sharex=True, figsize=(6.5,10))
zplot = np.array([4, 5, 6, 7,8 , 10])-0.1
axes_evolution[0].set_ylim(-3.5, -1.25)
axes_evolution[2].set_ylim(5e-9, 3e-3)
axes_evolution[0].errorbar(zplot, alpha_g, yerr=dalpha, color='black', fmt='o', markersize=markersize,capsize=4, elinewidth=2)
axes_evolution[1].errorbar(zplot, Mstar_g, yerr=dMstar, color='black', fmt='o', markersize=markersize,capsize=4, elinewidth=2)
axes_evolution[2].errorbar(zplot, phi_g, yerr=dphi, color='black', fmt='o', markersize=markersize,capsize=4, elinewidth=2)

axes_evolution[0].set_ylabel(r'$\alpha$', fontsize=fontsize)
axes_evolution[1].set_ylabel(r'$\mathrm{M_{UV}}$', fontsize=fontsize)
axes_evolution[2].set_ylabel(r'$\mathrm{\phi^* [dex^{-1} cMpc^{-3}]}$', fontsize=fontsize)
axes_evolution[2].set_yscale('log')
axes_evolution[2].set_xlabel(r'$\mathrm{z}$', fontsize=fontsize)

for i in [0,1,2]: 
    axes_evolution[i].set_xlim(3.5, 10.5)
for i in [0, 1]:    
    axes_evolution[i].yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
axes_evolution[2].yaxis.set_major_locator(pl.FixedLocator([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]))
#axes_evolution[2].yaxis.set_major_locator(MaxNLocator(prune='upper'))
fig_evolution.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

#axes_evolution[2].scatter(zplot, phi_g, color='black')
#axes_evolution[1].scatter(zplot, Mstar_g, color='black')
#axes_evolution[0].scatter(zplot, alpha_g, color='black')

#----------------------------------------------
#--------- PAPER FIGURE -----------------------
#----------------------------------------------

left  = 0.12  # the left side of the subplots of the figure
right = 0.99    # the right side of the subplots of the figure
bottom = 0.125   # the bottom of the subplots of the figure
top = 0.99      # the top of the subplots of the figure
wspace = 0.001   # the amount of width reserved for blank space between subplots
hspace = 0.001   # the amount of height reserved for white space between subplots

fig_paper, axes_paper = pl.subplots(2, 3, sharex=True, sharey=True, figsize=(9, 6))
fig_paper.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
nbins = {'001818':10, '001380':10, '001093':8, '000891':8, '000744':5, '000547':4}
axes_paper = axes_paper.ravel()

#---------------------------------------------------

#for step in s:
for j, step in enumerate(s):
    z = redshift[step]
    Mstar_guess = Mstar_g[ind[step]]
    sigma = dMstar[ind[step]]
    alpha_guess = alpha_g[ind[step]]
    phi_guess = phi_g[ind[step]]

    np.random.seed(123)

    data = np.genfromtxt('cosmo25/cosmo25p.768sg1bwK1C52.'  + step + '.tipsy.kroupa.fuv.mag1', 
                    names=['mass', 'lum', 'mag', 'grp'], dtype=['float32', 'float32', 'float32', 'int32'],
                    skip_header=1, usecols=[0, 1, 2, 3])
    data = data[~np.isnan(data['mag'])]
    mags = data['mag']*(0.6777/0.7)

    lim = np.genfromtxt('cosmo25/massMagsComplete.' + step + '.' + IMF + '.txt', 
                    dtype = ['int32', 'float32', 'float32'], 
                    names = ['comp', 'mass', 'mag'], skip_header=1)
    perComp = 50
    magCompThreshold = lim['mag'][lim['comp'] == perComp]
    magsfill = np.load('cosmo25/magsfill.'+step + '.' + IMF + '.npy')*(0.6777/0.7)
    allmags = np.concatenate((mags, magsfill))#[magsfill <= magCompThreshold]))
    A1600 = np.load('cosmo25/extinction.' + step + '.kroupa.npy')
    allmags = allmags + A1600 + 0.2 
    mags = allmags[allmags <= magCompThreshold]
    print(magCompThreshold)
    N = len(mags)

    ndim = 2
    #for ar in [Mstar_guess, alpha_guess]:
    #    ss = [np.random.normal(loc=ar, scale=np.abs(ar)/10.) for i in range(2000)]
    #    fig = pl.figure()
    #    pl.hist(ss)
    #    pl.show()
    pos = [[np.random.normal(loc=Mstar_guess, scale=np.abs(Mstar_guess)*fracDeviation), 
            np.clip(np.random.normal(loc=alpha_guess, scale=np.abs(alpha_guess)*fracDeviation), -4., 0.)] for i in range(nwalkers)]
    print(pos)
    #pl.clf()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(mags,), a=2.)

# Clear and run the production chain.
    print("Running MCMC...")
    sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
    print("Done.")
    print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

    # Estimate the integrated autocorrelation time for the time series in each
    # parameter.
    fig = pl.figure()
    print("Autocorrelation time:", sampler.get_autocorr_time())
    for i in range(nwalkers): pl.scatter(np.arange(len(sampler.lnprobability[i,150:])), sampler.lnprobability[i,150:])
    fig.savefig('logprob.' + step + '.png')
    #pl.show()
    #pl.clf()
    fig, axes = pl.subplots(2, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
    axes[0].yaxis.set_major_locator(MaxNLocator(5))
    axes[0].axhline(Mstar_guess, color="#888888", lw=2)
    axes[0].set_ylabel("$Mstar$")
    axes[0].grid(True)

    axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    axes[1].axhline(alpha_guess, color="#888888", lw=2)
    axes[1].set_ylabel("$alpha$")
    axes[1].grid(True)

    fig.tight_layout(h_pad=0.0)
    fig.savefig("lf-time."+step+".png")

# Make the triangle plot.
    burnin = 150
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    phistar = []
    for Muv, alpha in samples:
        norm = integrate.quad(schecter, np.min(mags), np.max(mags), args=(Muv, 1.0, alpha))[0]
        phistar.append(N/(norm*(25.*(0.6777/0.7))**3.))


    input = np.vstack(([samples[:,0], samples[:,1], np.log10(np.array(phistar))])).T
    fig = corner.corner(input, labels=["$Mstar$", "$alpha$", "$phistar$"],
                      truths=[Mstar_guess, alpha_guess, np.log10(phi_guess)])
    fig.savefig("lf-triangle."+step+".png")

# Plot some samples onto the data.
    xl = np.linspace(np.min(mags), np.max(mags), 100)
    #pl.figure()
    #pl.clf()
    fig, ax = pl.subplots()
    plotMcmc(ax, z)
    plotMcmc(axes_paper[j], z)
    for Muv, alpha, lnphi in input[np.random.randint(len(samples), size=nsamples)]:
        #norm = integrate.quad(schecter, np.min(mags), np.max(mags), args=(Muv, 1.0, alpha))[0]
        ax.plot(xl, schecter(xl, Muv, 10.**lnphi, alpha), color=color[z], alpha=0.05)
        axes_paper[j].plot(xl, schecter(xl, Muv, 10.**lnphi, alpha), color=color[z], alpha=0.05)
    p = pl.Rectangle((-14,2e-5), 0, 0, alpha=0.5, color=color[z], label='z~'+str(z))
    axes_paper[j].add_patch(p)
    #norm = integrate.quad(schecter, np.min(mags), np.max(mags), args=(Mstar_guess, 1.0, alpha_guess))[0]
    #ax.plot(xl, schecter(xl, Mstar_guess, phi_guess, alpha_guess), color="k", lw=2, alpha=0.8)

    hist, bins = np.histogram(mags, bins=nbins[step])
    normalization = (bins[1:]-bins[:-1])*(25.*(0.6777/0.7))**3.
    x =  (bins[1:]+bins[:-1])/2.
    err = 1. + (hist + 0.75)**0.5
    ax.errorbar(x, hist/normalization, yerr=err/normalization, 
           fmt='o', color=color[z], markersize=markersize, capsize=4, elinewidth=2)
    ax.scatter(x, hist/normalization, color=color[z], s=40)
    ax.grid(True)
    ax.set_yscale('log')
    ax.set_xlabel(r"$\mathrm{M_{UV}}$", fontsize=20)
    ax.set_ylabel(r"$\mathrm{\phi \;[mag^{-1} cMpc^{-3}]}$", fontsize=20)
    ax.set_ylim(8e-6, 0.3)
    ax.set_xlim(-13.25, -23)
    pl.tight_layout()
    fig.savefig('lf-mcmc.'+step+'.png')
    axes_paper[j].errorbar(x, hist/normalization, yerr=err/normalization, 
           fmt='o', color=color[z], markersize=markersize,capsize=4, elinewidth=2)
    axes_paper[j].scatter(x, hist/normalization, color=color[z], s=40)
    axes_paper[j].grid(True)
    axes_paper[j].set_yscale('log')
    if j in [3, 4, 5]: axes_paper[j].set_xlabel(r"$\mathrm{M_{UV}}$", fontsize=fontsize)
    if j in [0,3]: 
        axes_paper[j].set_ylabel(r"$\mathrm{\phi \;[mag^{-1} cMpc^{-3}]}$", fontsize=fontsize)
        axes_paper[j].yaxis.set_major_locator(pl.FixedLocator([1e-4, 1e-3, 1e-2, 1e-1]))
    axes_paper[j].set_ylim(8e-6, 0.4)
    axes_paper[j].set_xlim(-13.25, -23)
    fig.tight_layout()

    M_mcmc, alpha_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
    v = np.percentile(np.log10(phistar), [16, 50, 84])
    phi_mcmc = [10.**v[1], 10.**v[2]-10.**v[1], 10.**v[1]-10.**v[0]]

    #axes_evolution[0].scatter(z, alpha_mcmc[0], color=color[z])
    #axes_evolution[1].scatter(z, M_mcmc[0], color=color[z])
    #axes_evolution[2].scatter(z, phi_mcmc[0], color=color[z])
    axes_evolution[0].errorbar(z, alpha_mcmc[0], yerr=[[alpha_mcmc[2]], [alpha_mcmc[1]]], color=color[z], fmt='o', markersize=markersize,capsize=4, elinewidth=2)
    axes_evolution[1].errorbar(z, M_mcmc[0],     yerr=[[M_mcmc[2]]    , [M_mcmc[1]]],     color=color[z], fmt='o', markersize=markersize,capsize=4, elinewidth=2)
    axes_evolution[2].errorbar(z, phi_mcmc[0],   yerr=[[phi_mcmc[2]],   [phi_mcmc[2]]],   color=color[z], fmt='o', markersize=markersize,capsize=4, elinewidth=2)

    print("""Phistar {0[0]} +{0[1]} -{0[2]} (truth {1})""".format(phi_mcmc, phi_guess))
    print("""MCMC result:
        Mstar = {0[0]} +{0[1]} -{0[2]} (truth: {1})
        alpha = {2[0]} +{2[1]} -{2[2]} (truth: {3})"""
        .format(M_mcmc, Mstar_guess, alpha_mcmc, alpha_guess))
for i in [0, 1, 2]: axes_evolution[i].grid(True)
fig_evolution.savefig('paramEvolution.png')
fig_paper.savefig('lfEvolution.png')
"""    alpha_meas.append(alpha_mcmc[0])
    alpha_plus.append(alpha_mcmc[1])
    alpha_minus.append(alpha_mcmc[2])

    Mstar_meas.append(M_mcmc[0])
    Mstar_plus.append(M_mcmc[1])
    Mstar_minus.append(M_mcmc[2])

    phi_meas.append(phi_mcmc[0])
    phi_plus.append(phi_mcmc[1])
    phi_minus.append(phi_mcmc[2])
    print(alpha_mcmc[0], alpha_mcmc[1], alpha_mcmc[2])"""
