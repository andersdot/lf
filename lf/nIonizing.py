
import numpy as np
from scipy import integrate
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb
import efMuv

plt.style.use('seaborn-talk')
mpl.rcParams['xtick.labelsize']=30
mpl.rcParams['ytick.labelsize']=30
mpl.rcParams.update({'figure.autolayout': True})

def schecter(mag, Mstar, phi, alpha):
        return 2./5*phi*np.log(10)*(10.**(2./5*(Mstar-mag)))**(alpha+1.)*np.exp(-10.**(2./5*(Mstar-mag)))

def schecterLumMod3(mag, Mstar, phi, alpha, Mturn, beta, Mstop, gamma):
    L_Lstar = 10.**(0.4*(Mstar-mag))
    dL = 0.4*np.log(10)*L_Lstar
    L_Lturn = 10.**(0.4*(Mturn - mag))
    L_Lstop = 10.**(0.4*(Mstop - mag))
    return schecter(mag, Mstar, phi, alpha)*(1. - np.exp(-(L_Lturn**beta)))*np.exp(-(L_Lstop**gamma))

def ionizing(mag, Mstar, phi, alpha, fesc):
        return schecter(mag, Mstar, phi, alpha)*10**(45.3247 -0.4654*mag)*np.clip(10.**(0.53964*mag + 7.7528), 0, 0.414)
def ionizing_new(mag, Mstar, phi, alpha, slope, yint):
    #fesc = m*xl + b
    #fesc[xl > np.max(allmags)] = m*np.max(allmags) + b
    #fesc[fesc > 0] = 0
    #fesc = 10.**fesc
    
    #print mag, 10**(slope*mag + yint), np.max(efMuvmag), 10**(slope*np.max(efMuvmag) + yint), np.clip(10**(slope*mag + yint), 0, 10.**np.max(slope*efMuvmag + yint))
    return schecter(mag, Mstar, phi, alpha)*10**(45.3247 -0.4654*mag)*np.clip(10**(slope*mag + yint), 0, 1) #10.**np.max(slope*efMuvmag + yint)) #np.clip(10.**(0.53964*mag + 7.7528), 0, 0.414)

def ionizingModel(mag, Mstar, phi, alpha, fesc):
        return schecter(mag, Mstar, phi, alpha)*2e25*10**(0.4*(51.63-mag))*fesc #*8.97e45*10**(-0.4*mag)*fesc           
def nionCrit(CHII, T0, z):
        return 3e50*(CHII/3.)*(T0/2e4)**-0.7*((1. + z)/7.)**3.
def nionMuv(mag, slope, yint):
    return 10**(45.3247 -0.4654*mag)*np.clip(10**(slope*mag + yint), 0, 10.**np.max(slope*efMuvmag + yint))
def trec(CHII, z, nh):
    return 1./(CHII*alpha*(1+Y/(4*X))*nh*(1+z)**3)

nstar = 8
efMuv = np.load('efMuv.samples.npy')
fontsize = 25
ymin = 1e49 #5.5e49
efMuvmag = np.load('efMuv.npz')
efMuvmag = efMuvmag['mags']
efMuvSlope = efMuv[:,0]
efMuvYint = efMuv[:,1]
#print efMuvSlope[0]
#Bouwens concordence
ztable = [6, 7, 8, 9, 10]
nlower = 10.**np.array([50.77, 50.71, 50.58, 50.39, 50.15])
nupper = 10.**np.array([51.14, 50.92, 50.72, 50.62, 50.55]) 

#Kuhlen concordence
ztablek = [4, 4.2, 5.0, 6.0]
nlowerk = np.array([1.3, 1.3, 1.7, 1])*1e50
nupperk = np.array([5.4, 6.4, 6.9, 2.6])*1e50

boltonz = [2.4, 3.2, 4.0, 4.75]
boltondot = np.array([-0.046, -0.211, -0.139, -0.014])
boltondplus = np.array([0.444, 0.445, 0.451, 0.454])
boltondminus = np.array([0.372, 0.352, 0.346, 0.355])

#params = np.load('lfParams.kroupa.npz')
params = np.load('lf.parameters.mcmc.npz')
Mstar = [-20.73, -20.81, -21.13, -21.03, -20.89, -20.92]
phi = np.array([14.1, 8.95, 1.86, 1.57, 0.72, 0.08])*1e-4
alpha = [-1.56, -1.67, -2.02, -2.03, -2.36, -2.27]

minmags = np.linspace(0, -25, 100)
ans = np.zeros(100)
z = ['4', '5', '6', '7', '8', '10']
#color = ['blue', 'green', 'red', 'purple', 'orange', 'black']
color = {'4':'purple', '5':'blue', '6':'cyan', '7':'green', '8':'orange', '10':'red'}
CHII = 3
T0 = 2e4
zz = np.linspace(4, 10, 100)
fesc = 0.13
size = 150
#files = ['massMagsComplete.001818.kroupa.txt','massMagsComplete.001380.kroupa.txt','massMagsComplete.001093.kroupa.txt','massMagsComplete.000891.kroupa.txt','massMagsComplete.000744.kroupa.txt', 'massMagsComplete.000547.kroupa.txt']
files = []
#plt.plot(z, nionCrit)          
#plt.show()                     

fig, ax = plt.subplots()
plt.tight_layout()
figconv, axconv = plt.subplots()
figone, axone = plt.subplots()
figtwo, axtwo = plt.subplots()
figthree, axthree = plt.subplots()
figac, axac  = plt.subplots()

axes = [axconv, axone, axtwo, axthree, axac]

zplot = np.linspace(2.5, 11, 100)
axone.axvspan(5.9, 6.5, alpha=0.2, hatch='//', color='red', label='Reionization Complete')
axone.set_ylim(ymin, 4e51)
axone.set_xlim(3, 10.1)
axone.set_yscale('log')
axone.set_ylabel(r'$\rm{log_{10}\ \dot{N}_{ion} [s^{-1} cMpc^{-3}]}$ ', fontsize=fontsize)
axone.set_xlabel(r'$\rm{Redshift}$', fontsize=fontsize)
axone.legend(loc='upper right')


axtwo.axvspan(5.9, 6.5, alpha=0.2, hatch='//', color='red', label='Reionization Complete')
axtwo.fill_between(zplot, nionCrit(1, T0, zplot), nionCrit(5, T0, zplot), alpha=0.2, color='green')
axtwo.plot(zplot, nionCrit(3, T0, zplot), 'green', linewidth=2, linestyle='--')
ppp = plt.Rectangle((0,0), 0, 0, color='green', alpha=0.2, label=r'$\rm{\dot{N}_{ion, crit}}$')
axtwo.add_patch(ppp)
axtwo.set_ylim(ymin, 4e51)
axtwo.set_xlim(3, 10.1)
axtwo.set_yscale('log')
axtwo.set_ylabel(r'$\rm{log_{10}\ \dot{N}_{ion} [s^{-1} cMpc^{-3}]}$ ', fontsize=fontsize)
axtwo.set_xlabel(r'$\rm{Redshift}$', fontsize=fontsize)
axtwo.legend(loc='upper right')
ylow = 6.305003705032841e50
yhigh = 1.776991482382382e+51

axthree.axvspan(5.9, 6.5, alpha=0.2, hatch='//', color='red', label='Reionization Complete')
axthree.fill_between(zplot, nionCrit(1, T0, zplot), nionCrit(5, T0, zplot), alpha=0.2, color='green')
axthree.plot(zplot, nionCrit(3, T0, zplot), 'green', linewidth=2, linestyle='--')
axthree.fill_between(ztable, nlower, nupper, alpha=0.2, color='blue', label='Bouwens et al 2015')
axthree.fill_between(boltonz, 10.**(boltondot - boltondminus)*1e51, 10.**(boltondot + boltondplus)*1e51, color='blue', alpha=0.4)
#axthree.scatter(4.75, 10**50.99, marker=(5,1), s=200, c='blue', label='Bolton et al 2013')
#axthree.errorbar([4.75], [10**50.99], yerr=[[ylow], [yhigh]], color='blue', lw=2)
ppp = plt.Rectangle((0,0), 0, 0, color='green', alpha=0.2, label=r'$\rm{\dot{N}_{ion, crit}}$')
axthree.add_patch(ppp)
pp = plt.Rectangle((0,0), 0, 0, color='blue', alpha = 0.4, label='Becker+Bolton 2013')
axthree.add_patch(pp)
axthree.legend()
axthree.set_ylim(ymin, 4e51)
axthree.set_xlim(3, 10.1)
axthree.set_yscale('log')
axthree.set_ylabel(r'$\rm{log_{10}\ \dot{N}_{ion} [s^{-1} cMpc^{-3}]}$ ', fontsize=fontsize)
axthree.set_xlabel(r'$\rm{Redshift}$', fontsize=fontsize)
axthree.legend(loc='upper right')
"""
axone.savefig('nIon.1.png')
axtwo.savefig('nIon.2.png')
axthree.savefig('nIon.3.png')
"""
axac.set_xlim(3, 10.5)
axac.set_ylim(0.85, 1)
simArtist = plt.Line2D([],[], color='k', marker='o', linestyle='', label='Romulus Sim')
ax.add_patch(simArtist)
ax.fill_between(zplot, nionCrit(1, T0, zplot), nionCrit(5, T0, zplot), alpha=0.2, color='green')
ax.plot(zplot, nionCrit(3, T0, zplot), 'green', linewidth=2, linestyle='--')
ax.fill_between(ztable, nlower, nupper, alpha=0.2, color='blue', label='Bouwens et al 2015')


ax.fill_between(boltonz, 10.**(boltondot - boltondminus)*1e51, 10.**(boltondot + boltondplus)*1e51, color='blue', alpha=0.4)
#ax.scatter(4.75, 10**50.99, marker=(5,1), s=200, c='blue', label='Bolton et al 2013')
#ylow = 6.305003705032841e50
#yhigh = 1.776991482382382e+51
#ax.errorbar([4.75], [10**50.99], yerr=[[ylow], [yhigh]], color='blue', lw=2)
ax.axvspan(5.9, 6.5, alpha=0.2, hatch='//', color='red', label='Reionization Complete')
ax.set_ylim(ymin, 4e51)
ax.set_xlim(2.5, 10.1)
ax.set_yscale('log', nonposy='clip')
ax.set_ylabel(r'$\rm{log_{10}\ \dot{N}_{ion} [s^{-1} cMpc^{-3}]}$ ', fontsize=fontsize)
ax.set_xlabel(r'$\rm{Redshift}$', fontsize=fontsize)
plt.tight_layout()
#p = plt.Rectangle((0,0), 0, 0, color='blue', alpha = 0.2, label='Bouwens et al 2015')
pp = plt.Rectangle((0,0), 0, 0, color='blue', alpha = 0.4, label='Becker+Bolton 2013')
ppp = plt.Rectangle((0,0), 0, 0, color='green', alpha=0.2, label=r'$\rm{\dot{N}_{ion, crit}}$')
ax.add_patch(ppp)
ax.add_patch(pp)

#ax.add_patch(p)
ax.grid(True, which='major')
axone.grid(True, which='major')
axtwo.grid(True, which='major')
axthree.grid(True, which='major')


ansnew = []
for j in range(6):
    lfsamples = np.load('lf.samples.z' + z[j] +'.nstar8.Schecter.npz') #lf.samples.z' + z[j] + '.npz')

    mags = np.load('z' + z[j] + '.fuv.npy')
    filename = 'massMagsComplete.z' + z[j] + '.8.txt'
    data = np.genfromtxt(filename, names=['completeness', 'logmass', 'mag'], dtype=['float32', 'float32', 'float32'], skip_header=1)
    print data['mag']
    #ans = integrate.quad(ionizing, -50, data['mag'][0], args=(params['mstar'][j], params['phi'][j], params['alpha'][j], fesc))
    #ans50 = ans[0]
    #ans98 = integrate.quad(ionizing, -50, data['mag'][1], args=(params['mstar'][j], params['phi'][j], params['alpha'][j], fesc))[0]
    ratio = []
    for i in np.random.random_integers(0, high=len(lfsamples['mstar'])-1, size=20):
        for k in np.random.random_integers(0, high=len(efMuvSlope)-1, size=20): 
            integ = integrate.quad(ionizing_new, -25, data['mag'][0],
                args=(lfsamples['mstar'][i], lfsamples['phi'][i], lfsamples['alpha'][i], efMuvSlope[k], efMuvYint[k]))
                #args=(params['mstar'][j], params['phi'][j], params['alpha'][j], efMuvSlope[k], efMuvYint[k]))
            integ2 = integrate.quad(ionizing_new, -17, data['mag'][0],
                args=(lfsamples['mstar'][i], lfsamples['phi'][i], lfsamples['alpha'][i], efMuvSlope[k], efMuvYint[k]))
            ansnew.append(integ[0])
            #magx = np.linspace(-25, data['mag'][0], 10000)
            #magmin = np.linspace(-17, data['mag'][0], 10000)
            #cumsum = integrate.cumtrapz(ionizing_new(magx, params['mstar'][j], 
            #        params['phi'][j], params['alpha'][j], efMuvSlope[k], efMuvYint[k]))
            #cumsummin = integrate.cumtrapz(ionizing_new(magmin, params['mstar'][j], 
            #        params['phi'][j], params['alpha'][j], efMuvSlope[k], efMuvYint[k]))
            ratio.append(integ2[0]/integ[0])

    (_, caps, _) = axac.errorbar(np.int(z[j]), np.mean(ratio), yerr=np.std(ratio), color=color[z[j]], marker='o', elinewidth=4, capsize=4, markersize=20)
    for cap in caps: cap.set_markeredgewidth(4)
    #for l in np.random.random_integers(0, high=len(efMuvSlope), size=50):
    #    magx = np.linspace(-25, data['mag'][0], 1000)
    #    nionConv = nionMuv(magx, efMuvSlope[l], efMuvYint[l])
    #    axconv.plot(magx, nionConv, color=color[z[j]])
    #    axconv.set_yscale('log')
        #axac.plot(np.linspace(-25, data['mag'][0], len(cumsum)), cumsum/np.max(cumsum))
    #print z[j], data['mag'], ans50, ans98
    #ax.scatter(z[j], ans50, c='black', s=100, lw=0)
    #ax.scatter(z[j], ans98, c='grey', s=size, lw=0)
    print z[j], np.mean(ansnew)
    print np.mean(ansnew), np.std(ansnew)
    n_mcmc = np.percentile(ansnew, [16, 50, 84], axis=0)
    print n_mcmc
    #ax.scatter(z[j], np.mean(ansnew), s=40, c='black')
    try: 
        (_, caps, _) = ax.errorbar(z[j], np.mean(ansnew), yerr=np.std(ansnew), c='black', lw=2, markersize='10', elinewidth=2, marker='o')
        for cap in caps: cap.set_markeredgewidth(2)
    except AssertionError: 
        print z[j], np.mean(ansnew)
        (_, caps, _) = ax.errorbar(10, np.mean(ansnew), yerr=4e50, c='black', lw=2, markersize='10', elinewidth=2, marker='o') 
        for cap in caps: cap.set_markeredgewidth(2)
    #print z[j], params['phi'][j], params['alpha'][j]
    #pdb.set_trace()

#talk plots
axac.set_ylabel(r'$\rm{\dot{N}_{ion}[M_{UV} > -17] / \dot{N}_{ion}}$', fontsize=fontsize)
axac.set_xlabel('z', fontsize=fontsize)
ax.legend(loc='upper right')
plt.gcf().tight_layout()
plt.show()
figone.savefig('nIon.1.png')
figtwo.savefig('nIon.2.png')
figthree.savefig('nIon.3.png')
fig.savefig('nIon.4.png')
figac.savefig('nIon.lowmass.png')
"""
fig, ax = plt.subplots()
for j in range(5):
    for i in range(100):ans[i] = integrate.quad(ionizing, -50, minmags[i], args=(Mstar[j], params['phi'][j], params['al\
pha'][j], fesc))[0]
    ax.plot(minmags, ans, label='z~'+z[j], linewidth=2, color=color[j])
    ax.fill_between(minmags, np.zeros(100) + nionCrit(1, 1e4, int(z[j])), np.zeros(100) + nionCrit(3, 3e4, int(z[j])), \
alpha=0.3, color=color[j])
    crit = nionCrit(2, 2e4, int(z[j]))
    ind = np.where(np.abs(ans - crit) == np.min(np.abs(ans - crit)))
    ax.plot(np.zeros(100) + minmags[ind], np.logspace(26, 64, 100), color=color[j], linewidth=2)
    print minmags[ind]
for j in range(5):
    for i in range(100): ans[i] = integrate.quad(ionizingModel, -50, minmags[i], args=(Mstar[j], phi[j], alpha[j], fesc\
))[0]
    ax.plot(minmags, ans, linewidth=2, color=color[j], ls=':')
    crit = nionCrit(2, 2e4, int(z[j]))
    ind = np.where(np.abs(ans - crit) == np.min(np.abs(ans - crit)))
    ax.plot(np.zeros(100) + minmags[ind], np.logspace(26, 64, 100), color=color[j], linewidth=2, ls=':')
    print minmags[ind]

ax.set_ylabel('N Ionizing Photons/s')
ax.set_xlabel('limiting M$_{UV}$') 
ax.set_yscale('log')
plt.legend(loc='lower right')
plt.show()
"""



"""
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def schecter(mag, Mstar, phi, alpha):
    return 2./5*phi*np.log(10)*(10.**(2./5*(Mstar-mag)))**(alpha+1.)*np.exp(-10.**(2./5*(Mstar-mag)))
def ionizing(mag, Mstar, phi, alpha):
    return schecter(mag, Mstar, phi, alpha)*6e39*0.28**mag

def ionizingModel(mag, Mstar, phi, alpha):
    return schecter(mag, Mstar, phi, alpha)*8.97e45*10**(-0.4*mag)


params = np.load('lfParams.kroupa.npz')
Mstar = [-20.73, -20.81, -21.13, -21.03, -20.89]
phi = np.array([14.1, 8.95, 1.86, 1.57, 0.72])*1e-4

minmags = np.linspace(0, -25, 100)
ans = np.zeros(100)
z = ['4', '5', '6', '7', '8']
color = ['blue', 'green', 'red', 'purple', 'black']


fig, ax = plt.subplots()
for j in range(5):
    for i in range(100):ans[i] = integrate.quad(ionizing, -50, minmags[i], args=(Mstar[j], phi[j], params['alpha'][j]))[0]
    ax.plot(minmags, ans, label='z~'+z[j], linewidth=2, color=color[j])
for j in range(5):
    for i in range(100):ans[i] = integrate.quad(ionizing, -50, minmags[i], args=(Mstar[j], params['phi'][j], params['alpha'][j]))[0]
    ax.plot(minmags, ans, label='z~'+z[j], linewidth=2, color=color[j], ls='--')
for j in range(5):
    for i in range(100): ans[i] = integrate.quad(ionizingModel, -50, minmags[i], args=(Mstar[j], phi[j], params['alpha'][j]))[0]
    ax.plot(minmags, ans, label='z~'+z[j], linewidth=2, color=color[j], ls=':')
plt.yscale('log')
plt.legend()
plt.show()
"""
#ax.fill_between(ztablek, nlowerk, nupperk, alpha=0.4, color='blue', label='Kuhlen et al 2012')
