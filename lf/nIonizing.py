
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import pdb
import matplotlib as mpl

mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15

def schecter(mag, Mstar, phi, alpha):
        return 2./5*phi*np.log(10)*(10.**(2./5*(Mstar-mag)))**(alpha+1.)*np.exp(-10.**(2./5*(Mstar-mag)))
def ionizing(mag, Mstar, phi, alpha, fesc):
        return schecter(mag, Mstar, phi, alpha)*10**(45.3247 -0.4654*mag)*np.clip(10.**(0.53964*mag + 7.7528), 0, 0.414)
def ionizing_new(mag, Mstar, phi, alpha, fesc):
        return schecter(mag, Mstar, phi, alpha)*10**(45.3247 -0.4654*mag)*np.clip(10**(0.40019368*mag + 4.77804037), 0,0.0555385431664 ) #np.clip(10.**(0.53964*mag + 7.7528), 0, 0.414)

def ionizingModel(mag, Mstar, phi, alpha, fesc):
        return schecter(mag, Mstar, phi, alpha)*2e25*10**(0.4*(51.63-mag))*fesc #*8.97e45*10**(-0.4*mag)*fesc           
def nionCrit(CHII, T0, z):
        return 3e50*(CHII/3.)*(T0/2e4)**-0.7*((1. + z)/7.)**3.

#Bouwens concordence
ztable = [6, 7, 8, 9, 10]
nlower = 10.**np.array([50.77, 50.71, 50.58, 50.39, 50.15])
nupper = 10.**np.array([51.14, 50.92, 50.72, 50.62, 50.55]) 

#Kuhlen concordence
ztablek = [4, 4.2, 5.0, 6.0]
nlowerk = np.array([1.3, 1.3, 1.7, 1])*1e50
nupperk = np.array([5.4, 6.4, 6.9, 2.6])*1e50

params = np.load('lfParams.kroupa.npz')
Mstar = [-20.73, -20.81, -21.13, -21.03, -20.89, -20.92]
phi = np.array([14.1, 8.95, 1.86, 1.57, 0.72, 0.08])*1e-4
alpha = [-1.56, -1.67, -2.02, -2.03, -2.36, -2.27]

minmags = np.linspace(0, -25, 100)
ans = np.zeros(100)
z = ['4', '5', '6', '7', '8', '10']
color = ['blue', 'green', 'red', 'purple', 'orange', 'black']

CHII = 3
T0 = 2e4
zz = np.linspace(4, 10, 100)
fesc = 0.13
size = 150
files = ['massMagsComplete.001818.kroupa.txt','massMagsComplete.001380.kroupa.txt','massMagsComplete.001093.kroupa.txt','massMagsComplete.000891.kroupa.txt','massMagsComplete.000744.kroupa.txt', 'massMagsComplete.000547.kroupa.txt']

#plt.plot(z, nionCrit)          
#plt.show()                     

figall, axall = plt.subplots()
fig, ax = plt.subplots()
for j in range(6):
	data = np.genfromtxt(files[j], names=['completeness', 'logmass', 'mag'], dtype=['float32', 'float32', 'float32'], skip_header=1)
	ans = integrate.quad(ionizing, -50, data['mag'][0], args=(Mstar[j], params['phi'][j], params['alpha'][j], fesc))
	ans50 = ans[0]
	ans98 = integrate.quad(ionizing, -50, data['mag'][1], args=(Mstar[j], params['phi'][j], params['alpha'][j], fesc))[0]
	ansnew = integrate.quad(ionizing_new, -50, data['mag'][0], args=(Mstar[j], params['phi'][j], params['alpha'][j], fesc))[0]
	print z[j], data['mag'], ans50, ans98
	ax.scatter(z[j], ans50, c='black', s=100, lw=0)
	#ax.scatter(z[j], ans98, c='grey', s=size, lw=0)
	#ax.scatter(z[j], ansnew, c='white', s=size, lw=2)
	print z[j], params['phi'][j], params['alpha'][j]
	#pdb.set_trace()
zplot = np.linspace(4, 10, 100)
ax.fill_between(zplot, nionCrit(1, T0, zplot), nionCrit(5, T0, zplot), alpha=0.2, color='green')
ax.plot(zplot, nionCrit(3, T0, zplot), 'green', linewidth=2)
ax.fill_between(ztable, nlower, nupper, alpha=0.2, color='blue', label='Bouwens et al 2015')
#ax.fill_between(ztablek, nlowerk, nupperk, alpha=0.4, color='blue', label='Kuhlen et al 2012')
ax.scatter(4.75, 10**50.99, marker=(5,1), s=200, c='blue', label='Bolton et al 2013')
ylow = 6.305003705032841e50
yhigh = 1.776991482382382e+51
ax.errorbar([4.75], [10**50.99], yerr=[[ylow], [yhigh]], lw=2)
ax.axvspan(5.9, 6.5, alpha=0.2, hatch='//', color='red', label='Reionization Complete')
ax.set_ylim(5.5e49, 2e51)
ax.set_xlim(3.9, 10.1)
ax.set_yscale('log')
ax.set_ylabel(r'$\rm{log_{10}\ \dot{N}_{ion} [s^{-1} cMpc^{-3}]}$ ', fontsize=20)
ax.set_xlabel(r'$\rm{Redshift}$', fontsize=20)
p = plt.Rectangle((0,0), 0, 0, color='blue', alpha = 0.2, label='Bouwens et al 2015')
#pp = plt.Rectangle((0,0), 0, 0, color='blue', alpha = 0.4, label='Kuhlen et al 2012')
ppp = plt.Rectangle((0,0), 0, 0, color='green', alpha=0.2, label='Critical Rate')
ax.add_patch(ppp)
ax.add_patch(p)
#ax.add_patch(pp)
ax.grid(True, which='both')
plt.legend(loc='upper right')

plt.show()
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
