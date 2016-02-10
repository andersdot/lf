import nbdpt
import numpy as np
import pynbody
import matplotlib.pyplot as plt
import pdb
import nbdpt.readparam as rdp
import nbdpt.readstat as rds
from astroML.plotting import hist as astroMLhist
from scipy import integrate

import matplotlib as mpl
label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

redshift = {'000547':10, '000744':8, '000891':7, '001093':6, '001380':5, '001818':4}
ind = {4:0, 5:1, 6:2, 7:3, 8:4, 10:5}

mcmcz = {10:None, 8:'z8', 7:'z7', 6:'z6', 5:'z5', 4:'z4'}
color = {4:'purple', 5:'blue', 6:'cyan', 7:'green', 8:'orange', 10:'red'}
span = {4:[-17.5, -22.5], 
	5:[-17.5, -22.5], 
	6:[-17.5, -22], 
	7:[-18,   -22], 
	8:[-18.5, -21.5],
	10:[-18.5, -21.5]}

b195 = {4:-2.0, 5:-2.08, 6:-2.2, 7: -2.27, 8:-2.27, 10:-2.27}
dbdm = {4:-0.11, 5:-0.16, 6:-0.15, 7:-0.21, 8:-0.21, 10:-0.21}


alpha = [-1.56, -1.67, -2.02, -2.03, -2.36, -2.27]
phi = np.array([14.1, 8.95, 1.86, 1.57, 0.72, 0.08])*1e-4
Mstar = [-20.73, -20.81, -21.13, -21.03, -20.89, -20.92]

dalpha = [0.06, 0.05, 0.10, 0.21, 0.54, 1.0]
dphi   = np.array([2.05, 1.92, 0.94, 1.45, 2.52, 0.04])*1e-4
dMstar = [0.09, 0.13, 0.31, 0.50, 1.08, 0]

#minmagnitude = np.array(Mstar) + 2. #{4:-19.76, 5:-19.88, 6:-20.09, 7:-20.10, 8:-19.89}

#very min magnitude and percent complete 
"""
alpha =            [ -1.64,  -1.78,  -1.91,  -2.06,  -1.86,  -2.25]
phistar = np.array([  1.41,   0.64,   0.33,   0.22,   0.64,   0.024])*1e-3
Mstar =            [-21.07, -21.19, -21.16, -21.04, -19.97, -20.36]

dalpha   =          [0.04, 0.05, 0.09, 0.12, 0.27, 0.0]
plusphi  = np.array([0.23, 0.14, 0.15, 0.14, 0.65, 0.012])*1e-3
minusphi = np.array([0.20, 0.12, 0.10, 0.09, 0.32, 0.008])*1e-3
dMstar   =          [0.08, 0.11, 0.20, 0.26, 0.34, 0.0]
"""



def P(x, a, b):
	return a*x + b

def schecter(mag, Mstar, phi, alpha):
	return 2./5*phi*np.log(10)*(10.**(2./5*(Mstar-mag)))**(alpha+1.)*np.exp(-10.**(2./5*(Mstar-mag)))

def schecterprob_logL(mag, Mstar, phi, alpha, plot=False):
	sum = integrate.quad(schecter, -500., np.max(mag), args=(Mstar, 1.0, alpha))[0]
	#phi = N/(sum*boxsize**3)
	pdf = schecter(mag, Mstar, 1.0, alpha)/sum
	prob = np.sum(np.log(pdf))
	if plot:
		plt.plot(mag, pdf, label=str(alpha))
		plt.errorbar(x*(0.6777/0.7), (hist/normalization), 
			     yerr = err/normalization, 
			     linewidth=2, label=str(prob), drawstyle='steps-mid', color='k')
		
		plt.yscale('log')
		plt.legend()
		plt.show()
	return np.sum(np.log(pdf))
"""
def dust(mag, dbdm, b195):
	return 4.43 + 1.99*(dbdm*(mag+19.5) + b195)
"""

def dustgaus(beta, Muv, dbdm, b195, sigma):
    return np.max([0, 4.43 + 1.99*beta])/(sigma*np.sqrt(2*np.pi))*np.exp(-(beta-(dbdm*(Muv+19.5)+b195))**2./(2.*sigma**2.))

def extinction(mag, muv, ext):
	ind1 = np.where(np.abs(muv - mag) == np.min(np.abs(muv - mag)))[0]
	"""
	diff = muv[ind1] - mag
	if diff >= 0: ind2 = ind1 - 1 
	else: ind2 = ind1 + 1 
	m = (ext[ind2] - ext[ind1])/(muv[ind2] - muv[ind1])
	b = ext[ind2] - m*muv[ind2]
	print mag, muv[ind1], muv[ind2], ext[ind1], ext[ind2], m*mag + b
	"""
	return ext[ind1] 
	
def findLimMags(step, IMF, perComp, dmin):
	lim = np.genfromtxt('massMagsComplete.' + step + '.' + IMF + '.txt', 
			    dtype = ['int32', 'float32', 'float32'], 
			    names = ['comp', 'mass', 'mag'], skip_header=1)
	minmagnitude = np.array(Mstar) + dmin
	minmags = minmagnitude[ind[redshift[step]]]
	maxmags = lim['mag'][lim['comp'] == perComp][0] 
	return minmags, maxmags

def fillMags(step, IMF, grps):
	try: fillmags = np.load('magsfill.'+step + '.' + IMF + '.npy')#*(0.6777/0.7)
	except IOError:
		statpre = 'cosmo25p.768sg1bwK1C52.'
		try: stat = np.load(statpre +step+'.rockstar.stat.npy')
		except IOError: 
			stat = rds.readstat(statpre + step + '.rockstar.stat')
			np.save(statpre + step + '.rockstar.stat', stat)
		a = np.load('slope.' + step + '.' + IMF + '.npy')
		b = np.load('yint.'  + step + '.' + IMF + '.npy')

		wanted   = ~np.in1d(stat['grp'], grps) & (stat['npart'] >= 64)
		addmtots = np.log10(stat[wanted]['mvir'])

		avgs = a[0]*addmtots + b[0]
		sigsplus  =  a[1]*addmtots + b[1] - avgs
		sigsminus = -a[2]*addmtots - b[2] + avgs
		sigs = (sigsplus + sigsminus)/2.
		fillmags = np.random.normal(loc=avgs, scale=sigs, size=len(avgs))
		np.save('magsfill.'+ step + '.' + IMF, fillmags)
	return fillmags

def calcExtinction(allmags, step, IMF, z):
	try: A1600 = np.load('extinction.' + step + '.' + IMF +'.npy')
	except IOError:
		muv = np.linspace(-24, -12, 50)
		ext = np.zeros(len(muv))
		for j in range(len(ext)): ext[j] = integrate.quad(dustgaus, -np.inf, np.inf, args=(muv[j], dbdm[z], b195[z], 0.34))[0]
	#plt.plot(muv, ext)
	#plt.show()
		
		A1600 = np.zeros(len(allmags), dtype=np.float)
		for j in range(len(A1600)): A1600[j] = extinction(allmags[j], muv, ext)
		np.save('extinction.' + step + '.' + IMF, A1600)
	#print 'done with extinction ' + step
	return A1600 

def plotMcmc(ax, z):
	xx = np.arange(span[z][1], span[z][0], 0.1)
	if z == 10:
		plus = schecter(xx, Mstar[ind[z]], phi[ind[z]]+dphi[ind[z]], alpha[ind[z]])
		minus = schecter(xx, Mstar[ind[z]], phi[ind[z]]-dphi[ind[z]], alpha[ind[z]])
		ax.fill_between(xx, plus, minus, color='grey')
	else:
		mcmcdata = np.load('mcmc_chains.npy')
		par = mcmcdata[mcmcz[z]]
		Mstarmcmc = par[0:1000000-1]
		alphamcmc = par[1000000:2000000-1]
		phimcmc   = par[2000000:3000000-1]
		if z in [6, 8]: size=5000
		else: size=500
		pick = np.random.random_integers(0, high=1000000-1, size=size)
		for j in range(len(pick)): ax.plot(xx, schecter(xx, Mstarmcmc[pick[j]], phimcmc[pick[j]], alphamcmc[pick[j]]), linewidth=0.5, color='grey')
	p = plt.Rectangle((-20,1), 0, 0, alpha=0.5, color=color[z], label='z~'+str(z))
	ax.add_patch(p)
	

def plot(filename, subplot=False, i=0, plot=False, restore=False, save=False, IMF='kroupa', perComp = 50, dmin=1.):

	data = np.genfromtxt(filename, dtype=['float32', 'float32', 'float32', 'int32'], names=['mass', 'lum', 'mag', 'grp'], skip_header=1, usecols=[0, 1, 2, 3])
	data = data[~np.isnan(data['mag'])]
	mags = data['mag']*(0.6777/0.7)
	grps = data['grp']
	params = rdp.Params(filename='param')
	step = filename.split('.')[2]
	z = redshift[step]	
	minmag, maxmag = findLimMags(step, IMF, perComp, dmin)
	print minmag, maxmag, step, IMF
	minmag = np.min(mags)
      	magsfill = fillMags(step, IMF, grps)
	#print 'Number filled used for ' + step + ' ' + IMF + ': ', np.sum(magsfill <= maxmag)
	#print 'Number natural used for '+ step + ' ' + IMF + ': ', len(mags)
	allmags = np.concatenate((mags, magsfill))
	A1600 = calcExtinction(allmags, step, IMF, z)
	allmags = allmags + A1600	#why + 0.2, I think comparison with paper
	
	N = np.sum((allmags >= minmag) & (allmags <= maxmag))
	Ms = Mstar[ind[z]]
	n = 100
	if i == 4: width = 15.
	else:width = 15.
	
	a = np.linspace(alpha[ind[z]]-dalpha[ind[z]]*width, alpha[ind[z]]+dalpha[ind[z]]*width, n)
	
	logL = np.zeros(len(a), dtype=np.float)
	for k in range(len(a)):
		logL[k] = schecterprob_logL(allmags[(allmags <= maxmag) & (allmags >= minmag)], Ms, 1.0, a[k])
	logL -= np.max(logL[~np.isnan(logL)])
	logL = np.exp(logL) # - logL.max())
	logL /= np.sum(logL[~np.isnan(logL)])
	max = np.where(logL == np.max(logL[~np.isnan(logL)]))[0]
	
	try: sum = integrate.quad(schecter, minmag, maxmag, args=(Ms, 1.0, a[max]))[0]
	except: 
		sum = 0.
		print a[max]
		pdb.set_trace()
	print N, sum

	phistarpick = N/(sum*25.**3.)
	phisig = np.sqrt(N)/(sum*25.**3.)
	#if step == '000744': pdb.set_trace()
		#print a[max], minmag, maxmag, Ms, phistarpick, phisig
	maxL = np.max(logL[~np.isnan(logL)])
	max1 = logL[a < a[max]]
	alpha1 = a[a < a[max]]
	max2 = logL[a > a[max]]
	alpha2 = a[a > a[max]]
	d1 = alpha1[np.abs(max1[~np.isnan(max1)] - maxL/2.) == np.min(np.abs(max1[~np.isnan(max1)] - maxL/2.))]
	d2 = alpha2[np.abs(max2[~np.isnan(max2)] - maxL/2.) == np.min(np.abs(max2[~np.isnan(max2)] - maxL/2.))]
	try: sigma = (d2-d1)/(2*np.sqrt(2*np.log(2)))
	except ValueError: pdb.set_trace()
	alphafit = a[max][0]
	if plot:
		if subplot: ax = subplot
		else: f, ax = plt.subplots()
<<<<<<< HEAD
	
	#ffake, axfake = plt.subplots()

	#hist, bins = np.histogram(mags, bins = dbins[z])

	outadd = np.histogram(allmags, bins=20)
	#outadd = astroMLhist(allmags, bins='blocks', ax=axfake, log=True)
	out = np.histogram(mags, bins=outadd[1])

	hist = out[0]
	bins = out[1]
	histadd = outadd[0]
	binsadd = outadd[1]
	normalization = (bins[1:]-bins[:-1])*(np.float(params.paramfile['dKpcUnit'])/1000.*(0.6777/0.7))**3.
	x =  (bins[1:]+bins[:-1])/2.
	err = 1. + (hist + 0.75)**0.5
	if plot: 
		#ax.errorbar(x, (hist/normalization), 
        #    	yerr = err/normalization, 
        #       	linewidth=2, drawstyle='steps-mid', color='k', linestyle='--')
		normalization = (binsadd[1:]-binsadd[:-1])*(np.float(params.paramfile['dKpcUnit'])/1000.*(0.6777/0.7))**3.
		x =  (binsadd[1:]+binsadd[:-1])/2.
		err = 1. + (histadd + 0.75)**0.5
    	ax.errorbar(x, (histadd/normalization), 
    		yerr = err/normalization, linewidth=2, drawstyle='steps-mid', color='k')
	
	xx = np.arange(span[z][1], span[z][0], 0.1)

	
	#minmag  = minmagnitude[z]
	#maxmag = maxmagnitude[z] #np.min(mags[completeness <= 0.95])
	#maxmagadd = maxmagadds[step]
	
	phiminus = 2./5*(phistar[ind[z]]+plusphi[ind[z]])*np.log(10)*(10.**(2./5*((Mstar[ind[z]]-dMstar[ind[z]])-xx)))**((alpha[ind[z]]-dalpha[ind[z]])+1.)*np.exp(-10.**(2./5*((Mstar[ind[z]]-dMstar[ind[z]])-xx)))
	phiplus  = 2./5*(phistar[ind[z]]-minusphi[ind[z]])*np.log(10)*(10.**(2./5*((Mstar[ind[z]]+dMstar[ind[z]])-xx)))**((alpha[ind[z]]+dalpha[ind[z]])+1.)*np.exp(-10.**(2./5*((Mstar[ind[z]]+dMstar[ind[z]])-xx)))

	if plot:
		"""
		mcmcdata = np.load('mcmc_chains.npy')
		par = mcmcdata[mcmcz[z]]
		Mstarmcmc = par[0:1000000-1]
		alphamcmc = par[1000000:2000000-1]
		phimcmc   = par[2000000:3000000-1]
		pick = np.random.random_integers(0, high=1000000-1, size=500)
		#print np.min(Mstarmcmc[pick]), np.max(Mstarmcmc[pick]), np.min(phimcmc[pick]), np.max(phimcmc[pick]), np.min(alphamcmc[pick]), np.max(alphamcmc[pick])
		for j in range(len(pick)): ax.plot(xx, schecter(xx, Mstarmcmc[pick[j]], phimcmc[pick[j]], alphamcmc[pick[j]]), linewidth=0.5, color='grey')
		#ax.fill_between(xx, phiplus, phiminus, alpha=0.5, color='k',label='Bouwens et al 2014')
	#if i < 2:p = plt.Rectangle((-20,1), 0, 0, alpha=0.5, color='k', label='Bouwens et al 2014')
		p = plt.Rectangle((-20,1), 0, 0, alpha=0.5, color=color[z], label='z~'+str(z))
		ax.add_patch(p)
		"""
		if i in [3, 4]: ax.set_xlabel('M$_{UV}$',fontsize=20, labelpad=15)
=======
		xx = np.linspace(minmag, maxmag, 100)
		ax.plot(np.zeros(100) + maxmag, np.logspace(-5, 2, 100), 'k--', linewidth=2)
		ax.fill_between(xx,schecter(xx, Mstar[ind[z]]-dMstar[ind[z]], phistarpick+phisig, alphafit-sigma), schecter(xx, Mstar[ind[z]]+dMstar[ind[z]], phistarpick-phisig, alphafit+sigma),alpha=0.5, color=color[z])
		#ax.text(0.95, 0.9-0.15*m, r"$\alpha$ = "+'{:.3}'.format(alphafit[m]) + ' +- ' + '{:.1}'.format(3.*sigma[m]), ha='right', va='top', transform=ax.transAxes)
   	      	#bbox=dict(ec='w', fc='w'))
		#ax.text(0.95, 0.875-0.15*m, 'N: '+str(N), ha='right', va='top', transform=ax.transAxes) #, bbox=dict(ec='w',fc='w'))
 
		histallmags = np.histogram(allmags, bins=20)
		histmags = np.histogram(mags, bins=histallmags[1])
		for h in [histallmags]:#, histmags]:
			hist = h[0]
			bins = h[1]
			normalization = (bins[1:]-bins[:-1])*(
				np.float(params.paramfile['dKpcUnit'])/1000.*(0.6777/0.7))**3.
			x =  (bins[1:]+bins[:-1])/2.
			err = 1. + (hist + 0.75)**0.5
			#ax.errorbar(x, hist/normalization, yerr=err/normalization, 
			#	    linewidth=2, drawstyle='steps-mid', color='k')
		plotMcmc(ax, z)
		if i in [3, 4, 5]: ax.set_xlabel('M$_{UV}$',fontsize=20, labelpad=15)
>>>>>>> a3c1afffe558932da009e74cb1924925dc7d075e
		if i in [0,3]: ax.set_ylabel('$\phi$ [dex$^{-1}$ Mpc$^{-3}$]', fontsize=20)
		ax.grid(True)
		ax.set_yscale('log', nonposy='clip')
		ax.set_ylim(1e-5, 1)
		ax.set_xlim(-13.25, -23)
		ax.legend(loc='lower left')


#--------------------------------------------------------------------------------------------------------------------
	"""
	#phiminus = 2./5*(phistar[ind[z]]+plusphi[ind[z]])*np.log(10)*(10.**(2./5*((Mstar[ind[z]]-dMstar[ind[z]])-xx)))**((alpha[ind[z]]-dalpha[ind[z]])+1.)*np.exp(-10.**(2./5*((Mstar[ind[z]]-dMstar[ind[z]])-xx)))
	#phiplus  = 2./5*(phistar[ind[z]]-minusphi[ind[z]])*np.log(10)*(10.**(2./5*((Mstar[ind[z]]+dMstar[ind[z]])-xx)))**((alpha[ind[z]]+dalpha[ind[z]])+1.)*np.exp(-10.**(2./5*((Mstar[ind[z]]+dMstar[ind[z]])-xx)))

	N = np.sum((allmags >= minmag) & (allmags <= maxmagadd))
	Ms = Mstar[ind[z]]
	n = 100
	if i == 4: width = 5.
	else:width = 10.

	aadd = np.linspace(alpha[ind[z]]-dalpha[ind[z]]*width, alpha[ind[z]]+dalpha[ind[z]]*width, n)

	logL = np.zeros(len(aadd), dtype=np.float)
	for k in range(len(aadd)):
		logL[k] = schecterprob_logL(allmags[(allmags <= maxmagadd) & (allmags >= minmag)], Ms, 1.0, aadd[k])
	logL -= np.max(logL)
	logL = np.exp(logL) # - logL.max())
	logL /= logL.sum()
	maxadd = np.where(logL == np.max(logL))[0]


	sum = integrate.quad(schecter, minmag, maxmagadd, args=(Ms, 1.0, aadd[maxadd]))[0]
	phistarpickadd = N/(sum*25.**3.)
	phisigadd = np.sqrt(N)/(sum*25.**3.)
	#print aadd[max], minmag, maxmagadd, Ms, phistarpickadd, phisigadd
	maxL = np.max(logL)
	max1 = logL[aadd < aadd[maxadd]]
	alpha1 = aadd[aadd < aadd[maxadd]]
	max2 = logL[aadd > aadd[maxadd]]
	alpha2 = aadd[aadd > aadd[maxadd]]

	d1 = alpha1[np.abs(max1 - maxL/2.) == np.min(np.abs(max1 - maxL/2.))]
	d2 = alpha2[np.abs(max2 - maxL/2.) == np.min(np.abs(max2 - maxL/2.))]

	sigmaadd = (d2-d1)/(2*np.sqrt(2*np.log(2)))

	#ax1 = plt.subplot(121)
	if plot:
		#print 'Im plotting'
		xx = np.linspace(minmag, maxmagadd, 100)
		ax.fill_between(xx,schecter(xx, Mstar[ind[z]]-dMstar[ind[z]], phistarpickadd+phisigadd, aadd[maxadd]-3*sigmaadd),
			               schecter(xx, Mstar[ind[z]]+dMstar[ind[z]], phistarpickadd-phisigadd, aadd[maxadd]+3*sigmaadd),alpha=0.5, color=color[z])
		ax.text(0.95, 0.7, r"$\alpha$ = "+'{:.3}'.format(aadd[maxadd][0]) + ' +- ' + '{:.1}'.format(3.*sigmaadd[0]), ha='right', va='top',
   	      transform=ax.transAxes)
   	      #bbox=dict(ec='w', fc='w'))
   	 	ax.text(0.95, 0.6, 'N: '+str(N), ha='right', va='top', transform=ax.transAxes) #, bbox=dict(ec='w',fc='w'))
		ax.plot(np.zeros(100) + maxmagadd, np.logspace(-5, 2, 100), 'k', linewidth=2)
	#pdb.set_trace()
	"""
<<<<<<< HEAD
	return alphafit, sigma, phistarpick, phisig
=======
	return alphafit, sigma, phistarpick, phisig #, sum, N, minmag, maxmag
>>>>>>> a3c1afffe558932da009e74cb1924925dc7d075e
