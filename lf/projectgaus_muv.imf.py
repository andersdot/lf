import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
from astroML.datasets import generate_mu_z
from astroML.cosmology import Cosmology
from astroML.plotting.mcmc import convert_to_stdev
from astroML.decorators import pickle_results
from astroML.plotting import hist
import sys
import pdb
from mpl_toolkits.mplot3d import axes3d
from astroML.density_estimation import KDE, KNeighborsDensity
from scipy import stats
from matplotlib.colors import LogNorm
import matplotlib 
import nbdpt.readstat as rds

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

def getPlotArrays():
	NN = 1000
	xmin, xmax = (9, 12) #(np.min(logtotmass), np.max(logtotmass))
	ymin, ymax = (-24, -8) #(np.min(mags), np.max(mags))

	massPlot = np.linspace(xmin, xmax, NN) #this is what logmass is being sampled in below 
	magPlot  = np.linspace(ymin, ymax, NN)
	return massPlot, magPlot

def completeness(x, alpha, beta): 
	return 1./(1. + np.exp(alpha*x + beta))

# log likelihood in terms of the parameters
def P(x, a, b):
	return a*x + b

# chi squared
def compute_logL(x, y, a, b, sig):
	comp_pred = a*x + b
	return - np.sum(0.5 * ((y - comp_pred)/sig) ** 2.)

# Compute (and save to file) the log-likelihood
def comp_muv_mtot(x, y, sig, amin,amax,bmin,bmax,Nbins=20):
	a = np.linspace(amin, amax, Nbins)
	b = np.linspace(bmin, bmax, Nbins)
	logL = np.empty((Nbins, Nbins))
	for i in range(len(a)):
		for j in range(len(b)):
			logL[i, j] = compute_logL(x, y, a[i], b[j], sig)
	return a, b, logL


def getdata(statfilename):
	redshift = {'000744':8, '000891':7, '001093':6, '001380':5, '001818':4}
	z = redshift[statfilename.split('.')[2]]
	stat = np.load(statfilename)
	minNsfh = 1e4
	# sort stat file by mass
	stat = stat[np.argsort(stat['mvir'])[::-1]]
	# find the last halo that is nonzero
	last = np.where(stat['mvir'] == 0.)[0][0]
	# all good halos have nonzero mass 
	good = stat['mvir'] > 0
	# good = (stat['npart'] > 1024) & (stat['nstar'] > 0)
	# all wanted halos with reliable sfh
	wanted = (stat['npart'] > minNsfh) & (stat['nstar'] > 0)
	# accumulated number of halos from most to least massive
	allmass = np.array(np.cumsum(good[0:last]), dtype=np.float)
	# accumulated number of halos with reliable sfh from most to least massive
	wantmass = np.cumsum(wanted[0:last])
	#return the sorted masses and the ratio of accumulated wanted/all == completeness 
	return np.log10(stat['mvir'][0:last]), wantmass/allmass #, sigcomp

def calcKde(logtotmass, mags, z, plot=True):
	X = np.array(zip(logtotmass, mags))
	NN = 1000
	Nx = NN
	Ny = NN

	#xmin, xmax = (9, 12) #(np.min(logtotmass), np.max(logtotmass))
	#ymin, ymax = (-24, -8) #(np.min(mags), np.max(mags))
	massPlot, magPlot = getPlotArrays()
	xmin = np.min(massPlot)
	xmax = np.max(massPlot)
	ymin = np.min(magPlot)
	ymax = np.max(magPlot)
	#massSample   = np.linspace(xmin, xmax, NN) #this is what logmass is being sampled in below 
	#magplot = np.linspace(ymin, ymax, NN)
	#comp    = completeness(mtots, alpha, beta)

	Xgrid = np.vstack(map(np.ravel, np.meshgrid(np.linspace(xmin, xmax, Nx), 
												np.linspace(ymin, ymax, Ny)))).T

	metric = 'tophat'
	colors={0.075:'blue', 0.085:'green', 0.095:'yellow', 0.105:'orange', 0.115:'red', 0.125:'black'}
	hh = [0.075, 0.085, 0.095, 0.105, 0.115, 0.125]

	h = hh[2]

	kde = KDE(metric=metric, h=h)
	dens_KDE = kde.fit(X).eval(Xgrid).reshape((Nx, Ny))

	N = len(dens_KDE[:,0])
	mean = np.zeros(N)
	width = np.zeros(N)
	chi = np.zeros(N)
	maxdata = np.zeros(N)
	for i in range(N): 
		data = dens_KDE[:,i]
		if np.sum(data) > 0:
			median = np.median(data)
			mean[i] = np.sum(magPlot*data)/np.sum(data)
			width[i] = np.sqrt(np.abs(np.sum((magPlot-mean[i])**2.*data)/np.sum(data)))
			maxdata[i] = np.max(data)
			fit = lambda t: 1./(10.**t)*np.exp(-(t-mean[i])**2./(2.*width[i]**2.))
			gaus = fit(magPlot)
			gaus = gaus/np.max(gaus)*maxdata[i]
			nonzero = data > 0.
			chi[i] = np.sqrt(np.sum((gaus[nonzero] - data[nonzero])**2./data[nonzero]))/np.sum(nonzero)
	if plot:
		fig = plt.figure(figsize=(13.333, 6.666))
		ax1 = plt.subplot(121)
		ax2 = plt.subplot(122)
		fs = 20

		ax1.imshow(dens_KDE, origin='lower', norm=LogNorm(),interpolation=None, aspect='auto', extent=(xmin, xmax, ymin, ymax), cmap=plt.cm.binary)
		ax1.scatter(X[:, 0], X[:, 1], s=1, lw=0, c='k')
		ax1.text(0.35, 0.95, "Vulcan z ~ " + str(z), ha='right', va='top', transform=ax1.transAxes, bbox=dict(boxstyle='round', ec='k', fc='w'))
		ax1.set_xlim(xmin, xmax)
		ax1.set_ylim(ymax, ymin)
		ax1.set_ylabel('M$_{UV}$', fontsize=fs)
		ax1.set_xlabel('log M$_h$', fontsize=fs, labelpad=10)		

		#ax2.text(0.95, 0.1, 'KDE '+metric +' h=' + str(h), ha='right', va='top', transform=ax2.transAxes, bbox=dict(boxstyle='round', ec='k', fc='w'))
		ax2.imshow(dens_KDE, origin='lower', norm=LogNorm(),interpolation=None, aspect='auto', extent=(xmin, xmax, ymin, ymax), cmap=plt.cm.binary)
		ax2.scatter(X[:, 0], X[:, 1], s=1, lw=0, c='k')
		ax2.plot(massPlot, mean, linewidth=2, c='g', label='gaussian fit')
		ax2.fill_between(massPlot, (mean + width), (mean - width), alpha=0.5, color='g')
		ax2.set_xlim(xmin, xmax)
		ax2.set_ylim(ymax, ymin)
		ax2.set_xlabel('log M$_h$', fontsize=fs, labelpad=0)
		ax2.set_ylabel('M$_{UV}$', fontsize=fs, labelpad=0)
		ax2.legend(loc='lower right')
		ax2.grid(True, which='both')
		plt.tight_layout()
		plt.savefig('kde.'+ step + '.' + IMF + '.png')
		plt.clf()
	return mean, width, maxdata


def findSlopeYint(step, IMF, mean, width, maxdata, wanted, wanted_sig, z):
	try: 
		ab = np.load('slope.' + step + '.' + IMF + '.npy')
		bb = np.load('yint.' + step + '.' + IMF + '.npy')
	except IOError:

		labels = ['mean', 'mean+sigma', 'mean-sigma']
		colors = ['red',  'blue', 'blue']
		massPlot, magPlot = getPlotArrays()
		poses = [mean, mean + width, mean - width]
		wantednow = [wanted, wanted_sig, wanted_sig]
		#arange = [-5, -4]
		#brange = [23, 45]
		#a_guess = [-4.5, -1.]
		#b_guess = [26, 7]
		arange = [-5, -5, -5]
		brange = [23, 23, 23]
		a_guess = [-4.5, -4.5, -4.5]
		b_guess = [26, 26, 26]
		#------------------------------------------------------------
		# Plot the results
		
		Nbins = 100
		ab = np.zeros(len(poses), dtype=np.float)
		bb = np.zeros(len(poses), dtype=np.float)


		for i, pos in enumerate(poses):	
			fig = plt.figure(figsize=(20, 5))
			fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25,
            	       bottom=0.15, top=0.9)

			ax1 = fig.add_subplot(121)
			ax2 = fig.add_subplot(122)
			a_best = a_guess[i]
			b_best = b_guess[i]

			amin = 0.5*a_best
			amax = 1.5*a_best
			bmin = 0.5*b_best
			bmax = 1.5*b_best
			if amin > amax: amin = 1.5*a_best; amax = 0.5*a_best
			if bmin > bmax: bmin = 1.5*b_best; bmax = 0.5*b_best
			dela = np.abs(float(amax-amin))/2.
			delb = np.abs(float(bmax-bmin))/2.
			sig = np.sqrt(maxdata[wantednow[i]]) #1. + (masses[pos] + 0.75)**0.5
			#sig[masses[pos] < 5] = np.inf	

			for j in range(100):

				a, b, res = comp_muv_mtot(massPlot[wantednow[i]],pos[wantednow[i]],sig,amin,amax,bmin,bmax,Nbins=Nbins)
				#pdb.set_trace()
				res -= np.max(res)
				whr = np.where(res == np.max(res))
				a_best = a[whr[0][0]]
				b_best = b[whr[1][0]]

				#pdb.set_trace()
				dela = dela/1.25
				delb = delb/1.25
				amin = a_best - dela
				amax = a_best + dela
				bmin = b_best - delb
				bmax = b_best + delb
				if amin > amax: amin = a_best + dela; amax = a_best - dela
				if bmin > bmax: bmin = b_best + delb; bmax = b_best - delb
				print labels[i], 'a b = ', a_best, b_best
		
				if j%5 ==0:
					ax1.plot(massPlot, P(massPlot, a_best, b_best), color=colors[i])
					#ax1.plot(mtots, pos)
					ax1.plot(massPlot[wantednow[i]], pos[wantednow[i]])
					ax2.contour(a, b, convert_to_stdev(res.T),
   	     	  			levels=(0.683, 0.955, 0.997),
       	 	  			colors=colors[i])
		
		
			ax1.set_title(labels[i] + ' z~' + str(z))
			ax1.set_xlabel('$M_{h}$')
			ax1.set_ylabel('$M_{UV}$')
			#ax1.set_ylim(ymax, ymin)
			#ax1.set_xlim(np.min(logtotmass), np.max(logtotmass))
			ax2.legend()
			ax2.set_xlabel(r'$a$')
			ax2.set_ylabel(r'$b$')	

			plt.savefig(labels[i] + '.' + step + '.' + IMF + '.png')
			plt.clf()
			ab[i] = a_best
			bb[i] = b_best

		np.save('slope.' + step + '.' + IMF, ab)
		np.save('yint.' + step + '.' + IMF, bb)

	return ab, bb

def getWantMag(step, IMF):

	N = 1000
	completenessThreshold = 0.999

	xplotmin = 7
	xplotmax = 11.5
	yplotmax = -2.5
	yplotmin = -25


	#maxmass = {'001818':10.9945994599, '001380':10.8615861586, '001093':10.6845684568, '000891':10.4845484548, '000744':10.2685268527}
	maxmass = {'001818':10.5, '001380':10.5, '001093':10.5, '000891':10.5, '000744':10.}
	maxmass_sig = {'001818':10., '001380':9.8, '001093':9.8, '000891':9.8, '00744':9.8}
	massWidthSigFit = {'001818':0.8, '001380':0.8, '001093':0.8, '000891':0.5, '000744':0.3}
	redshift = {'000744':8, '000891':7, '001093':6, '001380':5, '001818':4}
	z = redshift[step]

	massPlot, magPlot = getPlotArrays()

	data = np.genfromtxt('cosmo25p.768sg1bwK1C52.'+step+'.tipsy.'+IMF+'.mag1', dtype=['float32', 'float32', 'float32', 'int32'], names=['mass', 'lum', 'mag', 'grp'], skip_header=1)
	grps = data['grp']
	mags = data['mag']
	logmass = np.log10(data['mass'])

	statfilename = 'cosmo25p.768sg1bwK1C52.'+step+'.rockstar.stat'
	try:
		stat = np.load(statfilename + '.npy')
	except IOError:
		stat = rds.readstat(statfilename)
		np.save(statfilename, stat)

	wantMag     = (~np.in1d(stat['grp'], grps) & (stat['npart'] > 64))
	massWantMag = np.log10(stat[wantMag]['mvir'])

	massCompleteness, completeness = getdata('cosmo25p.768sg1bwK1C52.'+step+'.rockstar.stat.npy')
	minmasscomp   = np.min(massCompleteness[completeness > completenessThreshold])
	compLogmassLA = massPlot > minmasscomp
	print '~ ~ ~ ~ ~ Minimum mass complete: ', minmasscomp 
	projectedMass = np.linspace(xplotmin, minmasscomp, 1000)



	mean, width, maxdata = calcKde(logmass, mags, z, plot=True)
	nonzero = mean != 0.0

	massPlot = massPlot[nonzero]
	mean     = mean[nonzero]
	width    = width[nonzero]
	compLogmassLA = compLogmassLA[nonzero]

	wantedFit    = (massPlot > minmasscomp) & (massPlot < maxmass[step])
	wantedSigFit = (massPlot > minmasscomp) & (massPlot < minmasscomp+massWidthSigFit[step])
	slope, yint  = findSlopeYint(step, IMF, mean, width, maxdata, wantedFit, wantedSigFit, z)


	highsigProj = P(projectedMass, slope[1], yint[1])
	lowsigProj  = P(projectedMass, slope[2], yint[2])

	Mass50perComp = massPlot[mean == np.max(mean[compLogmassLA])]
	mag50perComp  = P(Mass50perComp, slope[0], yint[0])
	mag98perComp  = P(Mass50perComp, slope[2], yint[2])
	avgmuv        = mag50perComp

	f = open('massMagsComplete.' + step + '.' + IMF + '.txt', 'w')
	f.write('Completeness [%]   logMass [Msol]    Mag \n')
	f.write('50   ' + str(Mass50perComp[0]) + '   ' + str(mag50perComp[0]) + '\n')
	f.write('98   ' + str(Mass50perComp[0]) + '   ' + str(mag98perComp[0]) + '\n')
	f.close()


	avgMag    =  slope[0]*massWantMag + yint[0]
	sigsplus  =  slope[1]*massWantMag + yint[1] - avgMag
	sigsminus = -slope[2]*massWantMag - yint[2] + avgMag
	sigMag = (sigsplus + sigsminus)/2.
	mags_fill = np.random.normal(loc=avgMag, scale=sigMag, size=len(avgMag))
	np.save('magsfill.'+step + '.' + IMF, mags_fill)

	xlinplot = np.linspace(xplotmin, xplotmax, 100)
	zeroplot = np.zeros(100)

	plt.figure(figsize=(13.333, 13.333))
	plt.plot(massPlot[compLogmassLA], mean[compLogmassLA], linewidth=2, c='g', label='gaussian fit')
	for i in range(len(slope)): plt.plot(projectedMass, P(projectedMass, slope[i], yint[i]), color='b')

	plt.fill_between(massPlot[compLogmassLA], (mean + width)[compLogmassLA], (mean - width)[compLogmassLA], alpha=0.5, color='g')
	plt.fill_between(projectedMass, P(projectedMass, slope[1], yint[1]), 
	                        	P(projectedMass, slope[2], yint[2]), alpha=0.5, color='b')

	plt.scatter(logmass, mags, s=1, lw=0, c='k')
	plt.scatter(massWantMag, mags_fill, s=1, lw=0, c='r')

	plt.plot(xlinplot, zeroplot + mag98perComp, '-', color='k', linewidth=2, label='current limit') # magplot[mtots == np.min(mtots[cc])], '--k')
	plt.plot(xlinplot, zeroplot + mag50perComp, '--', color='k', linewidth=2, label = '50% limit')
	plt.plot(xlinplot, zeroplot + P(Mass50perComp-1., slope[0], yint[0]), ':', color='k', linewidth=2, label = 'Enterprise limit')

	plt.plot(zeroplot + np.log10(np.min(stat['mvir'][stat['npart'] > 256])), np.linspace(yplotmin, yplotmax, 100))
	plt.plot(zeroplot + np.log10(np.min(stat['mvir'][stat['npart'] > 64])), np.linspace(yplotmin, yplotmax, 100))

	plt.xlabel('log M$_{h}$', fontsize=20, labelpad=10)
	plt.ylabel('M$_{UV,AB}$', fontsize=20)
	plt.ylim(yplotmax, yplotmin)
	plt.xlim(xplotmin, xplotmax)
	plt.legend(loc='upper left')
	plt.tight_layout()
	if (step == '000744') & (IMF == 'kroupa'): pdb.set_trace()
	plt.savefig('project.' + step + '.' + IMF + '.png')
	plt.clf()

if __name__ == '__main__':
	step = sys.argv[1]
	IMF = sys.argv[2]
	getWantMag(step, IMF)

