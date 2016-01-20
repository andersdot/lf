import lfplot_imf as lfplot
import glob
import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys

IMF = sys.argv[1] #'kroupa'

#data = glob.glob('mags.resolved.*.npy')
data = glob.glob('*.' + IMF + '.mag1')
data.sort()
data = data[::-1]
print data

redshift = {'000744':8, '000891':7, '001093':6, '001380':5, '001818':4}
bouwens_alpha =  [ -1.64,  -1.78,  -1.91,  -2.06,  -1.86]
bouwens_dalpha   =  np.array([0.04, 0.05, 0.09, 0.12, 0.27])
bouwens_z = [3.8, 4.9, 5.9, 6.8, 7.9]
color = {4:'purple', 5:'blue', 6:'green', 7:'orange', 8:'red'}

fink_alpha = [-1.56, -1.67, -1.98, -2.05, -2.40]
fink_dalpha = np.array([0.05, 0.06, 0.10, 0.20, 0.46])
fink_z=np.array([4, 5, 6, 7, 8])
fink_phi = np.array([13.9, 8.51, 2.02, 1.42, 0.77])*1e-4
fink_dphip = np.array([1.92, 1.67, 0.97, 1.20, 2.41])*1e-4
fink_dphim = np.array([1.92, 1.67, 0.97, 1.20, 0.76])*1e-4
alpha  = np.zeros(len(data), dtype=float)
dalpha = np.zeros(len(data), dtype=float)
ad     = np.zeros(len(data))
dad    = np.zeros(len(data))
phi    = np.zeros(len(data))
dphi   = np.zeros(len(data))
phiadd = np.zeros(len(data))
dphiadd = np.zeros(len(data))
z = np.zeros(len(data), dtype=float)

p = {4:True, 5:True, 6:True, 7:True, 8:True}

ax1 = plt.subplot2grid((2,3), (0,0))
ax3 = plt.subplot2grid((2,3), (0,2))
ax2 = plt.subplot2grid((2,3), (0,1))
ax4 = plt.subplot2grid((2,3), (1,0))
ax5 = plt.subplot2grid((2,3), (1,1))

ax6 = plt.subplot2grid((2,3), (1,2))
axs = [ax1, ax2, ax3, ax4, ax5]

fig, ax = plt.subplots()

for i in range(len(data)):
	subplot = axs[i]
	z[i] = redshift[data[i].split('.')[2]]
	(alpha, dalpha, phi, dphi) = lfplot.plot(data[i], subplot=subplot, i=i, plot=p[z[i]], restore=True, IMF=IMF)
	for k in range(len(alpha)):
		ax6.errorbar(z[i]+0.2*k, alpha[k], yerr=dalpha[k],ls='None', elinewidth=4, color=color[z[i]])
		ax6.scatter (z[i]+0.2*k, alpha[k], marker='o', s=60, color=color[z[i]])
		ax.errorbar(z[i]+0.2*k, phi[k], yerr=dphi[k], ls='None', elinewidth=4, color=color[z[i]])
		ax.scatter (z[i]+0.2*k, phi[k], marker='o', s=80, color=color[z[i]])
		print phi, dphi
ax6.errorbar(fink_z-0.2, fink_alpha, yerr=fink_dalpha,ls='None', label='Finkelstein+ 2014', elinewidth=2, color='k')
ax6.scatter(fink_z-0.2, fink_alpha, marker='o', s=40, linewidth=0, color='k')
ax6.legend(loc='lower left')
ax6.set_ylabel(r"$\alpha$", fontsize=30, labelpad=0)
ax6.set_xlabel('z', fontsize=20)
ax6.set_xlim(3, 9)
ax6.set_ylim(-2.5,)

ax.errorbar(fink_z-0.2, fink_phi, yerr = [fink_dphip,fink_dphim], ls='None', label='Finkelstein+ 2014', elinewidth=2, color='k')
ax.scatter(fink_z-0.2, fink_phi, marker='o', s=40, linewidth=0, color='k')

"""
for i in range(len(z)):
	plt.errorbar(z[i], phi[i], yerr=dphi[i],ls='None', elinewidth=3, color=color[z[i]])
	plt.scatter(z[i], phi[i], marker='o', s=40, color=color[z[i]])
	plt.errorbar(z[i]+0.1, phiadd[i], yerr=dphiadd[i],ls='None', elinewidth=4, color=color[z[i]])
	plt.scatter(z[i]+0.1, phiadd[i], marker='o', s=80, color=color[z[i]])
"""

ax.set_yscale('log')
ax.set_ylim(2e-6, 5e-3)
plt.legend(loc='lower left')
ax.set_ylabel('$\phi$* [dex$^{-1}$ Mpc$^{-3}$]', fontsize=20)
ax.set_xlabel('z', fontsize=20)
plt.show()
pdb.set_trace()
