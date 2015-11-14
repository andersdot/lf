import lfplot_imf as lfplot
import glob
import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys 

IMF = sys.argv[1]
perComp = 50
#data = glob.glob('mags.resolved.*.npy')
data = glob.glob('*.' + IMF + '*.mag1')
data.sort()
data = data[::-1]
#data = data[1:]
print data

redshift = {'000547':10, '000744':8, '000891':7, '001093':6, '001380':5, '001818':4}
bouwens_alpha =  [ -1.64,  -1.78,  -1.91,  -2.06,  -1.86, -2.27]
bouwens_dalpha   =  np.array([0.04, 0.05, 0.09, 0.12, 0.27, 0.0])
bouwens_z = [3.8, 4.9, 5.9, 6.8, 7.9, 10.4]
color = {4:'purple', 5:'blue', 6:'cyan', 7:'green', 8:'orange', 10:'red'}
#color = {4:'purple', 5:'blue', 6:'green', 7:'orange', 8:'red', 10:'grey'}

fink_alpha = [-1.56, -1.67, -1.98, -2.05, -2.40, -2.27]
fink_dalpha = np.array([0.05, 0.06, 0.10, 0.20, 0.46, 0.0])
fink_z=np.array([4, 5, 6, 7, 8, 10])
fink_phi = np.array([13.9, 8.51, 2.02, 1.42, 0.77, 0.008])*1e-4
fink_dphip = np.array([1.92, 1.67, 0.97, 1.20, 2.41, 0.004])*1e-4
fink_dphim = np.array([1.92, 1.67, 0.97, 1.20, 0.76, 0.003])*1e-4
alpha  = np.zeros(len(data), dtype=float)
dalpha = np.zeros(len(data), dtype=float)
ad     = np.zeros(len(data))
dad    = np.zeros(len(data))
phi    = np.zeros(len(data))
dphi   = np.zeros(len(data))
phiadd = np.zeros(len(data))
dphiadd = np.zeros(len(data))
z = np.zeros(len(data), dtype=float)

p = {4:True, 5:True, 6:True, 7:True, 8:True, 10:True}

fig = plt.figure(figsize=(21, 10))

ax1 = plt.subplot2grid((2,3), (0,0))
ax3 = plt.subplot2grid((2,3), (0,2))
ax2 = plt.subplot2grid((2,3), (0,1))
ax4 = plt.subplot2grid((2,3), (1,0))
ax5 = plt.subplot2grid((2,3), (1,1))

ax6 = plt.subplot2grid((2,3), (1,2))
axs = [ax1, ax2, ax3, ax4, ax5, ax6]

figa, axa = plt.subplots(2)

for i in range(len(data)):
	subplot = axs[i]
	z[i] = redshift[data[i].split('.')[2]]
	(alpha[i], dalpha[i], phi[i], dphi[i]) = lfplot.plot(data[i], subplot=subplot, i=i, plot=p[z[i]], restore=True, IMF=IMF, perComp=perComp)


for k in range(len(alpha)):
	axa[0].errorbar(z[k]+0.2, alpha[k], yerr=dalpha[k],ls='None', elinewidth=4, color=color[z[k]])
	axa[0].scatter (z[k]+0.2, alpha[k], marker='o', s=60, color=color[z[k]])
	axa[1].errorbar(z[k]+0.2, phi[k], yerr=dphi[k], ls='None', elinewidth=4, color=color[z[k]])
	axa[1].scatter (z[k]+0.2, phi[k], marker='o', s=80, color=color[z[k]])
	print phi, dphi
axa[0].errorbar(fink_z-0.2, fink_alpha, yerr=fink_dalpha,ls='None', elinewidth=2, color='k')
axa[0].scatter(fink_z-0.2, fink_alpha, marker='o', s=40, linewidth=0, color='k')
axa[0].legend(loc='lower left')
#axa[0].yaxis.tick_right()
#axa[0].yaxis.set_label_position("right")
axa[0].set_ylabel(r"$\alpha$", fontsize=20)
#axa[0].set_xlabel('z', fontsize=20)
axa[0].set_xlim(3, 11)
#axa[0].set_ylim(-3.2,)

axa[1].errorbar(fink_z-0.2, fink_phi, yerr = [fink_dphip,fink_dphim], ls='None', label='Finkelstein+ 2014', elinewidth=2, color='k')
axa[1].scatter(fink_z-0.2, fink_phi, marker='o', s=40, linewidth=0, color='k')

"""
for i in range(len(z)):
	plt.errorbar(z[i], phi[i], yerr=dphi[i],ls='None', elinewidth=3, color=color[z[i]])
	plt.scatter(z[i], phi[i], marker='o', s=40, color=color[z[i]])
	plt.errorbar(z[i]+0.1, phiadd[i], yerr=dphiadd[i],ls='None', elinewidth=4, color=color[z[i]])
	plt.scatter(z[i]+0.1, phiadd[i], marker='o', s=80, color=color[z[i]])
"""

axa[1].set_yscale('log', nonposy='clip')
#axa[1].set_ylim(2e-6, 5e-3)
axa[1].set_xlim(3, 11)
plt.legend(loc='lower left')
axa[1].set_ylabel('$\phi$* [dex$^{-1}$ Mpc$^{-3}$]', fontsize=20)
axa[1].set_xlabel('z', fontsize=20)
axa[1].grid(True)
axa[0].grid(True)
figa.savefig('phi.' + IMF + '.png')
fig.savefig('lf.' + IMF + '.png')

np.savez('lfParams.' + IMF, alpha=alpha, dalpha=dalpha, phi=phi, dphi=dphi)
