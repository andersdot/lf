import pynbody
import matplotlib.pyplot as plt
import numpy as np
import customCmap as cc
import pdb
import os

threads = 2

#color palette 1
color1 = (5, 16, 20)
color2 = (20, 33, 35)
color3 = (69, 88, 89)
color4 = (115, 57, 57) 
color5 = (219, 134, 145)
color6 = (253, 233, 205)
cp1 = [color1, color2, color3, color4, color5, color6]

#color palette 2
color1 = (9, 10, 15)
color2 = (6, 19, 33)
color3 = (9, 58, 98)
color4 = (39, 106, 155)
color5 = (102, 150, 188)
color6 = (216, 174, 125)
cp2 = [color1, color2, color3, color4, color5, color6]

#color palette 3
color1 = (35, 2, 3)
color2 = (60, 11, 11)
color3 = (31, 65, 111)
color4 = (69, 139, 224)
color5 = (103, 146, 189)
color6 = (255, 218, 181)
cp3 = [color1, color2, color3, color4, color5, color6]

#color palette 4
color1 = (0, 0, 12)
color2 = (12, 9, 29)
color3 = (36, 18, 42)
color4 = (78, 36, 45)
color5 = (252, 187, 96)
color6 = (253, 238, 152)
cp4 = [color1, color2, color3, color4, color5, color6]

#color palette 5
color1 = (0, 20, 29)
color2 = (1, 31, 36)
color3 = (17, 70, 82)
color4 = (29, 103, 107)
color5 = (229, 156, 145)
color6 = (252, 254, 224)
cp5 = [color1, color2, color3, color4, color5, color6]



colors = cp5

"""
virden = 25486.1655
logvirden = np.log10(virden/50.)
logvirden01 = np.log10(virden/200.)
logvirden001 = np.log10(virden/1000.)
vmin = 1
vmax = 5
posvir = (logvirden-vmin)/(vmax - vmin)
posvir01 = (logvirden01-vmin)/(vmax - vmin)
posvir001 = (logvirden001-vmin)/(vmax - vmin)
print posvir, posvir01, posvir001
position = [0, 0.05, posvir001, posvir01, posvir, 1]
"""

virden = 25486.1655
logvirden1 = np.log10(virden/20.)
logvirden2 = np.log10(virden/200.)
logvirden3 = np.log10(virden/1000.)
logvirden4 = np.log10(virden/2000.)
vmin = 0.5
vmax = 5
posvir1 = (logvirden1-vmin)/(vmax - vmin)
posvir2 = (logvirden2-vmin)/(vmax - vmin)
posvir3 = (logvirden3-vmin)/(vmax - vmin)
print posvir1, posvir2, posvir3
#position = [0, 0.1, 0.2, 0.5, 0.7, 1]
position = [0, 0.05, posvir3, posvir2, posvir1, 1]
cmap = cc.make_cmap(colors, bit=True, position=position)

dir = '/scratch/sciteam/lmanders/cosmo8.33PLK.256g2bwK1C52/'
filename = dir + 'cosmo8.33PLK.256g2bwK1C52.002763'
qty = 'rho'

resolution = 4000
n = resolution
nz = 200
width = 6000

prefix = filename.split('/')[-1].split('.')[0]
print prefix

#p.gray()
#maybe
#h = sim.halos()
#pynbody.angmom.sideon(h[1])
#my_slice = sim.gas[pynbody.filt.BandPass('z',-500,500)]
#pynbody.plot.sph.image(my_slice, width=width, qty=qty, av_z=True, cmap=cmap, threaded=threads, resolution=resolution)
#p.imshow(np.log10(grid[:,:,50]), cmap=cmap, vmin=-1, vmax=3)
#p.show()
#pdb.set_trace()


try: 
    print 'trying to load grid'
    grid = np.load(dir + 'grid.' + prefix + '.' + str(resolution) + '.' + qty +'.new.npy')
except IOError:
    print 'building grid'
    sim = pynbody.load(filename)
    sim.physical_units()
    sim.gas['smooth'] = (sim.gas['mass']/sim.gas['rho'])**(1,3)
    grid = pynbody.sph.to_3d_grid(sim.gas, qty = qty, nx=n, ny=n, nz=nz, threaded=threads)
    np.save(dir + 'grid.' + prefix + '.' + str(resolution) + '.' + qty + '.new', grid)

grid[np.where(grid == 0)] = abs(grid[np.where(abs(grid != 0))]).min()
plt.imshow(np.log10(grid[:,:,193]), cmap=cmap, vmin=vmin, vmax=vmax)
plt.show()
pdb.set_trace()
print 'creating images'
if not os.path.exists(dir + 'moviepngs'):
    os.makedirs(dir + 'moviepngs')

for i in range(nz):
    plt.imsave(dir + 'moviepngs/' + prefix + '.' + '{0:03d}'.format(i) + '.png', 
               np.log10(grid[:,:,i]), vmin=vmin, vmax=vmax, cmap=cmap)

"""
pynbody.plot.sph.image(dmy_slice, width=1000, qty='temp', av_z=True)

im = pynbody.plot.sph.image(my_slice, resolution=500)

p.imsave(im, 'gas.png')

#f.dm['rho'] = f.dm['den']
#f.dm['smooth'] = (f.dm['mass']/f.dm['rho'])**(1,3)

#my_slice = sim.dm[pynbody.filt.BandPass('z',-1000,1000)]

#pynbody.plot.sph.image(my_slice, width=1000, qty='temp', av_z=True)
"""
