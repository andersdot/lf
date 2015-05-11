import pynbody
import matplotlib.pyplot as p
import numpy as np
import customCmap as cc
import pdb

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
position = [0, 0.1, 0.2, 0.6, 0.7, 1]
cmap = cc.make_cmap(colors, bit=True, position=position)

filename = '/scratch/sciteam/lmanders/cosmo8.33PLK.256g2bwK1C52/cosmo8.33PLK.256g2bwK1C52.002763'
qty = 'rho'
threads = 8
resolution = 1000
n = resolution
nz = 200
width = 6000
vmin = 1
vmax = 4.5

prefix = filename.split('/')[-1].split('.')[0]
print prefix

sim = pynbody.load(filename)
sim.physical_units()
p.gray()

#maybe
#h = sim.halos()
#pynbody.angmom.sideon(h[1])


sim.gas['smooth'] = (sim.gas['mass']/sim.gas['rho'])**(1,3)

#my_slice = sim.gas[pynbody.filt.BandPass('z',-500,500)]
#pynbody.plot.sph.image(my_slice, width=width, qty=qty, av_z=True, cmap=cmap, threaded=threads, resolution=resolution)
#p.imshow(np.log10(grid[:,:,50]), cmap=cmap, vmin=-1, vmax=3)
#p.show()
#pdb.set_trace()


try: 
    grid = np.load('grid.' + prefix + '.' + str(resolution) + '.' + qty +'.npy')
except IOError:
    print 'building grid'
    grid = pynbody.sph.to_3d_grid(sim.gas, qty = qty, nx=n, ny=n, nz=nz, threaded=threads)
    np.save('grid.' + prefix + '.' + str(resolution) + '.' + qty, grid)

for i in range(nz):
    p.imsave('moviepngs/' + prefix + '.' + '{0:03d}'.format(i) + '.png', 
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
