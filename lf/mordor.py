f=pynbody.load(...)
f.physical_units()
p.gray()

#maybe
h = f.halos()
pynbody.angmom.sideon(h[1])


f.gas['smooth'] = (f.gas['mass']/f.gas['rho'])**(1,3)

my_slice = f.gas[pynbody.filt.BandPass('z',-1000,1000)]

pynbody.plot.sph.image(my_slice, width=1000, qty='temp', av_z=True)

im = pynbody.plot.sph.image(my_slice, resolution=4000, … all your other parameters here …)

p.imsave(im, 'gas.png')

f.dm['rho'] = f.dm['den']
f.dm['smooth'] = (f.dm['mass']/f.dm['rho'])**(1,3)

my_slice = f.dm[pynbody.filt.BandPass('z',-1000,1000)]

pynbody.plot.sph.image(my_slice, width=1000, qty='temp', av_z=True)
