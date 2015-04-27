import numpy as np 
def read(file):
	return np.genfromtxt(file, dtype=['float32', 'float32', 'float32', 'int32'], 
							   names=['mass', 'lum', 'mag', 'grp'], skip_header=1)