import numpy as np 

def calcComp(true, fake, comp):
	all = np.concatenate((true, fake))

	all = all[np.argsort(all)]
	true = true[np.argsort(true)]
	fake = fake[np.argsort(fake)]

	truesum = np.cumsum(np.in1d(all, true))
	fakesum = np.cumsum(np.in1d(all, fake))
	allsum = np.cumsum(np.ones(len(all)))


	whereComp = np.zeros(len(comp))
	for i in range(len(comp)): whereComp[i] = all[np.where(truesum/allsum < comp[i]/100.)[0][0]]

	return all, truesum/allsum, fakesum/allsum, whereComp