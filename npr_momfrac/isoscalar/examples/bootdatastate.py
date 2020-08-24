#!/opt/software/anaconda2/bin/python2.7

import os
import h5py
import numpy as np

statename = "prot"
intops = 2

infilename = "/data/d04/pshanahan_nucleardisco/3pt_"+statename+".h5"
infile = h5py.File(infilename,'r')

Nt = 48

outfilename = "/data/d04/pshanahan_nucleardisco/analysis/nuclearDiscoBoot_momfrac.h5"
outfile = h5py.File(outfilename,'w')

twoptbootfw = outfile.create_dataset("twoptfw",(200,Nt,intops),dtype=np.float64)
twoptbootbw = outfile.create_dataset("twoptbw",(200,Nt,intops),dtype=np.float64)

#O33bootfw = outfile.create_dataset("O33fw",(200,Nt,Nt,intops),dtype=np.float64)
#O33bootbw = outfile.create_dataset("O33bw",(200,Nt,Nt,intops),dtype=np.float64)
#O33bootvac = outfile.create_dataset("O33vac",(200,Nt),dtype=np.float64)

#O43bootfw = outfile.create_dataset("O43fw",(200,Nt,Nt,intops),dtype=np.float64)
#O43bootbw = outfile.create_dataset("O43bw",(200,Nt,Nt,intops),dtype=np.float64)
#O43bootvac = outfile.create_dataset("O43vac",(200,Nt),dtype=np.float64)

O44bootfw = outfile.create_dataset("O44fw",(200,Nt,Nt,intops),dtype=np.float64)
O44bootbw = outfile.create_dataset("O44bw",(200,Nt,Nt,intops),dtype=np.float64)
O44bootvac = outfile.create_dataset("O44vac",(200,Nt),dtype=np.float64)

#O33bootsubbed = outfile.create_dataset("O33",(200,Nt,Nt,intops),dtype=np.float64)
#O44bootsubbed = outfile.create_dataset("O44",(200,Nt,Nt,intops),dtype=np.float64)
#O34bootsubbed = outfile.create_dataset("O34",(200,Nt,Nt,intops),dtype=np.float64)
#O43bootsubbed = outfile.create_dataset("O43",(200,Nt,Nt,intops),dtype=np.float64)

def bootstrap_resample(X, n=None, nboots=200):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(X)

    np.random.seed(5)

    if len(X.shape) == 2:
        ALL_resamples=np.zeros((nboots,X.shape[1]),dtype=complex)
    elif len(X.shape) == 4:
        ALL_resamples=np.zeros((nboots,X.shape[1],X.shape[2],X.shape[3]),dtype=complex)
    elif len(X.shape) == 5:
        ALL_resamples=np.zeros((nboots,X.shape[1],X.shape[2],X.shape[3],X.shape[4]),dtype=complex)
    else:
        ALL_resamples=np.zeros((nboots,X.shape[1],X.shape[2]),dtype=complex)

    for ii in xrange(nboots):
        print ii
        resample_i = np.sort(np.floor(np.random.rand(n)*len(X)).astype(int))
        ALL_resamples[ii] = np.mean(X[resample_i],axis=0)

    return ALL_resamples


cfgs = list(infile['loop'].keys())

twoptdatfw = np.zeros((len(cfgs),Nt,intops))
twoptdatbw = np.zeros((len(cfgs),Nt,intops))
twoptdatfw = twoptdatfw + [infile[statename+"/fwd/"+conf+"/2pt_averaged-re"] for conf in cfgs]
twoptdatbw = twoptdatbw + [infile[statename+"/bwd/"+conf+"/2pt_averaged-re"] for conf in cfgs]

twoptbootfw[:] = bootstrap_resample(twoptdatfw).real
twoptbootbw[:] = bootstrap_resample(twoptdatbw).real

#O33datfw = np.zeros((len(cfgs),Nt,Nt,intops))
#O33datbw = np.zeros((len(cfgs),Nt,Nt,intops))
O44datfw = np.zeros((len(cfgs),Nt,Nt,intops))
O44datbw = np.zeros((len(cfgs),Nt,Nt,intops))
#O34datfw = np.zeros((len(cfgs),Nt,Nt,intops))
#O34datbw = np.zeros((len(cfgs),Nt,Nt,intops))
#O43datfw = np.zeros((len(cfgs),Nt,Nt,intops))
#O43datbw = np.zeros((len(cfgs),Nt,Nt,intops))

#O33datfw = O33datfw + [2*np.mean([infile[statename+'/fwd/'+conf+'/'+t+'/p000-d3-re/g_4'].value.real for t in infile[statename+'/fwd/'+conf].keys()[2:]], axis=0) for conf in cfgs] - [2*np.mean([infile[statename+'/fwd/'+conf+'/'+t+'/p000-d-3-re/g_4'].value.real for t in infile[statename+'/fwd/'+conf].keys()[2:]], axis=0) for conf in cfgs]
#O33datbw = O33datbw + [2*np.mean([infile[statename+'/bwd/'+conf+'/'+t+'/p000-d3-re/g_4'].value.real for t in infile[statename+'/bwd/'+conf].keys()[2:]], axis=0) for conf in cfgs] - [2*np.mean([infile[statename+'/bwd/'+conf+'/'+t+'/p000-d-3-re/g_4'].value.real for t in infile[statename+'/bwd/'+conf].keys()[2:]], axis=0) for conf in cfgs]

O44datfw = O44datfw + [2*np.mean([infile[statename+'/fwd/'+conf+'/'+t+'/p000-d4-re/g_8'].value.real for t in infile[statename+'/fwd/'+conf].keys()[2:]], axis=0) for conf in cfgs] - [2*np.mean([infile[statename+'/fwd/'+conf+'/'+t+'/p000-d-4-re/g_8'].value.real for t in infile[statename+'/fwd/'+conf].keys()[2:]], axis=0) for conf in cfgs]
O44datbw = O44datbw + [2*np.mean([infile[statename+'/bwd/'+conf+'/'+t+'/p000-d4-re/g_8'].value.real for t in infile[statename+'/bwd/'+conf].keys()[2:]], axis=0) for conf in cfgs] - [2*np.mean([infile[statename+'/bwd/'+conf+'/'+t+'/p000-d-4-re/g_8'].value.real for t in infile[statename+'/bwd/'+conf].keys()[2:]], axis=0) for conf in cfgs]

#O34datfw = O34datfw + [2*np.mean([infile[statename+'/fwd/'+conf+'/'+t+'/p000-d3-re/g_8'].value.real for t in infile[statename+'/fwd/'+conf].keys()[2:]], axis=0) for conf in cfgs] - [2*np.mean([infile[statename+'/fwd/'+conf+'/'+t+'/p000-d-3-re/g_8'].value.real for t in infile[statename+'/fwd/'+conf].keys()[2:]], axis=0) for conf in cfgs]
#O34datbw = O34datbw + [2*np.mean([infile[statename+'/bwd/'+conf+'/'+t+'/p000-d3-re/g_8'].value.real for t in infile[statename+'/bwd/'+conf].keys()[2:]], axis=0) for conf in cfgs] - [2*np.mean([infile[statename+'/bwd/'+conf+'/'+t+'/p000-d-3-re/g_8'].value.real for t in infile[statename+'/bwd/'+conf].keys()[2:]], axis=0) for conf in cfgs]

#O43datfw = O43datfw + [2*np.mean([infile[statename+'/fwd/'+conf+'/'+t+'/p000-d4-re/g_4'].value.real for t in infile[statename+'/fwd/'+conf].keys()[2:]], axis=0) for conf in cfgs] - [2*np.mean([infile[statename+'/fwd/'+conf+'/'+t+'/p000-d-4-re/g_4'].value.real for t in infile[statename+'/fwd/'+conf].keys()[2:]], axis=0) for conf in cfgs]
#O43datbw = O43datbw + [2*np.mean([infile[statename+'/bwd/'+conf+'/'+t+'/p000-d4-re/g_4'].value.real for t in infile[statename+'/bwd/'+conf].keys()[2:]], axis=0) for conf in cfgs] - [2*np.mean([infile[statename+'/bwd/'+conf+'/'+t+'/p000-d-4-re/g_4'].value.real for t in infile[statename+'/bwd/'+conf].keys()[2:]], axis=0) for conf in cfgs]

#subO33dat = np.zeros((200,Nt,Nt,intops))
#subO44dat = np.zeros((200,Nt,Nt,intops))
#subO34dat = np.zeros((200,Nt,Nt,intops))
#subO43dat = np.zeros((200,Nt,Nt,intops))

#subO33dat = np.einsum('ijk,im->ijmk',twoptboot,vacfile["O33"]).real
#subO44dat = np.einsum('ijk,im->ijmk',twoptboot,vacfile["O44"]).real
#subO34dat = np.einsum('ijk,im->ijmk',twoptboot,vacfile["O34"]).real
#subO43dat = np.einsum('ijk,im->ijmk',twoptboot,vacfile["O43"]).real

#print(O33datfw[0,0])
#print(O33datbw[0,0])
#print(subO33dat[0,0])

#O33bootfw[:] = bootstrap_resample(O33datfw)[:].real
#O33bootbw[:] = bootstrap_resample(O33datbw)[:].real
#O33bootvac[:] = vacfile["O33"][:].real

#O43bootfw[:] = bootstrap_resample(O43datfw)[:].real
#O43bootbw[:] = bootstrap_resample(O43datbw)[:].real
#O43bootvac[:] = vacfile["O43"][:].real

O44bootfw[:] = bootstrap_resample(O44datfw)[:].real
O44bootbw[:] = bootstrap_resample(O44datbw)[:].real
O44bootvac[:] = vacfile["O44"][:].real

#O33bootsubbed[:] = (bootstrap_resample(0.5*(O33datfw+O33datbw)) - subO33dat)[:].real
#O44bootsubbed[:] = (bootstrap_resample(0.5*(O44datfw+O44datbw)) - subO44dat)[:].real
#O34bootsubbed[:] = (bootstrap_resample(0.5*(O34datfw+O34datbw)) - subO34dat)[:].real
#O43bootsubbed[:] = (bootstrap_resample(0.5*(O43datfw+O43datbw)) - subO43dat)[:].real
