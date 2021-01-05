#!/opt/software/anaconda3/bin/python

from multiprocessing import Pool
import h5py
import numpy as np
import time
import random

datapath = "/data/wombat/users/phiala/beamNPRanalysis/su3_24_48_b6p10050/"
datafilenameroot = "pion_beam_npr_matrix_k0p1248_"

L = 24
T = 48

outdataname = "pion_beam_npr_24_Sept6_k0p1248.h5"
outdatapath = "//data/wombat/users/phiala/beamNPRanalysis/su3_24_48_b6p10050/"
outfile = h5py.File(outdatapath+outdataname,"w")

configs = list(range(100,115))
#configs = list(range(110,140))
#configs.remove(129)
#configs.remove(121)
#configs.remove(130)
#configs.remove(127)
#configs = list(range(110,120))
#configs = [110]

etalist = [7,9,11]

Nboot = 200
#Nboot = 1
#momenta = [[1,1,5,1]]
momenta = [[2,2,2,2],[2,2,2,4],[2,2,2,6],[3,3,3,2],[3,3,3,4],[3,3,3,6],[3,3,3,8],[4,4,4,4],[4,4,4,6],[4,4,4,8]]

gammas = np.zeros((4,4,4),dtype=complex)
gammas[0] = gammas[0] + np.array([[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]])
gammas[1] = gammas[1] + np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
gammas[2] = gammas[2] + np.array([[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]])
gammas[3] = gammas[3] + np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])

gammaList = [np.identity(4,dtype=complex),gammas[0],gammas[1],np.matmul(gammas[1],gammas[0]),gammas[2],np.matmul(gammas[2],gammas[0]),np.matmul(gammas[2],gammas[1]),np.matmul(gammas[2],np.matmul(gammas[1],gammas[0])), gammas[3], np.matmul(gammas[3],gammas[0]),np.matmul(gammas[3],gammas[1]),np.matmul(gammas[3],np.matmul(gammas[1],gammas[0])),np.matmul(gammas[3],gammas[2]),np.matmul(gammas[3],np.matmul(gammas[2],gammas[0])),np.matmul(gammas[3],np.matmul(gammas[2],gammas[1])),np.matmul(np.matmul(gammas[3],gammas[2]),np.matmul(gammas[1],gammas[0]))]

LL = [L,L,L,T]

def bootstrap_resample(X, weights=None, n=None, nboots=Nboot):

    if n == None:
        n = len(X)

    if weights == None:
        weights = np.ones((len(X)))
        
    np.random.seed(5)

    ALL_resamples=np.zeros((nboots,)+X.shape[1:],dtype=complex)

    weights2=weights/float(np.sum(weights)) 
    ttmp = range(len(X))

    for ii in range(nboots):
        resample_i = np.random.choice(ttmp,p=weights2,size=len(X),replace=True)
        ALL_resamples[ii] = np.mean(X[resample_i],axis=0)

    return ALL_resamples


V = (L**3)*T

#get two-points
for mom in momenta:
#for mom in [momenta[0]]:
    
    groupstring = str(mom[0])+str(mom[1])+str(mom[2])+str(mom[3])
    momgroup = outfile.create_group(groupstring)

    momdat = momgroup.create_dataset("mom", (4,), dtype=int)
    momdat[:] = mom

    twopt = np.zeros((len(configs),3,4,3,4),dtype=complex)
    INVERSEtwoptBOOT = np.zeros((Nboot,3,4,3,4),dtype=complex)

    for cfg in range(len(configs)):

#        print cfg
        twoptfile = h5py.File(datapath+datafilenameroot+str(configs[cfg])+".h5","r")
        tmp = twoptfile.get("/prop/p"+groupstring).value
        twopt[cfg,:,:,:,:] = np.einsum('ijab->aibj',tmp[:,:,:,:])


    twoptBOOT = bootstrap_resample(twopt)

    for b in range(Nboot):
        INVERSEtwoptBOOT[b] = np.linalg.tensorinv(twoptBOOT[b])

    print(Nboot)

    Zq = V*(1j)*(sum(np.einsum('ij,bajai->b',gammas[i],INVERSEtwoptBOOT)*np.sin(2*np.pi*(mom[i]+[0,0,0,0.5][i])/LL[i]) for i in range(4)))/(12*sum((np.sin(2*np.pi*(mom[i]+[0,0,0,0.5][i])/LL[i]))**2 for i in range(4)))

    print(Zq)

    Zqdat = momgroup.create_dataset("Zq", (Nboot,), dtype=complex)
    Zqdat[:] = Zq[:]
    
 #   for eta in [etalist[0]]:
    for eta in etalist:
        print("eta = "+str(eta))

        etadatV = momgroup.create_dataset("V_eta"+str(eta), (2*eta+1,2*eta+1,2,16,16,Nboot), dtype=complex)
        etadatZinv = momgroup.create_dataset("Zinv_eta"+str(eta), (2*eta+1,2*eta+1,2,16,16,Nboot), dtype=complex)

        for mu in range(2):


            staplestring = "/eta"+str(eta)+"/mu"+str(mu)
            threept = np.zeros((len(configs),2*eta+1,2*eta+1,16,3,4,3,4),dtype=complex)
                    
            for cfg in range(len(configs)):

                print(cfg)
                threeptfile = h5py.File(datapath+datafilenameroot+str(configs[cfg])+".h5","r")
                tmp = threeptfile.get("/threept"+staplestring+"/bTsign1/p"+groupstring).value
                tmpneg = threeptfile.get("/threept"+staplestring+"/bTsign-1/p"+groupstring).value                
                
                threept[cfg,eta::,:,:,:,:,:] = np.einsum('tzgijab->tzgaibj',tmp)
                threept[cfg,0:(eta+1),:,:,:,:,:] = np.einsum('tzgijab->tzgaibj',tmpneg[::-1])
                
            threeptBOOT = bootstrap_resample(threept)
                
            Lambda = V*np.einsum('baick,btzgckdl,bdlej->btzgaiej',INVERSEtwoptBOOT,threeptBOOT,INVERSEtwoptBOOT)

            Vproj = np.zeros((2*eta+1,2*eta+1,len(gammaList),len(gammaList),Nboot),dtype=complex)
               
            for gamma in range(len(gammaList)):

                #print(gamma)
                Vproj[:,:,:,gamma,:] = np.einsum('btzgaiaj,ji->tzgb',Lambda,gammaList[gamma])

            etadatV[:,:,mu,:,:,:] = Vproj[:,:,:,:]
    
            ipdotb = [[1/(np.exp((1j*2*np.pi/L)*(bz*mom[2]+bT*mom[mu]))) for bz in range(-eta,eta+1)] for bT in range(-eta,eta+1)]

            tmp = np.einsum('tzghb,tz->tzghb',Vproj,ipdotb)/(6*Zq[:])
            etadatZinv[:,:,mu,:,:,:] = tmp

