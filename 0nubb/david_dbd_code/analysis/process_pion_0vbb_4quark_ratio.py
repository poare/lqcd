#!/usr/bin/python3

from __future__ import print_function
import sys
import numpy as np
import os.path

###################################################################
## A little utility function to process raw Z_{V} correlator data
## and compute each jackknife sample of ratio we conventially fit.
##
## David Murphy (djm2131@columbia.edu)
## 10/05/2016
###################################################################

if len(sys.argv) != 12:
  print("Usage: ./process_pion0vbb_4quark_ratio.py <pion_stem> <pion0vbb_4quark_stem_in> <pion0vbb_4quark_ref_stem_in> <pion0vbb_4quark_stem_out> <traj_start> <traj_end> <traj_inc> <L> <T> <Z_stem> <m_stem>")
  exit(0)

pion_stem      = sys.argv[1]
pion0vbb_4quark_stem_in     = sys.argv[2]
pion0vbb_4quark_stem_ref_in = sys.argv[3]
pion0vbb_4quark_stem_out    = sys.argv[4]
traj_start     = int(sys.argv[5])
traj_end       = int(sys.argv[6])
traj_incr      = int(sys.argv[7])
L              = int(sys.argv[8])
T              = int(sys.argv[9])
Z_stem         = sys.argv[10]
m_stem         = sys.argv[11]

Ntraj                 = (traj_end - traj_start)//traj_incr + 1
pion_raw              = np.zeros((Ntraj,T))
pion_cv               = np.zeros((T))
pion_jacks            = np.zeros((Ntraj,T))
pion0vbb_4quark_1_re  = np.zeros((Ntraj,T,T))
# pion0vbb_4quark_1_im  = np.zeros((Ntraj,T,T))
pion0vbb_4quark_2_re  = np.zeros((Ntraj,T,T))
# pion0vbb_4quark_2_im  = np.zeros((Ntraj,T,T))
pion0vbb_4quark_3_re  = np.zeros((Ntraj,T,T))
# pion0vbb_4quark_3_im  = np.zeros((Ntraj,T,T))
pion0vbb_4quark_1p_re = np.zeros((Ntraj,T,T))
# pion0vbb_4quark_1p_im = np.zeros((Ntraj,T,T))
pion0vbb_4quark_2p_re = np.zeros((Ntraj,T,T))
# pion0vbb_4quark_2p_im = np.zeros((Ntraj,T,T))
pion0vbb_4quark_3p_re = np.zeros((Ntraj,T,T))
# pion0vbb_4quark_3p_im = np.zeros((Ntraj,T,T))

# Fetch fit parameter jackknife samples
V       = L**3
Z_cv    = np.genfromtxt(Z_stem + ".dat")
Z_jacks = np.genfromtxt(Z_stem + "_jacks.dat")
m_cv    = np.genfromtxt(m_stem + ".dat")            # pion mass
m_jacks = np.genfromtxt(m_stem + "_jacks.dat")

# Fetch the pion data
for traj in range(traj_start, traj_end+1, traj_incr):
  idx = (traj-traj_start)//traj_incr
  fs = pion_stem + "." + str(traj)
  print("Parsing file " + fs + "...")
  data = np.genfromtxt(fs)
  pion_raw[idx,:] = data[:,1]       # pion_raw[idx, :] is the same as the real part of my C2_tavg[idx, :]
print("done.\n")

# Subtract thermal state
pion_cv = np.mean(pion_raw, axis=0)     # pion_raw[cfg_idx, t], pion_cv[t] is averaged over configs
for t in range(0,T):
  pion_cv[t] -= 0.5 * pion_cv[T//2] * np.exp(m_cv*(t-T//2))
  # if t == 0:
  #   continue
  # pion_raw[idx,t] *= pion_raw[idx,T-t]
for traj in range(traj_start, traj_end+1, traj_incr):
  idx = (traj-traj_start)//traj_incr
  pion_jacks[idx,:] = np.mean( np.delete(pion_raw, idx, axis=0), axis=0 )
  for t in range(0,T):
    pion_jacks[idx,t] -= 0.5 * pion_jacks[idx,T//2] * np.exp(m_jacks[idx]*(t-T//2))
    # if t == 0:
    #   continue
    # pion_jacks[idx,t] *= pion_jacks[idx,T-t]

# Fetch the pion0vbb_4quark data
for traj in range(traj_start, traj_end+1, traj_incr):
  idx = (traj-traj_start)//traj_incr
  for sep in range(0,T):
    fs = pion0vbb_4quark_stem_in + "." + str(sep) + "." + str(traj)
    fs_ref = pion0vbb_4quark_stem_ref_in + "." + str(sep) + "." + str(traj)
    if os.path.isfile(fs):
      print("Parsing file " + fs + "...")
      data = np.genfromtxt(fs)
      data_ref = np.genfromtxt(fs_ref)
      data[:,1:17] *= 4.0
      data_ref[:,1:17] *= 4.0
      # pion0vbb_4quark_1_re[cfg_idx, t, tsep] where tsep is the index on the file name and t is the row index in the file
      pion0vbb_4quark_1_re[idx,:,sep]  = -2.0 * ( data[:,17]  - data[:,19] - data[:,33] + data[:,35] + data[:,25] - data[:,27] - data[:,41] + data[:,43] )
      pion0vbb_4quark_1_re[idx,:,sep] += -2.0 * ( data_ref[:,17]  - data_ref[:,19] - data_ref[:,33] + data_ref[:,35] + data_ref[:,25] - data_ref[:,27] - data_ref[:,41] + data_ref[:,43] )
      pion0vbb_4quark_1_re[idx,:,sep] *= 0.5
      # pion0vbb_4quark_1_re[idx,:,sep] *= 0.25
      # pion0vbb_4quark_1_im[idx,:,sep]  = -2.0 * ( data[:,18]  - data[:,20] - data[:,34] + data[:,36] + data[:,26] - data[:,28] - data[:,42] + data[:,44] )
      pion0vbb_4quark_2_re[idx,:,sep]  = -4.0 * ( data[:,1]  - data[:,3] + data[:,9] - data[:,11] )
      pion0vbb_4quark_2_re[idx,:,sep] += -4.0 * ( data_ref[:,1]  - data_ref[:,3] + data_ref[:,9] - data_ref[:,11] )
      pion0vbb_4quark_2_re[idx,:,sep] *= 0.5
      # pion0vbb_4quark_2_re[idx,:,sep] *= 0.25
      # pion0vbb_4quark_2_im[idx,:,sep]  = -4.0 * ( data[:,2]  - data[:,4] + data[:,10] - data[:,12] )
      pion0vbb_4quark_3_re[idx,:,sep]  = -4.0 * ( data[:,17] - data[:,19] + data[:,41] - data[:,43] )
      pion0vbb_4quark_3_re[idx,:,sep] += -4.0 * ( data_ref[:,17] - data_ref[:,19] + data_ref[:,41] - data_ref[:,43] )
      pion0vbb_4quark_3_re[idx,:,sep] *= 0.5
      # pion0vbb_4quark_3_re[idx,:,sep] *= 0.25
      # pion0vbb_4quark_3_im[idx,:,sep]  = -4.0 * ( data[:,18] - data[:,20] + data[:,42] - data[:,44] )
      pion0vbb_4quark_1p_re[idx,:,sep]  = -2.0 * ( data[:,21]  - data[:,23] - data[:,37] + data[:,39] + data[:,29] - data[:,31] - data[:,45] + data[:,47] )
      pion0vbb_4quark_1p_re[idx,:,sep] += -2.0 * ( data_ref[:,21]  - data_ref[:,23] - data_ref[:,37] + data_ref[:,39] + data_ref[:,29] - data_ref[:,31] - data_ref[:,45] + data_ref[:,47] )
      pion0vbb_4quark_1p_re[idx,:,sep] *= 0.5
      # pion0vbb_4quark_1p_re[idx,:,sep] *= 0.25
      # pion0vbb_4quark_1p_im[idx,:,sep]  = -2.0 * ( data[:,22]  - data[:,24] - data[:,38] + data[:,40] + data[:,30] - data[:,32] - data[:,46] + data[:,48] )
      pion0vbb_4quark_2p_re[idx,:,sep]  = -4.0 * ( data[:,5]  - data[:,7] + data[:,13] - data[:,15] )
      pion0vbb_4quark_2p_re[idx,:,sep] += -4.0 * ( data_ref[:,5]  - data_ref[:,7] + data_ref[:,13] - data_ref[:,15] )
      pion0vbb_4quark_2p_re[idx,:,sep] *= 0.5
      # pion0vbb_4quark_2p_re[idx,:,sep] *= 0.25
      # pion0vbb_4quark_2p_im[idx,:,sep]  = -4.0 * ( data[:,6]  - data[:,8] + data[:,14] - data[:,16] )
      pion0vbb_4quark_3p_re[idx,:,sep]  = -4.0 * ( data[:,21] - data[:,23] + data[:,45] - data[:,47] )
      pion0vbb_4quark_3p_re[idx,:,sep] += -4.0 * ( data_ref[:,21] - data_ref[:,23] + data_ref[:,45] - data_ref[:,47] )
      pion0vbb_4quark_3p_re[idx,:,sep] *= 0.5
      # pion0vbb_4quark_3p_re[idx,:,sep] *= 0.25
      # pion0vbb_4quark_3p_im[idx,:,sep]  = -4.0 * ( data[:,22] - data[:,24] + data[:,46] - data[:,48] )

print("\nComputing central values...")

# Compute central value and error
pion0vbb_4quark_ratio_1_re  = np.zeros((T))
# pion0vbb_4quark_ratio_1_im  = np.zeros((T))
pion0vbb_4quark_ratio_2_re  = np.zeros((T))
# pion0vbb_4quark_ratio_2_im  = np.zeros((T))
pion0vbb_4quark_ratio_3_re  = np.zeros((T))
# pion0vbb_4quark_ratio_3_im  = np.zeros((T))
pion0vbb_4quark_ratio_1p_re = np.zeros((T))
# pion0vbb_4quark_ratio_1p_im = np.zeros((T))
pion0vbb_4quark_ratio_2p_re = np.zeros((T))
# pion0vbb_4quark_ratio_2p_im = np.zeros((T))
pion0vbb_4quark_ratio_3p_re = np.zeros((T))
# pion0vbb_4quark_ratio_3p_im = np.zeros((T))
for sep in range(0,T):

  # if sep == 0:
  #   continue

  # Skip separations with no data
  if pion0vbb_4quark_1_re[0,1,sep] == 0.0:
    continue

  # Compute pion0vbb_4quark ratio
  num_1_re  = np.mean(pion0vbb_4quark_1_re[:,:,sep], axis=0)
  # num_1_im  = np.mean(pion0vbb_4quark_1_im[:,:,sep], axis=0)
  num_2_re  = np.mean(pion0vbb_4quark_2_re[:,:,sep], axis=0)
  # num_2_im  = np.mean(pion0vbb_4quark_2_im[:,:,sep], axis=0)
  num_3_re  = np.mean(pion0vbb_4quark_3_re[:,:,sep], axis=0)
  # num_3_im  = np.mean(pion0vbb_4quark_3_im[:,:,sep], axis=0)
  num_1p_re = np.mean(pion0vbb_4quark_1p_re[:,:,sep], axis=0)
  # num_1p_im = np.mean(pion0vbb_4quark_1p_im[:,:,sep], axis=0)
  num_2p_re = np.mean(pion0vbb_4quark_2p_re[:,:,sep], axis=0)
  # num_2p_im = np.mean(pion0vbb_4quark_2p_im[:,:,sep], axis=0)
  num_3p_re = np.mean(pion0vbb_4quark_3p_re[:,:,sep], axis=0)
  # num_3p_im = np.mean(pion0vbb_4quark_3p_im[:,:,sep], axis=0)
  for t in range(0,T):
    pion0vbb_4quark_ratio_1_re[t]  = 2.0*m_cv * num_1_re[t]  / pion_cv[sep]       # this is the matrix element <\pi^+ | O_i | \pi_i>
    # pion0vbb_4quark_ratio_1_im[t]  = 2.0*m_cv * num_1_im[t]  / pion_cv[sep]
    pion0vbb_4quark_ratio_2_re[t]  = 2.0*m_cv * num_2_re[t]  / pion_cv[sep]
    # pion0vbb_4quark_ratio_2_im[t]  = 2.0*m_cv * num_2_im[t]  / pion_cv[sep]
    pion0vbb_4quark_ratio_3_re[t]  = 2.0*m_cv * num_3_re[t]  / pion_cv[sep]
    # pion0vbb_4quark_ratio_3_im[t]  = 2.0*m_cv * num_3_im[t]  / pion_cv[sep]
    pion0vbb_4quark_ratio_1p_re[t] = 2.0*m_cv * num_1p_re[t] / pion_cv[sep]
    # pion0vbb_4quark_ratio_1p_im[t] = 2.0*m_cv * num_1p_im[t] / pion_cv[sep]
    pion0vbb_4quark_ratio_2p_re[t] = 2.0*m_cv * num_2p_re[t] / pion_cv[sep]
    # pion0vbb_4quark_ratio_2p_im[t] = 2.0*m_cv * num_2p_im[t] / pion_cv[sep]
    pion0vbb_4quark_ratio_3p_re[t] = 2.0*m_cv * num_3p_re[t] / pion_cv[sep]
    # pion0vbb_4quark_ratio_3p_im[t] = 2.0*m_cv * num_3p_im[t] / pion_cv[sep]
    # if t == 0:
    #   continue
    # print(t)
    # pion0vbb_4quark_ratio_1_re[t]  = num_1_re[t]  / ( pion_cv[sep] * pion_cv[T-sep] )
    # pion0vbb_4quark_ratio_1_im[t]  = num_1_im[t]  / ( pion_cv[sep] * pion_cv[T-sep] )
    # pion0vbb_4quark_ratio_2_re[t]  = num_2_re[t]  / ( pion_cv[sep] * pion_cv[T-sep] )
    # pion0vbb_4quark_ratio_2_im[t]  = num_2_im[t]  / ( pion_cv[sep] * pion_cv[T-sep] )
    # pion0vbb_4quark_ratio_3_re[t]  = num_3_re[t]  / ( pion_cv[sep] * pion_cv[T-sep] )
    # pion0vbb_4quark_ratio_3_im[t]  = num_3_im[t]  / ( pion_cv[sep] * pion_cv[T-sep] )
    # pion0vbb_4quark_ratio_1p_re[t] = num_1p_re[t] / ( pion_cv[sep] * pion_cv[T-sep] )
    # pion0vbb_4quark_ratio_1p_im[t] = num_1p_im[t] / ( pion_cv[sep] * pion_cv[T-sep] )
    # pion0vbb_4quark_ratio_2p_re[t] = num_2p_re[t] / ( pion_cv[sep] * pion_cv[T-sep] )
    # pion0vbb_4quark_ratio_2p_im[t] = num_2p_im[t] / ( pion_cv[sep] * pion_cv[T-sep] )
    # pion0vbb_4quark_ratio_3p_re[t] = num_3p_re[t] / ( pion_cv[sep] * pion_cv[T-sep] )
    # pion0vbb_4quark_ratio_3p_im[t] = num_3p_im[t] / ( pion_cv[sep] * pion_cv[T-sep] )
  jacks_1_re  = np.zeros((Ntraj,T))
  # jacks_1_im  = np.zeros((Ntraj,T))
  jacks_2_re  = np.zeros((Ntraj,T))
  # jacks_2_im  = np.zeros((Ntraj,T))
  jacks_3_re  = np.zeros((Ntraj,T))
  # jacks_3_im  = np.zeros((Ntraj,T))
  jacks_1p_re = np.zeros((Ntraj,T))
  # jacks_1p_im = np.zeros((Ntraj,T))
  jacks_2p_re = np.zeros((Ntraj,T))
  # jacks_2p_im = np.zeros((Ntraj,T))
  jacks_3p_re = np.zeros((Ntraj,T))
  # jacks_3p_im = np.zeros((Ntraj,T))
  for sample in range(0,Ntraj):
    # den    = np.mean( np.delete(pion, sample, axis=0), axis=0)
    # den    = np.ones(den.shape)
    num_1_re  = np.mean( np.delete(pion0vbb_4quark_1_re[:,:,sep], sample, axis=0), axis=0)
    # num_1_re[t] is evaluated at sep, t is the
    # num_1_im  = np.mean( np.delete(pion0vbb_4quark_1_im[:,:,sep], sample, axis=0), axis=0)
    num_2_re  = np.mean( np.delete(pion0vbb_4quark_2_re[:,:,sep], sample, axis=0), axis=0)
    # num_2_im  = np.mean( np.delete(pion0vbb_4quark_2_im[:,:,sep], sample, axis=0), axis=0)
    num_3_re  = np.mean( np.delete(pion0vbb_4quark_3_re[:,:,sep], sample, axis=0), axis=0)
    # num_3_im  = np.mean( np.delete(pion0vbb_4quark_3_im[:,:,sep], sample, axis=0), axis=0)
    num_1p_re = np.mean( np.delete(pion0vbb_4quark_1p_re[:,:,sep], sample, axis=0), axis=0)
    # num_1p_im = np.mean( np.delete(pion0vbb_4quark_1p_im[:,:,sep], sample, axis=0), axis=0)
    num_2p_re = np.mean( np.delete(pion0vbb_4quark_2p_re[:,:,sep], sample, axis=0), axis=0)
    # num_2p_im = np.mean( np.delete(pion0vbb_4quark_2p_im[:,:,sep], sample, axis=0), axis=0)
    num_3p_re = np.mean( np.delete(pion0vbb_4quark_3p_re[:,:,sep], sample, axis=0), axis=0)
    # num_3p_im = np.mean( np.delete(pion0vbb_4quark_3p_im[:,:,sep], sample, axis=0), axis=0)
    for t in range(0,T):
      jacks_1_re[sample,t]  = 2.0*m_jacks[sample] * num_1_re[t]  / pion_jacks[sample, sep]
      # jacks_1_im[sample,t]  = 2.0*m_jacks[sample] * num_1_im[t]  / pion_jacks[sample, sep]
      jacks_2_re[sample,t]  = 2.0*m_jacks[sample] * num_2_re[t]  / pion_jacks[sample, sep]
      # jacks_2_im[sample,t]  = 2.0*m_jacks[sample] * num_2_im[t]  / pion_jacks[sample, sep]
      jacks_3_re[sample,t]  = 2.0*m_jacks[sample] * num_3_re[t]  / pion_jacks[sample, sep]
      # jacks_3_im[sample,t]  = 2.0*m_jacks[sample] * num_3_im[t]  / pion_jacks[sample, sep]
      jacks_1p_re[sample,t] = 2.0*m_jacks[sample] * num_1p_re[t] / pion_jacks[sample, sep]
      # jacks_1p_im[sample,t] = 2.0*m_jacks[sample] * num_1p_im[t] / pion_jacks[sample, sep]
      jacks_2p_re[sample,t] = 2.0*m_jacks[sample] * num_2p_re[t] / pion_jacks[sample, sep]
      # jacks_2p_im[sample,t] = 2.0*m_jacks[sample] * num_2p_im[t] / pion_jacks[sample, sep]
      jacks_3p_re[sample,t] = 2.0*m_jacks[sample] * num_3p_re[t] / pion_jacks[sample, sep]
      # jacks_3p_im[sample,t] = 2.0*m_jacks[sample] * num_3p_im[t] / pion_jacks[sample, sep]
      # if t == 0:
      #   continue
      # jacks_1_re[sample,t]  = num_1_re[t]  / ( pion_jacks[sample, sep] * pion_jacks[sample, t-sep] )
      # jacks_1_im[sample,t]  = num_1_im[t]  / ( pion_jacks[sample, sep] * pion_jacks[sample, t-sep] )
      # jacks_2_re[sample,t]  = num_2_re[t]  / ( pion_jacks[sample, sep] * pion_jacks[sample, t-sep] )
      # jacks_2_im[sample,t]  = num_2_im[t]  / ( pion_jacks[sample, sep] * pion_jacks[sample, t-sep] )
      # jacks_3_re[sample,t]  = num_3_re[t]  / ( pion_jacks[sample, sep] * pion_jacks[sample, t-sep] )
      # jacks_3_im[sample,t]  = num_3_im[t]  / ( pion_jacks[sample, sep] * pion_jacks[sample, t-sep] )
      # jacks_1p_re[sample,t] = num_1p_re[t] / ( pion_jacks[sample, sep] * pion_jacks[sample, t-sep] )
      # jacks_1p_im[sample,t] = num_1p_im[t] / ( pion_jacks[sample, sep] * pion_jacks[sample, t-sep] )
      # jacks_2p_re[sample,t] = num_2p_re[t] / ( pion_jacks[sample, sep] * pion_jacks[sample, t-sep] )
      # jacks_2p_im[sample,t] = num_2p_im[t] / ( pion_jacks[sample, sep] * pion_jacks[sample, t-sep] )
      # jacks_3p_re[sample,t] = num_3p_re[t] / ( pion_jacks[sample, sep] * pion_jacks[sample, t-sep] )
      # jacks_3p_im[sample,t] = num_3p_im[t] / ( pion_jacks[sample, sep] * pion_jacks[sample, t-sep] )
  weights_1_re  = np.sqrt(Ntraj-1.0)*np.std(jacks_1_re, axis=0, ddof=0)
  # weights_1_im  = np.sqrt(Ntraj-1.0)*np.std(jacks_1_im, axis=0, ddof=0)
  weights_2_re  = np.sqrt(Ntraj-1.0)*np.std(jacks_2_re, axis=0, ddof=0)
  # weights_2_im  = np.sqrt(Ntraj-1.0)*np.std(jacks_2_im, axis=0, ddof=0)
  weights_3_re  = np.sqrt(Ntraj-1.0)*np.std(jacks_3_re, axis=0, ddof=0)
  # weights_3_im  = np.sqrt(Ntraj-1.0)*np.std(jacks_3_im, axis=0, ddof=0)
  weights_1p_re = np.sqrt(Ntraj-1.0)*np.std(jacks_1p_re, axis=0, ddof=0)
  # weights_1p_im = np.sqrt(Ntraj-1.0)*np.std(jacks_1p_im, axis=0, ddof=0)
  weights_2p_re = np.sqrt(Ntraj-1.0)*np.std(jacks_2p_re, axis=0, ddof=0)
  # weights_2p_im = np.sqrt(Ntraj-1.0)*np.std(jacks_2p_im, axis=0, ddof=0)
  weights_3p_re = np.sqrt(Ntraj-1.0)*np.std(jacks_3p_re, axis=0, ddof=0)
  # weights_3p_im = np.sqrt(Ntraj-1.0)*np.std(jacks_3p_im, axis=0, ddof=0)
  mcov_1_re = np.zeros((T,T))
  mcov_2_re = np.zeros((T,T))
  mcov_3_re = np.zeros((T,T))
  mcov_1p_re = np.zeros((T,T))
  mcov_2p_re = np.zeros((T,T))
  mcov_3p_re = np.zeros((T,T))
  for t1 in range(T):
    for t2 in range(T):
      mcov_1_re[t1,t2] = ( float(Ntraj) - 1.0 ) * np.mean( (jacks_1_re[:,t1] - pion0vbb_4quark_ratio_1_re[t1]) * (jacks_1_re[:,t2] - pion0vbb_4quark_ratio_1_re[t2]) )
      mcov_2_re[t1,t2] = ( float(Ntraj) - 1.0 ) * np.mean( (jacks_2_re[:,t1] - pion0vbb_4quark_ratio_2_re[t1]) * (jacks_2_re[:,t2] - pion0vbb_4quark_ratio_2_re[t2]) )
      mcov_3_re[t1,t2] = ( float(Ntraj) - 1.0 ) * np.mean( (jacks_3_re[:,t1] - pion0vbb_4quark_ratio_3_re[t1]) * (jacks_3_re[:,t2] - pion0vbb_4quark_ratio_3_re[t2]) )
      mcov_1p_re[t1,t2] = ( float(Ntraj) - 1.0 ) * np.mean( (jacks_1p_re[:,t1] - pion0vbb_4quark_ratio_1p_re[t1]) * (jacks_1p_re[:,t2] - pion0vbb_4quark_ratio_1p_re[t2]) )
      mcov_2p_re[t1,t2] = ( float(Ntraj) - 1.0 ) * np.mean( (jacks_2p_re[:,t1] - pion0vbb_4quark_ratio_2p_re[t1]) * (jacks_2p_re[:,t2] - pion0vbb_4quark_ratio_2p_re[t2]) )
      mcov_3p_re[t1,t2] = ( float(Ntraj) - 1.0 ) * np.mean( (jacks_3p_re[:,t1] - pion0vbb_4quark_ratio_3p_re[t1]) * (jacks_3p_re[:,t2] - pion0vbb_4quark_ratio_3p_re[t2]) )

  # Write to disk
  f = open(pion0vbb_4quark_stem_out + "_1." + str(sep), 'w')
  for t in range(0,T):
    # line = "{0:3d} {1:1.10e} {2:1.10e} {3:1.10e} {4:1.10e}".format(t, pion0vbb_4quark_ratio_1_re[t], weights_1_re[t], pion0vbb_4quark_ratio_1_im[t], weights_1_im[t])
    line = "{0:3d} {1:1.10e} {2:1.10e}".format(t, pion0vbb_4quark_ratio_1_re[t], weights_1_re[t])
    print(line, file=f)
  f.close()
  with open(pion0vbb_4quark_stem_out + "_1." + str(sep) + ".mcov", "w") as f:
    for t1 in range(T):
      line = "{0:1.10e}".format(mcov_1_re[t1,0])
      for t2 in range(1,T):
        line += " {0:1.10e}".format(mcov_1_re[t1,t2])
      print(line, file=f)

  f = open(pion0vbb_4quark_stem_out + "_2." + str(sep), 'w')
  for t in range(0,T):
    # line = "{0:3d} {1:1.10e} {2:1.10e} {3:1.10e} {4:1.10e}".format(t, pion0vbb_4quark_ratio_2_re[t], weights_2_re[t], pion0vbb_4quark_ratio_2_im[t], weights_2_im[t])
    line = "{0:3d} {1:1.10e} {2:1.10e}".format(t, pion0vbb_4quark_ratio_2_re[t], weights_2_re[t])
    print(line, file=f)
  f.close()
  with open(pion0vbb_4quark_stem_out + "_2." + str(sep) + ".mcov", "w") as f:
    for t1 in range(T):
      line = "{0:1.10e}".format(mcov_2_re[t1,0])
      for t2 in range(1,T):
        line += " {0:1.10e}".format(mcov_2_re[t1,t2])
      print(line, file=f)

  f = open(pion0vbb_4quark_stem_out + "_3." + str(sep), 'w')
  for t in range(0,T):
    # line = "{0:3d} {1:1.10e} {2:1.10e} {3:1.10e} {4:1.10e}".format(t, pion0vbb_4quark_ratio_3_re[t], weights_3_re[t], pion0vbb_4quark_ratio_3_im[t], weights_3_im[t])
    line = "{0:3d} {1:1.10e} {2:1.10e}".format(t, pion0vbb_4quark_ratio_3_re[t], weights_3_re[t])
    print(line, file=f)
  f.close()
  with open(pion0vbb_4quark_stem_out + "_3." + str(sep) + ".mcov", "w") as f:
    for t1 in range(T):
      line = "{0:1.10e}".format(mcov_3_re[t1,0])
      for t2 in range(1,T):
        line += " {0:1.10e}".format(mcov_3_re[t1,t2])
      print(line, file=f)

  f = open(pion0vbb_4quark_stem_out + "_1p." + str(sep), 'w')
  for t in range(0,T):
    # line = "{0:3d} {1:1.10e} {2:1.10e} {3:1.10e} {4:1.10e}".format(t, pion0vbb_4quark_ratio_1p_re[t], weights_1p_re[t], pion0vbb_4quark_ratio_1p_im[t], weights_1p_im[t])
    line = "{0:3d} {1:1.10e} {2:1.10e}".format(t, pion0vbb_4quark_ratio_1p_re[t], weights_1p_re[t])
    print(line, file=f)
  f.close()
  with open(pion0vbb_4quark_stem_out + "_1p." + str(sep) + ".mcov", "w") as f:
    for t1 in range(T):
      line = "{0:1.10e}".format(mcov_1p_re[t1,0])
      for t2 in range(1,T):
        line += " {0:1.10e}".format(mcov_1p_re[t1,t2])
      print(line, file=f)

  f = open(pion0vbb_4quark_stem_out + "_2p." + str(sep), 'w')
  for t in range(0,T):
    # line = "{0:3d} {1:1.10e} {2:1.10e} {3:1.10e} {4:1.10e}".format(t, pion0vbb_4quark_ratio_2p_re[t], weights_2p_re[t], pion0vbb_4quark_ratio_2p_im[t], weights_2p_im[t])
    line = "{0:3d} {1:1.10e} {2:1.10e}".format(t, pion0vbb_4quark_ratio_2p_re[t], weights_2p_re[t])
    print(line, file=f)
  f.close()
  with open(pion0vbb_4quark_stem_out + "_2p." + str(sep) + ".mcov", "w") as f:
    for t1 in range(T):
      line = "{0:1.10e}".format(mcov_2p_re[t1,0])
      for t2 in range(1,T):
        line += " {0:1.10e}".format(mcov_2p_re[t1,t2])
      print(line, file=f)

  f = open(pion0vbb_4quark_stem_out + "_3p." + str(sep), 'w')
  for t in range(0,T):
    # line = "{0:3d} {1:1.10e} {2:1.10e} {3:1.10e} {4:1.10e}".format(t, pion0vbb_4quark_ratio_3p_re[t], weights_3p_re[t], pion0vbb_4quark_ratio_3p_im[t], weights_3p_im[t])
    line = "{0:3d} {1:1.10e} {2:1.10e}".format(t, pion0vbb_4quark_ratio_3p_re[t], weights_3p_re[t])
    print(line, file=f)
  f.close()
  with open(pion0vbb_4quark_stem_out + "_3p." + str(sep) + ".mcov", "w") as f:
    for t1 in range(T):
      line = "{0:1.10e}".format(mcov_3p_re[t1,0])
      for t2 in range(1,T):
        line += " {0:1.10e}".format(mcov_3p_re[t1,t2])
      print(line, file=f)

# Double jackknife!
for jidx in range(0, Ntraj):

  this_traj = traj_start + jidx * traj_incr

  print("\nComputing jackknife for trajectory {0:d}...".format(this_traj))

  # Subtract thermal state
  pion_cv = np.mean(np.delete(pion_raw, jidx, axis=0), axis=0)
  pion_jjacks = np.zeros((Ntraj-1,T))
  for t in range(0,T):
    pion_cv[t] -= 0.5 * pion_cv[T//2] * np.exp(m_cv*(t-T//2))
    # if t == 0:
    #   continue
    # pion_cv[t] *= pion_cv[T-t]
    for idx in range(0, Ntraj-1):
      pion_jjacks[idx,:] = np.mean( np.delete( np.delete(pion_raw, jidx, axis=0), idx, axis=0), axis=0 )
      for t in range(0,T):
        pion_jjacks[idx,t] -= 0.5 * pion_jjacks[idx,T//2] * np.exp(m_jacks[idx]*(t-T//2))
        # if t == 0:
        #   continue
        # pion_jjacks[idx,t] *= pion_jjacks[idx,T-t]

  # Compute central value and error
  ratio_1  = np.zeros((T))
  ratio_2  = np.zeros((T))
  ratio_3  = np.zeros((T))
  ratio_1p = np.zeros((T))
  ratio_2p = np.zeros((T))
  ratio_3p = np.zeros((T))

  for sep in range(0,T):

    # Skip separations with no data
    if pion0vbb_4quark_1_re[0,1,sep] == 0.0:
      continue

    # Compute pion0vbb_4quark ratio
    num_1  = np.mean( np.delete(pion0vbb_4quark_1_re, jidx, axis=0)[:,:,sep], axis=0)
    num_2  = np.mean( np.delete(pion0vbb_4quark_2_re, jidx, axis=0)[:,:,sep], axis=0)
    num_3  = np.mean( np.delete(pion0vbb_4quark_3_re, jidx, axis=0)[:,:,sep], axis=0)
    num_1p = np.mean( np.delete(pion0vbb_4quark_1p_re, jidx, axis=0)[:,:,sep], axis=0)
    num_2p = np.mean( np.delete(pion0vbb_4quark_2p_re, jidx, axis=0)[:,:,sep], axis=0)
    num_3p = np.mean( np.delete(pion0vbb_4quark_3p_re, jidx, axis=0)[:,:,sep], axis=0)
    for t in range(0,T):
      ratio_1[t]  = 2.0*m_cv * num_1[t]  / pion_cv[sep]     # He's already jackknifed, so the data stored at each file path is already resampled.
      ratio_2[t]  = 2.0*m_cv * num_2[t]  / pion_cv[sep]
      ratio_3[t]  = 2.0*m_cv * num_3[t]  / pion_cv[sep]
      ratio_1p[t] = 2.0*m_cv * num_1p[t] / pion_cv[sep]
      ratio_2p[t] = 2.0*m_cv * num_2p[t] / pion_cv[sep]
      ratio_3p[t] = 2.0*m_cv * num_3p[t] / pion_cv[sep]
    jacks_1  = np.zeros((Ntraj-1,T))
    jacks_2  = np.zeros((Ntraj-1,T))
    jacks_3  = np.zeros((Ntraj-1,T))
    jacks_1p = np.zeros((Ntraj-1,T))
    jacks_2p = np.zeros((Ntraj-1,T))
    jacks_3p = np.zeros((Ntraj-1,T))
    for sample in range(0,Ntraj-1):
      num_1  = np.mean( np.delete( np.delete(pion0vbb_4quark_1_re, jidx, axis=0)[:,:,sep], sample, axis=0), axis=0)
      num_2  = np.mean( np.delete( np.delete(pion0vbb_4quark_2_re, jidx, axis=0)[:,:,sep], sample, axis=0), axis=0)
      num_3  = np.mean( np.delete( np.delete(pion0vbb_4quark_3_re, jidx, axis=0)[:,:,sep], sample, axis=0), axis=0)
      num_1p = np.mean( np.delete( np.delete(pion0vbb_4quark_1p_re, jidx, axis=0)[:,:,sep], sample, axis=0), axis=0)
      num_2p = np.mean( np.delete( np.delete(pion0vbb_4quark_2p_re, jidx, axis=0)[:,:,sep], sample, axis=0), axis=0)
      num_3p = np.mean( np.delete( np.delete(pion0vbb_4quark_3p_re, jidx, axis=0)[:,:,sep], sample, axis=0), axis=0)
      for t in range(0,T):
        jacks_1[sample,t]  = 2.0*m_jacks[sample] * num_1[t]  / pion_jjacks[sample, sep]
        jacks_2[sample,t]  = 2.0*m_jacks[sample] * num_2[t]  / pion_jjacks[sample, sep]
        jacks_3[sample,t]  = 2.0*m_jacks[sample] * num_3[t]  / pion_jjacks[sample, sep]
        jacks_1p[sample,t] = 2.0*m_jacks[sample] * num_1p[t] / pion_jjacks[sample, sep]
        jacks_2p[sample,t] = 2.0*m_jacks[sample] * num_2p[t] / pion_jjacks[sample, sep]
        jacks_3p[sample,t] = 2.0*m_jacks[sample] * num_3p[t] / pion_jjacks[sample, sep]
    weights_1  = np.sqrt(Ntraj-2.0)*np.std(jacks_1, axis=0, ddof=0)
    weights_2  = np.sqrt(Ntraj-2.0)*np.std(jacks_2, axis=0, ddof=0)
    weights_3  = np.sqrt(Ntraj-2.0)*np.std(jacks_3, axis=0, ddof=0)
    weights_1p = np.sqrt(Ntraj-2.0)*np.std(jacks_1p, axis=0, ddof=0)
    weights_2p = np.sqrt(Ntraj-2.0)*np.std(jacks_2p, axis=0, ddof=0)
    weights_3p = np.sqrt(Ntraj-2.0)*np.std(jacks_3p, axis=0, ddof=0)

    # Write to disk
    f = open(pion0vbb_4quark_stem_out + "_1." + str(sep) + "." + str(this_traj), 'w')
    for t in range(0,T):
      line = "{0:3d} {1:1.10e} {2:1.10e}".format(t, ratio_1[t], weights_1[t])
      print(line, file=f)
    f.close()

    f = open(pion0vbb_4quark_stem_out + "_2." + str(sep) + "." + str(this_traj), 'w')
    for t in range(0,T):
      line = "{0:3d} {1:1.10e} {2:1.10e}".format(t, ratio_2[t], weights_2[t])
      print(line, file=f)
    f.close()

    f = open(pion0vbb_4quark_stem_out + "_3." + str(sep) + "." + str(this_traj), 'w')
    for t in range(0,T):
      line = "{0:3d} {1:1.10e} {2:1.10e}".format(t, ratio_3[t], weights_3[t])
      print(line, file=f)
    f.close()

    f = open(pion0vbb_4quark_stem_out + "_1p." + str(sep) + "." + str(this_traj), 'w')
    for t in range(0,T):
      line = "{0:3d} {1:1.10e} {2:1.10e}".format(t, ratio_1p[t], weights_1p[t])
      print(line, file=f)
    f.close()

    f = open(pion0vbb_4quark_stem_out + "_2p." + str(sep) + "." + str(this_traj), 'w')
    for t in range(0,T):
      line = "{0:3d} {1:1.10e} {2:1.10e}".format(t, ratio_2p[t], weights_2p[t])
      print(line, file=f)
    f.close()

print("done.")
