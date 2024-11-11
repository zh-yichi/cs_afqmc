from functools import partial
from jax import random

print = partial(print, flush=True)
nwalkers = 20
options1 = {
    "dt": 0.005,
    "n_eql": 4,
    "n_ene_blocks": 1,
    "n_sr_blocks": 10,
    "n_blocks": 200,
    "n_walkers": nwalkers,
    "seed": 98,
    "walker_type": "rhf",
    "trial": "rhf",
}

options2 = {
    "dt": 0.005,
    "n_eql": 4,
    "n_ene_blocks": 1,
    "n_sr_blocks": 10,
    "n_blocks": 200,
    "n_walkers": nwalkers,
    "seed": 2,
    "walker_type": "rhf",
    "trial": "rhf",
}

mo_file1="h2_mo1.npz"
chol_file1="h2_chol1"
mo_file2="h2_mo2.npz"
chol_file2="h2_chol2"

from mpi4py import MPI
import numpy as np
from ad_afqmc.corr_sample_test import corr_sample
from ad_afqmc import mpi_jax
import time


comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank
size = comm.Get_size()  # Total number of processes

ham_data1, ham1, prop1, trial1, wave_data1, sampler1, observable1, options1, _ \
    = mpi_jax._prep_afqmc(options1,mo_file=mo_file1,chol_file=chol_file1)
ham_data2, ham2, prop2, trial2, wave_data2, sampler2, observable2, options2, _ \
    = mpi_jax._prep_afqmc(options2,mo_file=mo_file2,chol_file=chol_file2)

#print(f'# rank = {rank} size = {size}')
#print(f'# random key1 {prop_data1["key"]} random key2 {prop_data2["key"]}')

loc_steps = 100

init_time = time.time()
if rank == 0:
    print('# test err -- correlated sampling')
    print('n_step n_walkers   system1_en \t err \t'
            '\t system2_en \t err \t \t en_diff \t err \t  totol_time')

prop_data1, ham_data1 = corr_sample.init_prop(ham_data1, ham1, prop1, trial1, wave_data1, options1, MPI)
prop_data2, ham_data2 = corr_sample.init_prop(ham_data2, ham2, prop2, trial2, wave_data2, options2, MPI)

for step in range(loc_steps):
    prop_data1,prop_data2,_ = corr_sample.cs_block_scan(prop_data1,ham_data1,prop1,trial1,wave_data1,prop_data2,ham_data2,prop2,trial2,wave_data2)
    
    loc_en_sample1 = corr_sample.en_samples(prop_data1,ham_data1,prop1,trial1,wave_data1)
    loc_en_sample2 = corr_sample.en_samples(prop_data2,ham_data2,prop2,trial2,wave_data2)
    loc_weight1 = prop_data1["weights"]
    loc_weight2 = prop_data2["weights"]
    loc_en_sample1 = np.asarray(loc_en_sample1,dtype='float32')
    loc_en_sample2 = np.asarray(loc_en_sample2,dtype='float32')
    loc_weight1 = np.asarray(loc_weight1,dtype='float32')
    loc_weight2 = np.asarray(loc_weight2,dtype='float32')

    comm.Barrier()
    if rank == 0:
        en_sample1 = np.empty(size*loc_en_sample1.size,dtype='float32')
        en_sample2 = np.empty(size*loc_en_sample2.size,dtype='float32')
        weight1 = np.empty(size*loc_weight2.size,dtype='float32')
        weight2 = np.empty(size*loc_weight2.size,dtype='float32')
    else:
        en_sample1 = None
        en_sample2 = None
        weight1 = None
        weight2 = None

    comm.Gather(loc_en_sample1,en_sample1,root=0)
    comm.Gather(loc_en_sample2,en_sample2,root=0)
    comm.Gather(loc_weight1,weight1,root=0)
    comm.Gather(loc_weight2,weight2,root=0)

    comm.Barrier()
    if rank == 0:
        norm_weight1 = weight1/sum(weight1)
        norm_weight2 = weight2/sum(weight2)
        weight_en_sample1 = en_sample1*norm_weight1
        weight_en_sample2 = en_sample2*norm_weight2
        weight_en_diff_sample = weight_en_sample1 - weight_en_sample2
        weight_en_diff_err = weight_en_diff_sample.std()
        energy1_err = weight_en_sample1.std()
        energy2_err = weight_en_sample2.std()
        energy1 = sum(weight_en_sample1)
        energy2 = sum(weight_en_sample2)
        en_diff = sum(weight_en_diff_sample)
        
        end_time = time.time()
        print(f'{step+1} \t {en_sample1.size} \t  {energy1:.6f} \t {energy1_err:.6f} \t'
            f' {energy2:.6f} \t {energy2_err:.6f} \t {en_diff:.6f} \t {weight_en_diff_err:.6f}   {end_time-init_time:.2f}')


init_time = time.time()
if rank == 0:
    print('# test err -- uncorrelated sampling')
    print('n_steps n_walkers   system1_en \t err \t'
            '\t system2_en \t err \t \t en_diff \t err \t  totol_time')

prop_data1, ham_data1 = corr_sample.init_prop(ham_data1, ham1, prop1, trial1, wave_data1, options1, MPI)
prop_data2, ham_data2 = corr_sample.init_prop(ham_data2, ham2, prop2, trial2, wave_data2, options2, MPI)

sr_steps = 5 #do sr every sr_steps
for step in range(loc_steps):
    prop_data1,prop_data2,_,_ = corr_sample.ucs_block_scan(prop_data1,ham_data1,prop1,trial1,wave_data1,prop_data2,ham_data2,prop2,trial2,wave_data2)
    if step % sr_steps == sr_steps-1:
        prop_data1 = prop1.stochastic_reconfiguration_local(prop_data1)
        prop_data2 = prop2.stochastic_reconfiguration_local(prop_data2)

    loc_en_sample1 = corr_sample.en_samples(prop_data1,ham_data1,prop1,trial1,wave_data1)
    loc_en_sample2 = corr_sample.en_samples(prop_data2,ham_data2,prop2,trial2,wave_data2)
    loc_weight1 = prop_data1["weights"]
    loc_weight2 = prop_data2["weights"]
    loc_en_sample1 = np.asarray(loc_en_sample1,dtype='float32')
    loc_en_sample2 = np.asarray(loc_en_sample2,dtype='float32')
    loc_weight1 = np.asarray(loc_weight1,dtype='float32')
    loc_weight2 = np.asarray(loc_weight2,dtype='float32')

    comm.Barrier()
    if rank == 0:
        en_sample1 = np.empty(size*loc_en_sample1.size,dtype='float32')
        en_sample2 = np.empty(size*loc_en_sample2.size,dtype='float32')
        weight1 = np.empty(size*loc_weight2.size,dtype='float32')
        weight2 = np.empty(size*loc_weight2.size,dtype='float32')
    else:
        en_sample1 = None
        en_sample2 = None
        weight1 = None
        weight2 = None

    comm.Gather(loc_en_sample1,en_sample1,root=0)
    comm.Gather(loc_en_sample2,en_sample2,root=0)
    comm.Gather(loc_weight1,weight1,root=0)
    comm.Gather(loc_weight2,weight2,root=0)

    comm.Barrier()
    if rank == 0:
        norm_weight1 = weight1/sum(weight1)
        norm_weight2 = weight2/sum(weight2)
        weight_en_sample1 = en_sample1*norm_weight1
        weight_en_sample2 = en_sample2*norm_weight2
        weight_en_diff_sample = weight_en_sample1 - weight_en_sample2
        weight_en_diff_err = weight_en_diff_sample.std()
        energy1_err = weight_en_sample1.std()
        energy2_err = weight_en_sample2.std()
        energy1 = sum(weight_en_sample1)
        energy2 = sum(weight_en_sample2)
        en_diff = sum(weight_en_diff_sample)
        end_time = time.time()
        print(f'{step+1} \t {en_sample1.size} \t  {energy1:.6f} \t {energy1_err:.6f} \t'
            f' {energy2:.6f} \t {energy2_err:.6f} \t {en_diff:.6f} \t {weight_en_diff_err:.6f}   {end_time-init_time:.2f}')