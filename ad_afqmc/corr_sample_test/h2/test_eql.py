from functools import partial
from jax import random
from jax import numpy as jnp
import jax
import psutil
import os

print = partial(print, flush=True)
nwalkers = 20
n_runs = 10
eql_steps = 6
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
amp_file1="amp1.npz"
chol_file1="h2_chol1"
mo_file2="h2_mo2.npz"
amp_file2="amp2.npz"
chol_file2="h2_chol2"

from mpi4py import MPI
import numpy as np
from ad_afqmc.corr_sample_test import corr_sample
from ad_afqmc import mpi_jax
import time

init_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank
size = comm.Get_size()  # Total number of processes

ham_data1, ham1, prop1, trial1, wave_data1, sampler1, observable1, options1, _ \
    = mpi_jax._prep_afqmc(options1,mo_file=mo_file1,amp_file=amp_file1,chol_file=chol_file1)
ham_data2, ham2, prop2, trial2, wave_data2, sampler2, observable2, options2, _ \
    = mpi_jax._prep_afqmc(options2,mo_file=mo_file2,amp_file=amp_file2,chol_file=chol_file2)


seeds1 = random.randint(random.PRNGKey(options1["seed"]), shape=(n_runs,), minval=0, maxval=1000)
seeds2 = random.randint(random.PRNGKey(options2["seed"]), shape=(n_runs,), minval=0, maxval=1000)

comm.Barrier()
if rank == 0:
    print('# test equilibrium')
    print(f'tot_walkers: {nwalkers*size}, eql_steps: {eql_steps}')
    print('n_run \t system1_en \t system2_en \t   en_diff \t totol_time \t memory_usage(MB)')
    energy1 = np.zeros(n_runs)
    energy2 = np.zeros(n_runs)
    en_diff = np.zeros(n_runs)
comm.Barrier()


for i in range(n_runs):

    options1["seed"] = seeds1[i]
    options2["seed"] = seeds2[i]
    prop_data1, ham_data1 = corr_sample.init_prop(ham_data1, ham1, prop1, trial1, wave_data1, options1, MPI)
    prop_data2, ham_data2 = corr_sample.init_prop(ham_data2, ham2, prop2, trial2, wave_data2, options2, MPI)

    (prop_data1,prop_data2) \
        = corr_sample.cs_steps_scan(
            eql_steps,prop_data1,ham_data1,prop1,trial1,wave_data1,prop_data2,ham_data2,prop2,trial2,wave_data2
            )
        
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
    comm.Barrier()

    comm.Gather(loc_en_sample1,en_sample1,root=0)
    comm.Gather(loc_en_sample2,en_sample2,root=0)
    comm.Gather(loc_weight1,weight1,root=0)
    comm.Gather(loc_weight2,weight2,root=0)

    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss  # in bytes
    all_memory_usages = comm.gather(memory_usage, root=0)

    comm.Barrier()
    if rank == 0:
        norm_weight1 = weight1/sum(weight1)
        norm_weight2 = weight2/sum(weight2)
        weight_en_sample1 = en_sample1*norm_weight1
        weight_en_sample2 = en_sample2*norm_weight2
        weight_en_diff_sample = weight_en_sample1 - weight_en_sample2
        #weight_en_diff_err = weight_en_diff_sample.std()
        #energy1_err = weight_en_sample1.std()
        #energy2_err = weight_en_sample2.std()
        energy1[i] = sum(weight_en_sample1)
        energy2[i] = sum(weight_en_sample2)
        en_diff[i] = sum(weight_en_diff_sample)
        total_memory_usage = sum(all_memory_usages)
        end_time = time.time()
        print(f'{i+1} \t  {energy1[i]:.6f} \t {energy2[i]:.6f} \t  {en_diff[i]:.6f} \t   {end_time-init_time:.2f} \t {total_memory_usage/ 1024 ** 2}')
    comm.Barrier()
    #jax.clear_backends_cache()

comm.Barrier()
if rank == 0:
    en_mean1 = energy1.mean()
    en_mean2 = energy2.mean()
    en_diff_mean = en_diff.mean()
    en_err1 = energy1.std()
    en_err2 = energy2.std()
    en_diff_err = en_diff.std()
    print(f'averaged over {n_runs}')
    print(f'system1 energy: {en_mean1}, error: {en_err1}')
    print(f'system2 energy: {en_mean2}, error: {en_err2}')
    print(f'cs energy difference: {en_diff_mean}, error: {en_diff_err}')
comm.Barrier()
