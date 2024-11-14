from functools import partial
from jax import random
from jax import numpy as jnp

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


seeds = random.randint(random.PRNGKey(options1["seed"]),
                        shape=(n_runs,), minval=0, maxval=1000)

comm.Barrier()
if rank == 0:
    print('# test equilibrium')
    print(f'tot_walkers: {nwalkers*size}, eql_steps: {eql_steps}')
    print('n_run \t system1_en \t system2_en \t en_diff')
    energy1 = np.zeros(n_runs)
    energy2 = np.zeros(n_runs)
    en_diff = np.zeros(n_runs)
comm.Barrier()

prop_data1_init, ham_data1_init = \
    corr_sample.init_prop(ham_data1, ham1, prop1, trial1, wave_data1, 2, MPI)
prop_data2_init, ham_data2_init = \
    corr_sample.init_prop(ham_data2, ham2, prop2, trial2, wave_data2, 98, MPI)

loc_en1,loc_weight1,loc_en2,loc_weight2 = corr_sample.scan_seeds(seeds,6,
               prop_data1_init,ham_data1_init,prop1,trial1,wave_data1,
               prop_data2_init,ham_data2_init,prop2,trial2,wave_data2, 
               MPI)

comm.Barrier()
if rank == 0:
    glb_en1 = np.empty(size*loc_en1.size,dtype='float32')
    glb_en2 = np.empty(size*loc_en2.size,dtype='float32')
    glb_weight1 = np.empty(size*loc_weight2.size,dtype='float32')
    glb_weight2 = np.empty(size*loc_weight2.size,dtype='float32')
else:
    glb_en1 = None
    glb_en2 = None
    glb_weight1 = None
    glb_weight2 = None
comm.Barrier()

loc_en1 = np.asarray(loc_en1,dtype='float32')
loc_en2 = np.asarray(loc_en2,dtype='float32')
loc_weight1 = np.asarray(loc_weight1,dtype='float32')
loc_weight2 = np.asarray(loc_weight2,dtype='float32')

comm.Gather(loc_en1,glb_en1,root=0)
comm.Gather(loc_en2,glb_en2,root=0)
comm.Gather(loc_weight1,glb_weight1,root=0)
comm.Gather(loc_weight2,glb_weight2,root=0)

comm.Barrier()
if rank == 0:
    glb_en1 = glb_en1.reshape(size,n_runs).T
    glb_en2 = glb_en2.reshape(size,n_runs).T
    glb_weight1 = glb_weight1.reshape(size,n_runs).T
    glb_weight2 = glb_weight2.reshape(size,n_runs).T

    for i in range(n_runs):
        energy1[i] = sum(glb_en1[i])/sum(glb_weight1[i])
        energy2[i] = sum(glb_en2[i])/sum(glb_weight2[i])
        en_diff[i] = energy1[i] - energy2[i]
        print(f'{i} \t {energy1[i]:.6f} \t {energy2[i]:.6f} \t {en_diff[i]:.6f}')

comm.Barrier()


comm.Barrier()
if rank == 0:
    en_mean1 = energy1.mean()
    en_mean2 = energy2.mean()
    en_diff_mean = en_diff.mean()
    en_err1 = energy1.std()
    en_err2 = energy2.std()
    en_diff_err = en_diff.std()
    print(f'averaged over {n_runs}')
    print(f'system1 energy: {en_mean1:.6f}, error: {en_err1:.6f}')
    print(f'system2 energy: {en_mean2:.6f}, error: {en_err2:.6f}')
    print(f'cs energy difference: {en_diff_mean:.6f}, error: {en_diff_err:.6f}')
    end_time = time.time()
    print(f'total run time: {end_time - init_time:.2f}')
comm.Barrier()
