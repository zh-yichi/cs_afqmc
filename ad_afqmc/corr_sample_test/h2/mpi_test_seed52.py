from functools import partial
from jax import random
#from jax import numpy as jnp
import argparse
#from mpi4py import MPI
import numpy as np
from ad_afqmc.corr_sample_test import corr_sample
from ad_afqmc import mpi_jax, config
import time

print = partial(print, flush=True)
nwalkers = 10
n_runs = 100
rlx_steps = 10
prop_steps = 0
dt = 0.005
n_exp_terms = 6
seed = 514825
cs = True

options = {
    "dt": dt,
    "n_exp_terms": n_exp_terms,
    "n_eql": 4,
    "n_ene_blocks": 1,
    "n_sr_blocks": 10,
    "n_blocks": 200,
    "n_walkers": nwalkers,
    "seed": seed,
    "walker_type": "rhf",
    "trial": "rhf",
    "corr_samp": cs,
    "free_proj": False,
}
import pickle
with open("options.pkl", "wb") as file:
    pickle.dump(options, file)


mo_file1="h2_mo1.npz"
chol_file1="h2_chol1"
mo_file2="h2_mo2.npz"
chol_file2="h2_chol2"


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--use_gpu", action="store_true")
#     args = parser.parse_args()

#     if args.use_gpu:
#         config.afqmc_config["use_gpu"] = True

config.setup_jax()
MPI = config.setup_comm()

init_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank
size = comm.Get_size()  # Total number of processes

ham_data1, ham1, prop1, trial1, wave_data1, sampler1, observable1, options1, _ \
    = mpi_jax._prep_afqmc(options,mo_file=mo_file1,chol_file=chol_file1)
ham_data2, ham2, prop2, trial2, wave_data2, sampler2, observable2, options2, _ \
    = mpi_jax._prep_afqmc(options,mo_file=mo_file2,chol_file=chol_file2)

prop_data1_init, ham_data1_init = \
    corr_sample.init_prop(ham_data1, ham1, prop1, trial1, wave_data1, seed, MPI)
prop_data2_init, ham_data2_init = \
    corr_sample.init_prop(ham_data2, ham2, prop2, trial2, wave_data2, seed, MPI)

prop_data1_init["imp_fun"] = np.zeros(nwalkers)
prop_data2_init["imp_fun"] = np.zeros(nwalkers)

### relaxation ###
comm.Barrier()
if rank == 0:
    print(f'# relaxation from mean-field object using {nwalkers*size} walkers')
    print('# rlx_step \t system1_en \t system2_en \t en_diff')
    print(f'    {0}'
          f'\t \t {prop_data1_init["e_estimate"]:.6f}' 
          f'\t {prop_data2_init["e_estimate"]:.6f}'
          f'\t {prop_data1_init["e_estimate"]-prop_data2_init["e_estimate"]:.6f}')
comm.Barrier()

(prop_data1_rlx,prop_data2_rlx),(loc_en1,loc_weight1,loc_en2,loc_weight2) \
    = corr_sample.cs_steps_scan(rlx_steps,
                                prop_data1_init,ham_data1_init,prop1,trial1,wave_data1,
                                prop_data2_init,ham_data2_init,prop2,trial2,wave_data2, 
                                )

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
    glb_en1 = glb_en1.reshape(size,rlx_steps).T
    glb_en2 = glb_en2.reshape(size,rlx_steps).T
    glb_weight1 = glb_weight1.reshape(size,rlx_steps).T
    glb_weight2 = glb_weight2.reshape(size,rlx_steps).T

    energy1 = np.zeros((rlx_steps))
    energy2 = np.zeros((rlx_steps))
    en_diff = np.zeros((rlx_steps))
    
    for step in range(rlx_steps):

        energy1[step] = sum(glb_en1[step,:])/sum(glb_weight1[step,:])
        energy2[step] = sum(glb_en2[step,:])/sum(glb_weight2[step,:])
        en_diff[step] = energy1[step] - energy2[step]

        print(f'    {step+1} \t \t {energy1[step]:.6f} \t {energy2[step]:.6f} \t {en_diff[step]:.6f}')
    
    now_time = time.time()
    print(f'# relaxation time: {now_time - init_time:.2f}')
comm.Barrier()


walker_en1 = corr_sample.en_samples(prop_data1_rlx,ham_data1_init,prop1,trial1,wave_data1)
walker_en2 = corr_sample.en_samples(prop_data2_rlx,ham_data2_init,prop1,trial1,wave_data1)
walker_wt1 = prop_data1_rlx["weights"]
walker_wt2 = prop_data2_rlx["weights"]
olp1 = prop_data1_rlx["overlaps"]
olp2 = prop_data2_rlx["overlaps"]
imp1 = prop_data1_rlx["imp_fun"]
imp2 = prop_data2_rlx["imp_fun"]

comm.Barrier()
if rank == 0:
    print()
    print('# walkers analyze of the last step')
    glb_walker_en1 = np.empty(size*walker_en1.size,dtype='float32')
    glb_walker_en2 = np.empty(size*walker_en2.size,dtype='float32')
    glb_walker_wt1 = np.empty(size*walker_wt1.size,dtype='float32')
    glb_walker_wt2 = np.empty(size*walker_wt2.size,dtype='float32')
    glb_olp1 = np.empty(size*olp1.size,dtype=np.complex64)
    glb_olp2 = np.empty(size*olp2.size,dtype=np.complex64)
    glb_imp1 = np.empty(size*imp1.size,dtype='float32')
    glb_imp2 = np.empty(size*imp2.size,dtype='float32')
else:
    glb_walker_en1 = None
    glb_walker_en2 = None
    glb_walker_wt1 = None
    glb_walker_wt2 = None
    glb_olp1 = None
    glb_olp2 = None
    glb_imp1 = None
    glb_imp2 = None
comm.Barrier()

walker_en1 = np.asarray(walker_en1,dtype='float32')
walker_en2 = np.asarray(walker_en2,dtype='float32')
walker_wt1 = np.asarray(walker_wt1,dtype='float32')
walker_wt2 = np.asarray(walker_wt2,dtype='float32')
olp1 = np.asarray(olp1,dtype=np.complex64)
olp2 = np.asarray(olp2,dtype=np.complex64)
imp1 = np.asarray(imp1,dtype='float32')
imp2 = np.asarray(imp2,dtype='float32')
comm.Gather(walker_en1,glb_walker_en1,root=0)
comm.Gather(walker_en2,glb_walker_en2,root=0)
comm.Gather(walker_wt1,glb_walker_wt1,root=0)
comm.Gather(walker_wt2,glb_walker_wt2,root=0)
comm.Gather(olp1,glb_olp1,root=0)
comm.Gather(olp2,glb_olp2,root=0)
comm.Gather(imp1,glb_imp1,root=0)
comm.Gather(imp2,glb_imp2,root=0)

comm.Barrier()
if rank == 0:
    en1 = sum(glb_walker_en1*glb_walker_wt1)/sum(glb_walker_wt1)
    en2 = sum(glb_walker_en2*glb_walker_wt2)/sum(glb_walker_wt2)
    print(f'# the energy system1 is {en1:.6f}')
    print(f'# the energy system2 is {en2:.6f}')
    print('# walker \t system1_en \t system1_wt \t \t overlap1 \t imp1 \t system2_en \t system2_wt \t \t overlap2 \t imp2')
    for n in range(len(glb_walker_en1)):
        print(f'  {n+1} \t \t'
              f'  {glb_walker_en1[n]:.6f} \t {glb_walker_wt1[n]:.6f} \t' 
              f'  {glb_olp1[n].real:.6f}+{glb_olp1[n].imag:.6f}j \t {glb_imp1[n]:.6f}' 
              f'  {glb_walker_en2[n]:.6f} \t {glb_walker_wt2[n]:.6f} \t '
              f'  {glb_olp2[n].real:.6f}+{glb_olp2[n].imag:.6f}j \t {glb_imp2[n]:.6f}')
comm.Barrier()



### post relaxation propagation ###
# comm.Barrier()
# if rank == 0:
#     print()
#     print(f'# multiple independent post relaxation propagation with step size {dt}s')
#     if options["corr_samp"]:
#         print('# correlated sampling')
#     else: print('# uncorrelated sampling')

#     print(f'# tot_walkers: {nwalkers*size}, propagation steps: {prop_steps}, number of independent runs: {n_runs}')
#     print('# step' 
#           '\t system1_en \t error' 
#           '\t \t system2_en \t error'
#           '\t \t energy_diff \t error')
    
# comm.Barrier()

# if options["corr_samp"]:
#     seeds = random.randint(random.PRNGKey(options["seed"]),
#                         shape=(n_runs,), minval=0, maxval=10000*n_runs)
#     comm.Barrier()
#     if rank == 0:
#         print(seeds[51])
#     comm.Barrier()

#     loc_en1,loc_weight1,loc_en2,loc_weight2 \
#     = corr_sample.scan_seeds(seeds,prop_steps,
#                              prop_data1_rlx,ham_data1_init,prop1,trial1,wave_data1,
#                              prop_data2_rlx,ham_data2_init,prop2,trial2,wave_data2, 
#                              MPI)
# else:
#     seeds = random.randint(random.PRNGKey(options["seed"]),
#                         shape=(n_runs,2), minval=0, maxval=10000*n_runs)

#     loc_en1,loc_weight1,loc_en2,loc_weight2 \
#         = corr_sample.ucs_scan_seeds(seeds,prop_steps,
#                                      prop_data1_rlx,ham_data1_init,prop1,trial1,wave_data1,
#                                      prop_data2_rlx,ham_data2_init,prop2,trial2,wave_data2, 
#                                      MPI)

# comm.Barrier()
# if rank == 0:
#     glb_en1 = np.empty(size*loc_en1.size,dtype='float32')
#     glb_en2 = np.empty(size*loc_en2.size,dtype='float32')
#     glb_weight1 = np.empty(size*loc_weight2.size,dtype='float32')
#     glb_weight2 = np.empty(size*loc_weight2.size,dtype='float32')
# else:
#     glb_en1 = None
#     glb_en2 = None
#     glb_weight1 = None
#     glb_weight2 = None
# comm.Barrier()

# loc_en1 = np.asarray(loc_en1,dtype='float32')
# loc_en2 = np.asarray(loc_en2,dtype='float32')
# loc_weight1 = np.asarray(loc_weight1,dtype='float32')
# loc_weight2 = np.asarray(loc_weight2,dtype='float32')

# comm.Gather(loc_en1,glb_en1,root=0)
# comm.Gather(loc_en2,glb_en2,root=0)
# comm.Gather(loc_weight1,glb_weight1,root=0)
# comm.Gather(loc_weight2,glb_weight2,root=0)


# comm.Barrier()
# if rank == 0:
#     glb_en1 = glb_en1.reshape(size,n_runs,prop_steps).T
#     glb_en2 = glb_en2.reshape(size,n_runs,prop_steps).T
#     glb_weight1 = glb_weight1.reshape(size,n_runs,prop_steps).T
#     glb_weight2 = glb_weight2.reshape(size,n_runs,prop_steps).T

#     energy1 = np.zeros((prop_steps,n_runs))
#     energy2 = np.zeros((prop_steps,n_runs))
#     en_diff = np.zeros((prop_steps,n_runs))

#     for step in range(prop_steps):

#         for run in range(n_runs):
#             energy1[step,run] = sum(glb_en1[step,run,:])/sum(glb_weight1[step,run,:])
#             energy2[step,run] = sum(glb_en2[step,run,:])/sum(glb_weight2[step,run,:])
#             en_diff[step,run] = energy1[step,run] - energy2[step,run]

#         en_mean1 = energy1[step,:].mean()
#         en_mean2 = energy2[step,:].mean()
#         en_diff_mean = en_diff[step,:].mean()
#         en_err1 = energy1[step,:].std()/np.sqrt(n_runs)
#         en_err2 = energy2[step,:].std()/np.sqrt(n_runs)
#         en_diff_mean_err = en_diff[step,:].std()/np.sqrt(n_runs)
#         print(f'  {step+1}'
#               f'\t {en_mean1:.6f} \t {en_err1:.6f}' 
#               f'\t {en_mean2:.6f} \t {en_err2:.6f}'
#               f'\t {en_diff_mean:.6f} \t {en_diff_mean_err:.6f}')

#     print('# debug the big uncertainty step')
#     print('# run \t sys1_en \t sys2_en \t en_diff')
#     for run in range(n_runs):
#         print(f'  {run+1} \t {energy1[prop_steps-1,run]:.6f} \t {energy2[prop_steps-1,run]:.6f} \t {en_diff[prop_steps-1,run]:.6f}')

#     print(f'# system1 energy: {energy1[prop_steps-1,:].mean():.6f}')
#     print(f'# system2 energy: {energy2[prop_steps-1,:].mean():.6f}')
#     print(f'# energy difference: {en_diff[prop_steps-1,:].mean():.6f}')

#     end_time = time.time()
#     print(f'# total run time: {end_time - init_time:.2f}')
# comm.Barrier()
